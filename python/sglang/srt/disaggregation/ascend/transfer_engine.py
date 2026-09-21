import logging
from typing import List

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.environ import envs
from sglang.srt.utils.network import NetworkAddress

try:
    from memfabric_hybrid import TransferEngine

    import_error = None
except ImportError as e:
    import_error = e

logger = logging.getLogger(__name__)

_DEFAULT_PROTOCOL = "sdma"

# MemFabric binds nic base_port + its group rank (0..7), so each worker owns a
# stride of ports. Decode and Prefill get disjoint strides so both roles can
# run on the same host (up to 16 workers per role per host).
_WORKER_PORT_STRIDE = 8
_ROLE_PORT_STRIDE = 128


class AscendTransferEngine(MooncakeTransferEngine):
    def __init__(
        self,
        hostname: str,
        npu_id: int,
        disaggregation_mode: DisaggregationMode,
    ):
        if import_error is not None:
            logger.warning(
                "Please install memfabric_hybrid, for details, see docs/docs/advanced_features/pd_disaggregation.mdx"
            )
            raise import_error

        self.engine = TransferEngine()
        self.hostname = hostname
        self.npu_id = npu_id

        # Only the protocol prefix of the store URL participates in linking;
        # each engine hosts/addresses the store at its own session ip:port.
        self.store_url = envs.ASCEND_MF_STORE_URL.get()
        if not self.store_url:
            raise ValueError(
                "ASCEND_MF_STORE_URL (e.g. tcp://<ip>:<port>) must be set for "
                "the ascend disaggregation transfer backend."
            )
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = "Prefill"
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self.role = "Decode"
        else:
            raise ValueError(f"Unsupported DisaggregationMode: {disaggregation_mode}")
        # Session port 0 lets memfabric pick a free one; refetched after init.
        self.session_id = NetworkAddress(self.hostname, 0).to_host_port_str()
        self.initialize()
        rpc_port = self.engine.get_rpc_port()
        self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()

    def initialize(self) -> None:
        from sglang.srt.distributed.parallel_state import (
            get_world_group,
            get_world_size,
        )

        transfer_protocol = self._get_transfer_protocol()
        nic = ""
        if transfer_protocol == "device_rdma":
            # with device RDMA for PD transfer: initialize hccl in advance
            # through all_gather to avoid conflicts with rdma initialization.
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [
                torch.empty_like(tmp_tensor) for _ in range(get_world_size())
            ]
            torch.distributed.all_gather(
                output_tensor_list, tmp_tensor, group=get_world_group().device_group
            )
        elif transfer_protocol == "host_rdma":
            nic = self._resolve_worker_hcom_url(
                envs.ASCEND_MF_HCOM_URL.get(),
                self.role,
                get_world_group().rank_in_group,
            )
        ret_value = self.engine.initialize(
            store_url=self.store_url,
            session_id=self.session_id,
            # memfabric requires the PD role here ("Prefill"=sender /
            # "Decode"=receiver); the store server role stays "Decode".
            role=self.role,
            device_id=self.npu_id,
            data_op_type=self._resolve_trans_op_type(transfer_protocol),
            nic=nic,
        )
        if ret_value != 0:
            logger.error("Ascend Transfer Engine initialization failed.")
            raise RuntimeError("Ascend Transfer Engine initialization failed.")

    def batch_register(self, ptrs: List[int], lengths: List[int]):
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception:
            # Mark register as failed
            ret_value = -1
        if ret_value != 0:
            logger.debug(f"Ascend memory registration for ptr {ptrs} failed.")

    @staticmethod
    def _get_transfer_protocol() -> str:
        protocol = envs.ASCEND_MF_TRANSFER_PROTOCOL.get()
        return protocol.strip().lower() if protocol else _DEFAULT_PROTOCOL

    @staticmethod
    def _resolve_worker_hcom_url(hcom_url: str, role: str, world_rank: int) -> str:
        if not hcom_url:
            raise ValueError(
                "ASCEND_MF_HCOM_URL (tcp://<rdma-nic-ip>:<port>) is required "
                "for host_rdma; memfabric would otherwise bind a loopback "
                "endpoint that peers cannot reach."
            )

        address, separator, port_str = hcom_url.rpartition(":")
        if not separator or not address.startswith("tcp://"):
            raise ValueError(f"Invalid port in ASCEND_MF_HCOM_URL: {hcom_url!r}")

        try:
            base_port = int(port_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid port in ASCEND_MF_HCOM_URL: {hcom_url!r}"
            ) from exc

        role_offset = 0 if role == "Decode" else _ROLE_PORT_STRIDE
        worker_port = base_port + role_offset + world_rank * _WORKER_PORT_STRIDE
        if not (1024 <= worker_port and worker_port + _WORKER_PORT_STRIDE - 1 <= 65535):
            raise ValueError(
                "Resolved ASCEND_MF_HCOM_URL port is out of range: "
                f"base_port={base_port}, world_rank={world_rank}, "
                f"role={role}, "
                f"resolved_port_range={worker_port}-{worker_port + _WORKER_PORT_STRIDE - 1}"
            )

        worker_hcom_url = f"{address}:{worker_port}"
        logger.info(
            "Resolved Ascend Host RDMA endpoint: role=%s, world_rank=%d, "
            "base=%s, endpoint=%s",
            role,
            world_rank,
            hcom_url,
            worker_hcom_url,
        )
        return worker_hcom_url

    @staticmethod
    def _resolve_trans_op_type(protocol: str):
        op_type = getattr(TransferEngine.TransDataOpType, protocol.upper(), None)
        if op_type is None:
            logger.warning(
                "Transfer protocol %r is not supported by the installed "
                "memfabric_hybrid, falling back to %r.",
                protocol,
                _DEFAULT_PROTOCOL,
            )
            op_type = TransferEngine.TransDataOpType.SDMA
        return op_type
