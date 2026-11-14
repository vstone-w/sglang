"""NPU device operations for the SRT platform layer."""

from typing import Any, Optional

import torch

from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform


class NPUDeviceMixin(DeviceMixin):
    """NPU implementation of the shared device operations."""

    _enum: PlatformEnum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(torch.npu.get_device_properties(device_id).total_memory)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        return float(torch.npu.max_memory_allocated(device))

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("npu", local_rank)

    def set_device(self, device: "torch.device") -> None:
        torch.npu.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return str(torch.npu.get_device_name(device_id))

    def get_device_uuid(self, device_id: int = 0) -> str:
        return str(torch.npu.get_device_properties(device_id).uuid)

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability:
        # The return value of torch_npu.npu.get_device_capability() is configured
        # via the environment variable TORCH_NPU_DEVICE_CAPABILITY, which is only
        # used for compatibility with native PyTorch and does not represent the
        # actual capabilities of the NPU hardware
        return DeviceCapability(0, 0)

    def empty_cache(self) -> None:
        torch.npu.empty_cache()

    def synchronize(self) -> None:
        torch.npu.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        return torch.npu.mem_get_info(device_id)

    def is_pin_memory_available(self, device=None) -> bool:
        if device is not None and str(device) == "cpu":
            return False
        return True

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            super().seed_everything(seed)
            if hasattr(torch, "npu"):
                torch.npu.manual_seed_all(seed)


class NPUSRTPlatform(NPUDeviceMixin, SRTPlatform):
    """Default in-tree NPU SRT platform."""

    capabilities = PlatformCapabilities(
        supports_triton=True,
        graph_capture=True,
    )

    def get_default_attention_backend(self) -> str:
        return "ascend"

    def get_dispatch_key_name(self) -> str:
        return "npu"

    def _supports_dsv41_swa_bounded_replay(self, model_type: str) -> bool:
        if model_type != "deepseek_v41":
            return False
        # A3 and A5 (arch35) share the row-based window implementation; the
        # arch35 layouts (fp8/fp4 rows) are selected by the pool.
        return True

    def supports_decoder_swa_bounded_replay(self, *, model_type: str) -> bool:
        return self._supports_dsv41_swa_bounded_replay(model_type)

    def supports_encoder_swa_bounded_replay(self, *, model_type: str) -> bool:
        return self._supports_dsv41_swa_bounded_replay(model_type)

    def decoder_replay_dp_moe_backends(self) -> tuple[str, ...]:
        return ("none", "deepep")

    def create_decoder_replay_tail(self, model):
        from sglang.srt.hardware_backend.npu.dsv4.npu_decoder_replay_tail import (
            NpuDecoderReplayTail,
        )

        return NpuDecoderReplayTail.maybe_create(model)

    def _dsv4_backend_module(self):
        """A5 uses the upstream backend; A3 keeps the fork's validated one."""
        from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

        if is_npu_arch35():
            from sglang.srt.hardware_backend.npu.attention import (
                ascend_dsv4_backend as module,
            )
        else:
            from sglang.srt.hardware_backend.npu.attention import (
                ascend_dsv4_backend_a3 as module,
            )
        return module

    def get_dsv4_attention_backend_cls(self):
        return self._dsv4_backend_module().DeepseekV4AscendAttnBackend

    def get_dsv4_draft_backend_cls(self):
        return self._dsv4_backend_module().DeepseekV4AscendMultiStepDraftBackend

    def supports_dsv41_bf16_kv_rows(self) -> bool:
        from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

        return not is_npu_arch35()

    def dsv4_kernel_num_heads(self, layer, num_tokens: int):
        if not layer.is_dsv41:
            return None
        from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

        # A3 BF16 SMLA accepts the per-rank width (e.g. TP16/DP8: 32).
        # The A5 mixed-quant kernel retains its 64-head padding.
        return None if is_npu_arch35() else layer.n_local_heads

    def request_window_nbytes(
        self,
        *,
        num_slots: int,
        layers: int,
        page_size: int,
        capacity: int,
        workspace_rows: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        draft_ring_size: int,
        draft_ring_layers: int,
    ) -> Optional[int]:
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_kv_layout import (
            NPUDSV4KVLayout,
            npu_pool_pages,
        )
        from sglang.srt.hardware_backend.npu.dsv4.npu_request_window import (
            NPURequestWindowSpec,
        )
        from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

        row_layout = (
            NPUDSV4KVLayout.V41_SWA_FP8 if is_npu_arch35() else NPUDSV4KVLayout.V41_BF16
        )
        row_dim, row_dtype = row_layout.kv_info(qk_nope_head_dim, qk_rope_head_dim)
        spec = NPURequestWindowSpec.create(
            num_slots=num_slots,
            layers=layers,
            page_size=page_size,
            capacity=capacity,
            workspace_rows=workspace_rows,
            row_dim=row_dim,
            row_dtype=row_dtype,
        )
        nbytes = spec.nbytes
        if draft_ring_size and draft_ring_layers:
            # Share the pool's page formula so the budget matches the
            # allocation (whole pages plus one spare).
            ring_rows = npu_pool_pages(num_slots * draft_ring_size, page_size)
            row_bytes = row_dim * row_dtype.itemsize
            nbytes += ring_rows * page_size * draft_ring_layers * row_bytes
        return nbytes

    def validate_encoder_swa_bounded_replay_args(self, cfg: Any) -> None:
        # The shared resolver owns the model-structure and speculative checks;
        # this hook only states what has actually been validated on NPU.
        # The A3 capture-time metadata fault is fixed by initializing HCCP
        # before the first custom AICPU metadata kernel.
        # Prefill builds one compact window group per request (grouped by req),
        # so a DP rank may host several running requests; the pool and ring
        # slots are sized from --max-running-requests.
        #
        # PD: the prefill side sends the request window's ring pages through
        # ``resume_units`` (``NPURequestWindow.page_indices``); the decode side
        # receives them into its paged SWA pages, so the two roles share one
        # positional layout. Page sizes are checked by the PD handshake. The
        # decode role does not enable encoder replay -- the shared resolver's
        # ``PD decode`` check rejects it; decode only receives the window state.
        return
