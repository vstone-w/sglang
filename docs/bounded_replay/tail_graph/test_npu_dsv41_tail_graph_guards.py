"""CPU contract checks for the NPU decoder tail-graph guards.

AST-extracted from the production sources so no torch_npu or ACL is needed.
These pin the three guards a replay step depends on: the padded request count
that keys a graph, the MoE transport's capturability, and the SWA sink row that
dead padding writes land on.
"""

import ast
import copy
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.dsv4.ring_layout import sink_dead_rows

ROOT = Path(__file__).resolve().parents[5]
TAIL = (
    ROOT / "python/sglang/srt/hardware_backend/npu/dsv4/npu_decoder_replay_tail.py"
)
POOL = ROOT / "python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py"

register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def _extract(
    path: Path,
    class_name: str,
    names: set[str],
    *,
    global_names: set[str] = frozenset(),
) -> dict:
    tree = ast.parse(path.read_text(), str(path))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    found = {}
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name in names:
            node = copy.deepcopy(item)
            node.decorator_list = []
            found[item.name] = node
    missing = names - set(found)
    assert not missing, missing
    globals_ = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in global_names
            for target in node.targets
        )
    ]
    module_funcs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in global_names
    ]
    body = [
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
        *globals_,
        *module_funcs,
        *found.values(),
    ]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {"torch": torch, "copy": copy}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class _Backend:
    def __init__(self, kind: str):
        self.kind = kind

    def is_none(self) -> bool:
        return self.kind == "none"

    def is_deepep(self) -> bool:
        return self.kind == "deepep"

    def is_deepep_v2(self) -> bool:
        return self.kind == "deepep_v2"


class _Mode:
    def __init__(self, low_latency: bool):
        self.low_latency = low_latency

    def resolve(self, is_extend_in_batch: bool):
        return SimpleNamespace(is_low_latency=lambda: self.low_latency)


class _Env:
    def __init__(self, value: int):
        self.value = value

    def get(self) -> int:
        return self.value


class TestTailGraphGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tail = _extract(
            TAIL,
            "NpuDecoderReplayTail",
            {"_padded_bs", "_pad_lens", "_moe_can_capture"},
        )
        cls.pool = _extract(
            POOL,
            "DSV4NPUTokenToKVPool",
            {"_swa_sink_row", "_dead_rows_to_sink"},
        )
        cls.pool["sink_dead_rows"] = sink_dead_rows
        cls.copy_ns = _extract(
            TAIL,
            "NpuDecoderReplayTail",
            {"_make_static_batch", "_copy_batch_values"},
            global_names={
                "_BATCH_TENSOR_FIELDS",
                "_COUNT_FIELDS",
                "_BATCH_LIVE_FIELDS",
                "_is_tensor",
                "_copy_value",
            },
        )

    def _fake_tail(self):
        fake = SimpleNamespace(_row_step=128)
        # A static method assigned to an instance is not rebound.
        fake._pad_lens = self.tail["_pad_lens"]
        return fake

    def test_padded_request_count_distinguishes_one_by_128_and_two_by_64(self):
        """The graph key needs the request dimension, not only the row bucket."""
        fake = self._fake_tail()
        padded_bs = self.tail["_padded_bs"]
        self.assertEqual(padded_bs(fake, [128], 128), 1)
        self.assertEqual(padded_bs(fake, [64, 64], 128), 2)
        # An idle rank pads the bucket with one dummy request.
        self.assertEqual(padded_bs(fake, [], 128), 1)
        # A short tail adds a dummy chunk.
        self.assertEqual(padded_bs(fake, [100], 128), 2)

    def _moe_capture(self, kind: str, low_latency: bool, capacity: int):
        namespace = self.tail
        namespace["get_moe_a2a_backend"] = lambda: _Backend(kind)
        namespace["get_deepep_mode"] = lambda: _Mode(low_latency)
        namespace["envs"] = SimpleNamespace(
            SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=_Env(capacity),
            SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=_Env(capacity),
        )
        return namespace["_moe_can_capture"](128)

    def test_moe_capture_rejects_normal_mode_and_over_capacity(self):
        """A captured segment cannot run DeepEP's normal sync or exceed its buffer."""
        self.assertTrue(self._moe_capture("none", False, 0))
        self.assertTrue(self._moe_capture("deepep", True, 128))
        self.assertFalse(self._moe_capture("deepep", False, 1024))
        self.assertFalse(self._moe_capture("deepep", True, 64))
        self.assertTrue(self._moe_capture("deepep_v2", True, 128))
        self.assertFalse(self._moe_capture("deepep_v2", False, 128))
        self.assertFalse(self._moe_capture("ascend_ops", True, 1024))

    def test_replay_refreshes_positions_and_input_ids(self):
        """A same-shape replay must not read the capture step's tail inputs."""
        ns = self.copy_ns
        fake = SimpleNamespace()
        first = SimpleNamespace(
            positions=torch.tensor([128, 129, 130]),
            input_ids=torch.tensor([7, 8, 9]),
        )
        static = ns["_make_static_batch"](
            fake, first, first.positions, first.input_ids, None
        )
        ns["_copy_batch_values"](fake, first, static)
        second = SimpleNamespace(
            positions=torch.tensor([256, 257, 258]),
            input_ids=torch.tensor([17, 18, 19]),
        )
        ns["_copy_batch_values"](fake, second, static)
        self.assertEqual(static.positions.tolist(), [256, 257, 258])
        self.assertEqual(static.input_ids.tolist(), [17, 18, 19])

    def test_swa_sink_is_the_reserved_page_or_the_ring_spare_row(self):
        """A paged pool's last page is live KV; only page 0 may take dummy writes."""
        sink = self.pool["_swa_sink_row"]
        paged = SimpleNamespace(swa_req_ring_size=None, size=1000)
        self.assertEqual(sink(paged, SimpleNamespace(size=1024)), 0)
        ring = SimpleNamespace(swa_req_ring_size=256, size=2048)
        # A ring owns rows [0, size); row size is the spare page.
        self.assertEqual(sink(ring, SimpleNamespace(size=2048)), 2048)

    def test_dead_rows_map_to_the_sink_and_their_payload_is_zeroed(self):
        cast = self.pool["_dead_rows_to_sink"]
        cache = torch.arange(6, dtype=torch.float32).view(3, 2)
        loc = torch.tensor([5, -1, 7], dtype=torch.int64)
        values, mapped = cast(cache, loc, 0)
        self.assertEqual(mapped.tolist(), [5, 0, 7])
        self.assertEqual(values[1].tolist(), [0.0, 0.0])
        self.assertEqual(values[0].tolist(), [0.0, 1.0])
