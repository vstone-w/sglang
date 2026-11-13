"""Tests for DeepSeek-V4.1 PP partition validation and PD state pairing."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.deepseek_v4_hook import validate_dsv41_pp_partition
from sglang.srt.disaggregation.utils import build_transfer_entry_pairs
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=2, suite="base-a-test-npu")

# The checkpoint's decoder ratios: two plain layers, 18 ratio-2 layers and 20
# ratio-1 layers; the trailing nextn entries are not decoder layers.
_DECODER_RATIOS = [0, 0] + [2] * 18 + [1] * 20


def _hf_config(**overrides):
    values = dict(
        num_hidden_layers=40,
        compress_ratios=_DECODER_RATIOS,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        vision_n_layers=0,
        dspark_target_layer_ids=[37, 38, 39],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDsv41PpPartition(unittest.TestCase):
    def _validate(self, partition, pp_size, *, cfg=None, language_only=True):
        # The validator reads the runtime's own env entry, exactly like
        # get_pp_indices does at model construction.
        with patch.dict(os.environ):
            if partition is None:
                os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
            else:
                os.environ["SGLANG_PP_LAYER_PARTITION"] = partition
            validate_dsv41_pp_partition(
                cfg if cfg is not None else _hf_config(),
                pp_size=pp_size,
                language_model_only=language_only,
            )

    def test_documented_partitions_are_accepted(self):
        for partition, pp_size in (
            ("20,20", 2),
            (None, 2),  # uniform 20/20
            ("14,6,20", 3),
            ("8,12,20", 3),
            ("8,6,6,20", 4),
            ("2,6,6,26", 4),
            ("1,1,6,6,6,20", 6),
        ):
            with self.subTest(partition=partition, pp_size=pp_size):
                self._validate(partition, pp_size)

    def test_empty_partition_env_means_unset(self):
        # Matches get_pp_indices: "" is the even split, not a parse error.
        self._validate("", 2)

    def test_uniform_pp4_split_is_rejected(self):
        # Boundary 10 splits the ratio-2 group [8,14); boundary 30 splits the
        # candidate group 20..39.
        with self.assertRaisesRegex(ValueError, "kv_source"):
            self._validate("10,10,10,10", 4)

    def test_pp3_split_inside_source_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "kv_source"):
            self._validate("13,7,20", 3)

    def test_interleaved_ratio_consumer_keeps_its_source(self):
        # [0, 2, 0, 2] with source 1: layer 3 consumes source 1 even though
        # layer 2 has another ratio.
        cfg = _hf_config(
            num_hidden_layers=4,
            compress_ratios=[0, 2, 0, 2],
            kv_source_layer_ids=[1],
            index_source_layer_ids=[1],
            candidate_source_layer_id=-1,
        )
        self._validate("1,3", 2, cfg=cfg)
        with self.assertRaisesRegex(ValueError, "kv_source"):
            self._validate("2,2", 2, cfg=cfg)

    def test_index_source_split_is_rejected(self):
        # KV and index sources need not coincide.
        cfg = _hf_config(
            num_hidden_layers=4,
            compress_ratios=[0, 2, 2, 2],
            kv_source_layer_ids=[1, 2],
            index_source_layer_ids=[1],
            candidate_source_layer_id=-1,
        )
        self._validate("1,3", 2, cfg=cfg)
        with self.assertRaisesRegex(ValueError, "index source"):
            self._validate("2,2", 2, cfg=cfg)

    def test_split_after_candidate_source_is_rejected(self):
        cfg = _hf_config(
            compress_ratios=[0] * 40,
            kv_source_layer_ids=[],
            candidate_source_layer_id=5,
        )
        with self.assertRaisesRegex(ValueError, "candidate"):
            self._validate("20,20", 2, cfg=cfg)

    def test_default_split_with_too_many_stages_hits_a_group(self):
        # 7-way even split puts layer 10 (ratio 2, source 8) past its stage.
        with self.assertRaisesRegex(ValueError, "kv_source"):
            self._validate(None, 7)

    def test_empty_or_negative_stage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty or negative"):
            self._validate("0,40", 2)
        with self.assertRaisesRegex(ValueError, "empty or negative"):
            self._validate("-1,41", 2)

    def test_vision_requires_language_model_only(self):
        with self.assertRaisesRegex(ValueError, "language-model-only"):
            self._validate(
                "20,20", 2, cfg=_hf_config(vision_n_layers=32), language_only=False
            )

    def test_model_capture_check_uses_the_actual_stage_range(self):
        # DSpark's draft runs only on the last stage. The check lives in the
        # model, where the resolved draft config and stage range are known;
        # the public arg/resolver path must not reject other DSpark models
        # (e.g. Kimi K3 relays captured states across stages).
        model = object.__new__(DeepseekV4ForCausalLM)
        model.pp_group = SimpleNamespace(is_last_rank=True)
        model.model = SimpleNamespace(
            start_layer=20,
            end_layer=40,
            dspark_layers_to_capture=None,
        )
        model.capture_aux_hidden_states = False
        model.set_dspark_layers_to_capture([37, 38, 39])
        self.assertTrue(model.capture_aux_hidden_states)
        with self.assertRaisesRegex(ValueError, "not on this stage"):
            model.set_dspark_layers_to_capture([5, 37])
        # A non-last rank hosts no draft and stays silent.
        model.pp_group = SimpleNamespace(is_last_rank=False)
        model.set_dspark_layers_to_capture([5])

    def test_capture_field_does_not_constrain_arg_boundaries(self):
        # The target's dspark_target_layer_ids may be stale; the resolved draft
        # config is checked at model build instead.
        cfg = _hf_config()
        del cfg.dspark_target_layer_ids
        self._validate("20,20", 2, cfg=cfg)


class TestDsv41RequestStatePairing(unittest.TestCase):
    @staticmethod
    def _pool(entries):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.compress_state_pools = [
            None
            if item is None
            else SimpleNamespace(request_scoped=item[0], ratio=item[1])
            for item in entries
        ]
        return pool

    def test_layer_ids_follow_request_scoped_pools(self):
        pool = self._pool([(False, 4), (True, 2), None, (False, 128), (True, 2)])
        self.assertEqual(pool.get_request_state_layer_ids(), [1, 4])

    def test_ids_match_buf_infos_length(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)

        def state_pool(ratio):
            return SimpleNamespace(
                request_scoped=True,
                ratio=ratio,
                ring_size=2,
                kv_score_buffer=SimpleNamespace(kv_score=torch.zeros(4, 8)),
            )

        pool.compress_state_pools = [None, state_pool(2), None, state_pool(2)]
        self.assertEqual(pool.get_request_state_layer_ids(), [1, 3])
        self.assertEqual(
            len(pool.get_request_state_layer_ids()),
            len(pool.get_request_state_buf_infos()[0]),
        )

    def test_pp_subset_pairs_by_layer_id(self):
        # A PP stage owns only the ratio-2 source at layer 4; the PP=1 decode
        # peer owns layers 1 and 4.
        pairs = build_transfer_entry_pairs(
            [4], [1, 4], 1, 2, allow_positional_fallback=False
        )
        self.assertEqual(pairs, [(0, 1)])

    def test_pp_subset_without_ids_is_rejected(self):
        # Before the ids existed this was the PP=3/4 failure: one request-scoped
        # pool on the stage, three on the decode peer, no metadata.
        with self.assertRaisesRegex(RuntimeError, "layer ids"):
            build_transfer_entry_pairs([], [], 1, 2, allow_positional_fallback=False)


if __name__ == "__main__":
    unittest.main()
