import os
import tempfile
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.utils import _guard_official_opapi_resolution


class TestOfficialOpapiGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cust_dir = os.path.join(self.tmp.name, "cust")
        os.makedirs(self.cust_dir)
        with open(os.path.join(self.cust_dir, "libcust_opapi.so"), "w"):
            pass
        self.plain_dir = os.path.join(self.tmp.name, "plain")
        os.makedirs(self.plain_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_call_hides_cust_vendors_and_restores_env(self):
        # A CANN cust-op vendor in ASCEND_CUSTOM_OPP_PATH exports its own
        # aclnn entry points, which the JIT extensions prefer over the official
        # transformer lib on their first call; later calls reuse that choice.
        seen = []

        def op():
            seen.append(
                (
                    os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
                    os.environ.get("LD_LIBRARY_PATH"),
                )
            )

        guarded = _guard_official_opapi_resolution(op, "kv_compress_epilog")
        with patch.dict(
            os.environ,
            {
                "ASCEND_CUSTOM_OPP_PATH": "/vendor/customize",
                "LD_LIBRARY_PATH": os.pathsep.join([self.cust_dir, self.plain_dir]),
            },
        ):
            guarded()
            self.assertEqual(seen[0], (None, self.plain_dir))
            self.assertEqual(
                os.environ["ASCEND_CUSTOM_OPP_PATH"], "/vendor/customize"
            )
            self.assertIn(self.cust_dir, os.environ["LD_LIBRARY_PATH"].split(os.pathsep))

            guarded()
            self.assertEqual(seen[1][0], "/vendor/customize")
            self.assertIn(self.cust_dir, seen[1][1].split(os.pathsep))

    def test_failed_first_call_keeps_the_guard_armed(self):
        seen = []

        def op():
            seen.append(os.environ.get("ASCEND_CUSTOM_OPP_PATH"))
            if len(seen) == 1:
                raise RuntimeError("tiling failed")

        guarded = _guard_official_opapi_resolution(op, "compressor")
        with patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": "/vendor/customize"}):
            with self.assertRaises(RuntimeError):
                guarded()
            guarded()
            guarded()
        self.assertEqual(seen, [None, None, "/vendor/customize"])


if __name__ == "__main__":
    unittest.main()
