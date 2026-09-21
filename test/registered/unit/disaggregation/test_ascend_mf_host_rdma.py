"""Unit tests for the ascend memfabric host_rdma nic endpoint resolution."""

import unittest

from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAscendMfHostRdmaNic(CustomTestCase):
    def test_roles_get_disjoint_port_ranges(self):
        # Both PD roles bind their nic endpoint, so a shared host needs
        # disjoint ranges: Decode = base + rank * stride, Prefill additionally
        # offset by the role stride.
        resolve = AscendTransferEngine._resolve_worker_hcom_url
        self.assertEqual(
            resolve("tcp://10.0.0.1:24000", "Decode", 0), "tcp://10.0.0.1:24000"
        )
        self.assertEqual(
            resolve("tcp://10.0.0.1:24000", "Decode", 7), "tcp://10.0.0.1:24056"
        )
        self.assertEqual(
            resolve("tcp://10.0.0.1:24000", "Prefill", 0), "tcp://10.0.0.1:24128"
        )
        self.assertEqual(
            resolve("tcp://10.0.0.1:24000", "Prefill", 7), "tcp://10.0.0.1:24184"
        )

    def test_missing_or_malformed_url_is_rejected(self):
        resolve = AscendTransferEngine._resolve_worker_hcom_url
        with self.assertRaisesRegex(ValueError, "ASCEND_MF_HCOM_URL"):
            resolve("", "Decode", 0)
        for malformed in ("10.0.0.1:24000", "tcp://10.0.0.1:notaport"):
            with self.assertRaisesRegex(ValueError, "Invalid port"):
                resolve(malformed, "Decode", 0)

    def test_out_of_range_port_is_rejected(self):
        resolve = AscendTransferEngine._resolve_worker_hcom_url
        with self.assertRaisesRegex(ValueError, "out of range"):
            resolve("tcp://10.0.0.1:65500", "Prefill", 7)


if __name__ == "__main__":
    unittest.main()
