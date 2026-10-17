import unittest

from local_http_security import is_local_http_request, local_http_authority


class LocalHttpSecurityTests(unittest.TestCase):
    def test_builds_ipv4_and_ipv6_authorities(self):
        self.assertEqual(local_http_authority(("127.0.0.1", 8765)), "127.0.0.1:8765")
        self.assertEqual(local_http_authority(("::1", 8765)), "[::1]:8765")

    def test_accepts_only_the_bound_host_and_local_origin(self):
        address = ("127.0.0.1", 8765)
        self.assertTrue(
            is_local_http_request(
                {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"},
                address,
                require_origin=True,
            )
        )
        self.assertFalse(
            is_local_http_request(
                {"Host": "attacker.example:8765", "Origin": "http://attacker.example:8765"},
                address,
            )
        )
        self.assertFalse(
            is_local_http_request(
                {"Host": "127.0.0.1:8765", "Origin": "http://attacker.example"},
                address,
            )
        )

    def test_mutating_requests_require_an_origin(self):
        self.assertFalse(
            is_local_http_request(
                {"Host": "127.0.0.1:8765"},
                ("127.0.0.1", 8765),
                require_origin=True,
            )
        )
        self.assertTrue(
            is_local_http_request(
                {"Host": "127.0.0.1:8765"},
                ("127.0.0.1", 8765),
            )
        )


if __name__ == "__main__":
    unittest.main()
