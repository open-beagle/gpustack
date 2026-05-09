from types import SimpleNamespace

from gpustack.utils.client_ip import get_client_ip


class FakeRequest:
    def __init__(self, client_host, headers=None):
        self.client = SimpleNamespace(host=client_host)
        self.headers = headers or {}


def test_get_client_ip_ignores_forwarded_headers_without_trusted_proxy():
    request = FakeRequest(
        "10.0.0.10",
        {
            "x-forwarded-for": "1.2.3.4",
            "x-real-ip": "5.6.7.8",
        },
    )

    assert get_client_ip(request) == "10.0.0.10"


def test_get_client_ip_uses_first_forwarded_for_from_trusted_proxy():
    request = FakeRequest(
        "10.0.0.10",
        {
            "x-forwarded-for": "1.2.3.4, 10.0.0.1",
        },
    )

    assert get_client_ip(request, ["10.0.0.0/24"]) == "1.2.3.4"


def test_get_client_ip_uses_real_ip_from_trusted_proxy():
    request = FakeRequest(
        "10.0.0.10",
        {
            "x-real-ip": "5.6.7.8",
        },
    )

    assert get_client_ip(request, ["10.0.0.0/24"]) == "5.6.7.8"


def test_get_client_ip_uses_forwarded_header_from_trusted_proxy():
    request = FakeRequest(
        "10.0.0.10",
        {
            "forwarded": 'proto=https;for="9.9.9.9"',
        },
    )

    assert get_client_ip(request, ["10.0.0.0/24"]) == "9.9.9.9"


def test_get_client_ip_falls_back_to_client_host_without_forwarded_headers():
    request = FakeRequest("10.0.0.10")

    assert get_client_ip(request, ["10.0.0.0/24"]) == "10.0.0.10"
