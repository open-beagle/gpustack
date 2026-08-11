import io
import re
import ssl

import pytest

from gpustack.worker.model_preheat import executor
from gpustack.worker.model_preheat.executor import (
    execute_connectivity_check,
    execute_profile_connectivity_check,
)


class Response:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload

    def close(self):
        pass

    def release_conn(self):
        pass


class MemoryS3:
    def __init__(self, failure=None):
        self.failure = failure
        self.objects = {}
        self.deleted = []

    def list_objects(self, bucket, prefix=None, recursive=False):
        if self.failure == "auth":
            raise RuntimeError("access denied")
        return []

    def put_object(self, bucket, name, data, length, **kwargs):
        if self.failure == "write":
            raise RuntimeError("write denied")
        self.objects[name] = data.read(length)

    def get_object(self, bucket, name):
        if self.failure == "read":
            raise RuntimeError("read denied")
        return Response(self.objects[name])

    def remove_object(self, bucket, name):
        if self.failure == "delete":
            raise RuntimeError("delete denied")
        self.deleted.append(name)
        self.objects.pop(name, None)


class PutTimeoutAfterWriteS3(MemoryS3):
    def put_object(self, bucket, name, data, length, **kwargs):
        self.objects[name] = data.read(length)
        raise TimeoutError("client timeout with plain-access-key and plain-secret-key")


def test_profile_connectivity_check_sanitizes_client_factory_exception():
    access_key = "factory-plain-access-key"
    secret_key = "factory-plain-secret-key"

    def failing_factory(**kwargs):
        raise RuntimeError(
            f"failed to build client with {kwargs['access_key']} {kwargs['secret_key']}"
        )

    result = execute_profile_connectivity_check(
        {
            "endpoint": "https://s3.example.com",
            "bucket": "models",
            "access_key": access_key,
            "secret_key": secret_key,
        },
        3,
        "worker-uuid",
        failing_factory,
    )

    assert result["state"] == "error"
    assert result["failed_stage"] == "client"
    assert result["error_code"] == "s3_client_initialization_failed"
    assert access_key not in repr(result)
    assert secret_key not in repr(result)


@pytest.mark.parametrize(
    ("failure", "stage", "code"),
    [
        ("auth", "auth", "s3_authentication_failed"),
        ("write", "write", "s3_write_failed"),
        ("read", "read", "s3_read_failed"),
        ("delete", "delete", "s3_delete_failed"),
    ],
)
def test_connectivity_check_returns_stable_s3_stage_errors(failure, stage, code):
    result = execute_connectivity_check(
        MemoryS3(failure), "models", "prefix", 3, "worker-uuid", check_network=False
    )
    assert result["state"] == "error"
    assert result["failed_stage"] == stage
    assert result["error_code"] == code


def test_connectivity_check_reads_writes_and_removes_random_probe():
    client = MemoryS3()
    results = [
        execute_connectivity_check(
            client, "models", "prefix", 3, "worker-uuid", check_network=False
        )
        for _ in range(2)
    ]
    assert all(result["state"] == "ready" for result in results)
    assert all(
        result["readable"] and result["writable"] and result["deletable"]
        for result in results
    )
    assert all(result["latency_ms"] >= 0 for result in results)
    assert client.objects == {}
    assert len(set(client.deleted)) == 2
    assert all(
        re.fullmatch(
            r"prefix/_healthchecks/3/worker-uuid/probe-[0-9a-f]{32}", probe_name
        )
        for probe_name in client.deleted
    )


def test_connectivity_check_cleanup_failure_is_reported_without_credentials():
    result = execute_connectivity_check(
        MemoryS3("delete"), "models", "prefix", 3, "worker-uuid", check_network=False
    )
    assert result["cleanup_failed"] is True
    assert "access" not in str(result).lower()
    assert "secret" not in str(result).lower()


def test_connectivity_check_deletes_probe_when_put_writes_then_times_out():
    client = PutTimeoutAfterWriteS3()

    result = execute_connectivity_check(
        client, "models", "prefix", 3, "worker-uuid", check_network=False
    )

    assert result["state"] == "error"
    assert result["failed_stage"] == "write"
    assert result["error_code"] == "s3_write_failed"
    assert result["cleanup_failed"] is False
    assert client.objects == {}
    assert len(client.deleted) == 1
    assert re.fullmatch(
        r"prefix/_healthchecks/3/worker-uuid/probe-[0-9a-f]{32}", client.deleted[0]
    )
    assert "plain-access-key" not in repr(result)
    assert "plain-secret-key" not in repr(result)


def test_connectivity_check_preserves_put_error_when_uncertain_probe_cleanup_fails():
    client = PutTimeoutAfterWriteS3("delete")

    result = execute_connectivity_check(
        client, "models", "prefix", 3, "worker-uuid", check_network=False
    )

    assert result["state"] == "error"
    assert result["failed_stage"] == "write"
    assert result["error_code"] == "s3_write_failed"
    assert result["cleanup_failed"] is True
    assert "plain-access-key" not in repr(result)
    assert "plain-secret-key" not in repr(result)


@pytest.mark.parametrize(
    ("failure", "stage", "code"),
    [
        (("dns", "dns_resolution_failed"), "dns", "dns_resolution_failed"),
        (("tcp", "tcp_connection_failed"), "tcp", "tcp_connection_failed"),
        (
            ("tls", "tls_certificate_verify_failed"),
            "tls",
            "tls_certificate_verify_failed",
        ),
    ],
)
def test_connectivity_check_reports_network_stage_errors(failure, stage, code):
    result = execute_connectivity_check(
        MemoryS3(),
        "models",
        "prefix",
        3,
        "worker-uuid",
        endpoint="https://s3.example.com",
        network_probe=lambda endpoint, tls_verify: failure,
    )

    assert result["state"] == "error"
    assert result["failed_stage"] == stage
    assert result["error_code"] == code


class _Connection:
    def close(self):
        pass


class _TLSContext:
    def __init__(self, failure):
        self.failure = failure

    def wrap_socket(self, connection, server_hostname):
        raise self.failure


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("TLS timeout"),
        ConnectionResetError("TLS reset"),
        OSError("TLS I/O failure"),
    ],
)
def test_probe_network_maps_tls_transport_errors_to_handshake_failed(
    monkeypatch, failure
):
    monkeypatch.setattr(executor.socket, "getaddrinfo", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        executor.socket, "create_connection", lambda *args, **kwargs: _Connection()
    )
    monkeypatch.setattr(
        executor.ssl, "create_default_context", lambda: _TLSContext(failure)
    )

    assert executor._probe_network("https://s3.example.com", True) == (
        "tls",
        "tls_handshake_failed",
    )


def test_probe_network_preserves_tls_certificate_verification_error(monkeypatch):
    monkeypatch.setattr(executor.socket, "getaddrinfo", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        executor.socket, "create_connection", lambda *args, **kwargs: _Connection()
    )
    monkeypatch.setattr(
        executor.ssl,
        "create_default_context",
        lambda: _TLSContext(ssl.SSLCertVerificationError("bad certificate")),
    )

    assert executor._probe_network("https://s3.example.com", True) == (
        "tls",
        "tls_certificate_verify_failed",
    )
