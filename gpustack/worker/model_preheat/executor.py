import hashlib
import io
import secrets
import socket
import ssl
import time
from urllib.parse import urlparse


def execute_connectivity_check(
    client,
    bucket: str,
    prefix: str,
    check_id: int,
    worker_uuid: str,
    *,
    endpoint: str | None = None,
    tls_verify: bool = True,
    check_network: bool = True,
    network_probe=None,
) -> dict:
    started = time.monotonic()
    if check_network and endpoint:
        failure = (network_probe or _probe_network)(endpoint, tls_verify)
        if failure is not None:
            return _failure(failure[0], failure[1], started)

    probe_name = _probe_name(prefix, check_id, worker_uuid)
    payload = secrets.token_bytes(32)
    payload_digest = hashlib.sha256(payload).hexdigest()
    object_may_exist = False
    result = None
    try:
        try:
            list(client.list_objects(bucket, prefix=probe_name.rsplit("/", 1)[0]))
        except Exception as exc:
            code = (
                "s3_authentication_failed"
                if _is_auth_failure(exc)
                else "s3_list_failed"
            )
            return _failure(
                "auth" if code.endswith("authentication_failed") else "list",
                code,
                started,
            )

        try:
            object_may_exist = True
            client.put_object(
                bucket,
                probe_name,
                io.BytesIO(payload),
                len(payload),
                content_type="application/octet-stream",
                metadata={"sha256": payload_digest},
            )
        except Exception:
            result = _failure("write", "s3_write_failed", started)
            return result

        try:
            response = client.get_object(bucket, probe_name)
            try:
                received = response.read()
            finally:
                _close_response(response)
            if hashlib.sha256(received).hexdigest() != payload_digest:
                result = _failure("read", "s3_read_content_mismatch", started)
                return result
        except Exception:
            result = _failure("read", "s3_read_failed", started)
            return result

        result = {
            "state": "ready",
            "readable": True,
            "writable": True,
            "deletable": True,
            "cleanup_failed": False,
            "latency_ms": _latency_ms(started),
        }
        return result
    finally:
        if object_may_exist:
            try:
                client.remove_object(bucket, probe_name)
            except Exception:
                if result is not None:
                    if result["state"] == "ready":
                        result.update(_failure("delete", "s3_delete_failed", started))
                else:
                    result = _failure("delete", "s3_delete_failed", started)
                result["cleanup_failed"] = True


def execute_profile_connectivity_check(
    profile: dict,
    check_id: int,
    worker_uuid: str,
    client_factory,
) -> dict:
    started = time.monotonic()
    endpoint = profile["endpoint"]
    parsed = urlparse(endpoint)
    try:
        client = client_factory(
            endpoint=parsed.netloc,
            access_key=profile["access_key"],
            secret_key=profile["secret_key"],
            secure=parsed.scheme == "https" and profile.get("tls_enabled", True),
            region=profile.get("region") or None,
        )
    except Exception:
        return _failure("client", "s3_client_initialization_failed", started)
    return execute_connectivity_check(
        client,
        profile["bucket"],
        profile.get("prefix", ""),
        check_id,
        worker_uuid,
        endpoint=endpoint,
        tls_verify=profile.get("tls_verify", True),
    )


def _probe_network(endpoint: str, tls_verify: bool):
    parsed = urlparse(endpoint)
    host = parsed.hostname
    if not host:
        return "dns", "dns_resolution_failed"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return "dns", "dns_resolution_failed"
    try:
        connection = socket.create_connection((host, port), timeout=10)
    except OSError:
        return "tcp", "tcp_connection_failed"
    try:
        if parsed.scheme == "https":
            context = (
                ssl.create_default_context()
                if tls_verify
                else ssl._create_unverified_context()
            )
            with context.wrap_socket(connection, server_hostname=host):
                pass
        else:
            pass
    except ssl.SSLCertVerificationError:
        return "tls", "tls_certificate_verify_failed"
    except ssl.SSLError:
        return "tls", "tls_handshake_failed"
    except (TimeoutError, ConnectionResetError, OSError):
        return "tls", "tls_handshake_failed"
    finally:
        connection.close()
    return None


def _probe_name(prefix: str, check_id: int, worker_uuid: str) -> str:
    clean_prefix = prefix.strip("/")
    parts = [
        part
        for part in (
            clean_prefix,
            "_healthchecks",
            str(check_id),
            worker_uuid,
            f"probe-{secrets.token_hex(16)}",
        )
        if part
    ]
    return "/".join(parts)


def _failure(stage: str, code: str, started: float) -> dict:
    return {
        "state": "error",
        "readable": False,
        "writable": False,
        "deletable": False,
        "cleanup_failed": False,
        "failed_stage": stage,
        "error_code": code,
        "latency_ms": _latency_ms(started),
    }


def _latency_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


def _close_response(response):
    close = getattr(response, "close", None)
    release_conn = getattr(response, "release_conn", None)
    if callable(close):
        close()
    if callable(release_conn):
        release_conn()


def _is_auth_failure(exc: Exception) -> bool:
    code = str(getattr(exc, "code", "")).lower()
    message = str(exc).lower()
    return code in {
        "accessdenied",
        "invalidaccesskeyid",
        "signaturedoesnotmatch",
    } or any(
        token in message for token in ("access denied", "invalid access", "signature")
    )
