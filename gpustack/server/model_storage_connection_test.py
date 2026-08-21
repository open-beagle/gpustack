"""``POST /model-storage/connection-tests`` 的 Server 侧短生命周期检查。

任务 3 步骤 4（设计文档 §10.4）：

- 直接使用**未保存**的 Profile 表单从 Server 检查连接、Bucket、Prefix、
  临时对象写入、读取和删除；不创建 Profile 或持久任务，不记录请求体；
- 异常路径也清理临时对象（``finally`` 中删除）；
- 响应固定 ``scope=server``，分阶段报告连接、Bucket、写、读、删除结果，
  权限不足不被折叠为笼统“连接失败”；
- 受控解析器使用固定短超时、禁止重定向，拒绝 link-local、云元数据地址
  及 DNS 解析后落入这些范围的目标；已验证地址用于同一次连接；允许 loopback
  和 RFC1918 企业内网 MinIO；
- 加密能力不可用时返回稳定错误码（由调用方处理）。

凭据不入库、不写日志、不进入 SSE；临时对象 Key 由 Server 在当前 Prefix 下
生成并在 ``finally`` 中清理。
"""

import hashlib
import io
import ipaddress
import secrets
import socket
import ssl
import urllib.parse
from typing import Optional

from gpustack.schemas.model_storage_sync import (
    ModelStorageConnectionStagePublic,
    ModelStorageConnectionTestPublic,
)

# 固定短超时（秒）：受控解析与 TCP/TLS 探测均使用，避免长挂起。
CONNECTION_TIMEOUT = 3


class ModelStorageConnectionError(Exception):
    """连接测试失败：携带稳定阶段与错误码，不泄露凭据。"""

    def __init__(self, stage: str, code: str):
        super().__init__(f"{stage}:{code}")
        self.stage = stage
        self.code = code


def validate_endpoint_url(endpoint: str) -> urllib.parse.ParseResult:
    """与 Profile CRUD 共用的 Endpoint/TLS 校验：仅 http(s)，host 非空。"""
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("invalid_endpoint_scheme")
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("invalid_endpoint_scheme")
    return parsed


def _is_disallowed_address(ip: ipaddress._BaseAddress) -> bool:
    # 拒绝 link-local（含云元数据 169.254.169.254）、组播、保留、未指定；
    # 允许 loopback 与 RFC1918 企业内网 MinIO（is_private 为真），其余公网允许。
    if ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True
    return False


def resolve_verified_address(endpoint: str) -> str:
    """受控 DNS 解析：固定短超时，返回一个通过安全校验的 IP。

    拒绝 link-local、云元数据及解析后落入这些范围的目标；允许 loopback 与
    RFC1918。解析失败或不安全时抛出 :class:`ModelStorageConnectionError`。
    调用方应把返回地址用于同一次连接，避免校验与连接分别解析。
    """
    parsed = validate_endpoint_url(endpoint)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    # host 本身可能是字面 IP，也需按同样规则校验（拒绝 link-local/元数据）。
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _is_disallowed_address(literal):
            raise ModelStorageConnectionError("dns", "s3_forbidden_address")
        return str(literal)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        raise ModelStorageConnectionError(
            "dns", "dns_resolution_failed"
        ) from None
    if not infos:
        raise ModelStorageConnectionError("dns", "dns_resolution_failed")
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _is_disallowed_address(ip):
            continue
        return str(ip)
    # 所有解析结果都不安全（例如全部落入 link-local/元数据）时拒绝。
    raise ModelStorageConnectionError("dns", "s3_forbidden_address")


def _stage(ok: bool, error_code: Optional[str] = None) -> ModelStorageConnectionStagePublic:
    return ModelStorageConnectionStagePublic(ok=ok, error_code=error_code)


def _not_reached() -> ModelStorageConnectionStagePublic:
    return _stage(False, "not_reached")


def _connection_result(
    error_code: str,
) -> ModelStorageConnectionTestPublic:
    """连接/DNS/client 阶段失败：连接标记失败，后续阶段 not_reached。"""
    return ModelStorageConnectionTestPublic(
        scope="server",
        ok=False,
        connection=_stage(False, error_code),
        bucket=_not_reached(),
        write=_not_reached(),
        read=_not_reached(),
        delete=_not_reached(),
        error_code=error_code,
    )


def _full_result(
    *,
    bucket: ModelStorageConnectionStagePublic,
    write: ModelStorageConnectionStagePublic,
    read: ModelStorageConnectionStagePublic,
    delete: ModelStorageConnectionStagePublic,
    error_code: Optional[str],
    ok: bool,
) -> ModelStorageConnectionTestPublic:
    return ModelStorageConnectionTestPublic(
        scope="server",
        ok=ok,
        connection=_stage(True),
        bucket=bucket,
        write=write,
        read=read,
        delete=delete,
        error_code=error_code,
    )


def _probe_tcp(
    endpoint: str, verified_host: str, tls_verify: bool
) -> Optional[str]:
    """固定短超时的 TCP/TLS 探测；成功返回 None，失败返回稳定错误码。"""
    parsed = validate_endpoint_url(endpoint)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        connection = socket.create_connection(
            (verified_host, port), timeout=CONNECTION_TIMEOUT
        )
    except OSError:
        return "tcp_connection_failed"
    try:
        if parsed.scheme == "https":
            context = (
                ssl.create_default_context()
                if tls_verify
                else ssl._create_unverified_context()
            )
            with context.wrap_socket(connection, server_hostname=parsed.hostname):
                pass
        return None
    except ssl.SSLCertVerificationError:
        return "tls_certificate_verify_failed"
    except Exception:
        return "tls_handshake_failed"
    finally:
        connection.close()


def run_model_storage_connection_test(
    *,
    endpoint: str,
    bucket: str,
    prefix: str,
    access_key: str,
    secret_key: str,
    tls_enabled: bool,
    tls_verify: bool,
    region: Optional[str],
    use_virtual_hosted_style: bool,
    client_factory,
) -> ModelStorageConnectionTestPublic:
    """分阶段执行连接测试；``client_factory`` 返回底层 client（可注入）。

    阶段顺序：连接（受控 DNS + TCP/TLS）→ Bucket(list) → 写 → 读 → 删除。
    已验证地址交给同一次连接使用（``verified_host``），避免校验与连接分别解析。
    已写入的临时对象在 ``finally`` 中清理，异常路径也不泄露。
    加密能力不可用由调用方提前以稳定错误码处理，不进入本函数。
    """
    # 阶段 1：受控 DNS。
    try:
        verified_host = resolve_verified_address(endpoint)
    except ModelStorageConnectionError as exc:
        return _connection_result(exc.code)

    # 阶段 1：TCP/TLS 探测（禁止重定向，固定短超时）。
    tcp_code = _probe_tcp(endpoint, verified_host, tls_verify)
    if tcp_code is not None:
        return _connection_result(tcp_code)

    # 阶段 2+：Bucket/写/读/删除。用注入的 client，连接同一已验证地址。
    try:
        client = client_factory(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=(validate_endpoint_url(endpoint).scheme == "https")
            and tls_enabled,
            tls_verify=tls_verify,
            region=region,
            use_virtual_hosted_style=use_virtual_hosted_style,
            verified_host=verified_host,
        )
    except Exception:
        return _connection_result("s3_client_initialization_failed")

    probe_prefix = prefix.strip("/")
    probe_name = (
        f"{probe_prefix}/_connection-tests/probe-{secrets.token_hex(12)}"
        if probe_prefix
        else f"_connection-tests/probe-{secrets.token_hex(12)}"
    )
    payload = secrets.token_bytes(32)
    payload_digest = hashlib.sha256(payload).hexdigest()
    object_may_exist = False
    try:
        # 阶段 2：Bucket 访问（list）。
        try:
            list(client.list_objects(bucket, prefix=probe_prefix))
        except Exception as exc:
            code = (
                "s3_authentication_failed"
                if _is_auth_failure(exc)
                else "s3_list_failed"
            )
            failed = _stage(False, code)
            return _full_result(
                bucket=failed,
                write=failed,
                read=failed,
                delete=failed,
                error_code=code,
                ok=False,
            )

        # 阶段 3：写入临时对象。
        try:
            object_may_exist = True
            client.put_object(
                bucket,
                probe_name,
                io.BytesIO(payload),
                len(payload),
                content_type="application/octet-stream",
            )
            write_stage = _stage(True)
        except Exception:
            failed = _stage(False, "s3_write_failed")
            return _full_result(
                bucket=_stage(True),
                write=failed,
                read=failed,
                delete=failed,
                error_code="s3_write_failed",
                ok=False,
            )

        # 阶段 4：读取并校验。
        try:
            response = client.get_object(bucket, probe_name)
            try:
                received = response.read()
            finally:
                _close_response(response)
            if hashlib.sha256(received).hexdigest() != payload_digest:
                failed = _stage(False, "s3_read_content_mismatch")
                return _full_result(
                    bucket=_stage(True),
                    write=write_stage,
                    read=failed,
                    delete=failed,
                    error_code="s3_read_content_mismatch",
                    ok=False,
                )
            read_stage = _stage(True)
        except Exception:
            failed = _stage(False, "s3_read_failed")
            return _full_result(
                bucket=_stage(True),
                write=write_stage,
                read=failed,
                delete=failed,
                error_code="s3_read_failed",
                ok=False,
            )

        # 阶段 5：删除临时对象（在 finally 中执行，异常路径也清理）。
        return _full_result(
            bucket=_stage(True),
            write=write_stage,
            read=read_stage,
            delete=_stage(True),
            error_code=None,
            ok=True,
        )
    finally:
        if object_may_exist:
            try:
                client.remove_object(bucket, probe_name)
            except Exception:
                # 尽力清理；删除失败不改变已计算的阶段结果（成功路径已返回）。
                pass


def _close_response(response) -> None:
    close = getattr(response, "close", None)
    release = getattr(response, "release_conn", None)
    if callable(close):
        close()
    if callable(release):
        release()


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
