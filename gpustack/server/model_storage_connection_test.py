"""``POST /model-storage/connection-tests`` 的 Server 侧短生命周期检查。

任务 3 步骤 4（设计文档 §10.4）：

- 直接使用**未保存**的 Profile 表单从 Server 检查连接、Bucket、Prefix、
  临时对象写入、读取和删除；不创建 Profile 或持久任务，不记录请求体；
- 每一步（连接/Bucket/写/读/删除）失败都报告稳定的脱敏错误码，删除失败
  绝不折叠为整体成功；异常路径也清理临时对象（``finally`` 中删除）；
- 响应固定 ``scope=server``，分阶段报告连接、Bucket、写、读、删除结果，
  权限不足不被折叠为笼统“连接失败”；
- 受控解析器使用固定短超时，拒绝 link-local、云元数据地址及 DNS 解析后
  落入这些范围的目标；允许 loopback 和 RFC1918 企业内网 MinIO；
- **DNS 固定**：已验证 IP 真正用于本次连接的全部 TCP 连接（连接池连接类
  覆写 ``_new_conn``），Host 头、TLS SNI 与证书主机名校验仍使用原始
  主机名，防止校验与连接分别解析造成 DNS rebinding；
- **禁止重定向**：client 层拦截一切 3xx 响应并稳定失败，绝不跟随到
  未经验证的目标；
- 凭据不入库、不写日志、不进入 SSE；异常只映射为稳定错误码，endpoint
  query、access/secret key 与 bucket 等敏感信息不进入结果。
"""

import hashlib
import ipaddress
import io
import secrets
import socket
import ssl
from dataclasses import dataclass
from typing import Callable, Optional
import urllib.parse

from gpustack.schemas.model_storage_sync import (
    ModelStorageConnectionStagePublic,
    ModelStorageConnectionTestPublic,
)

#  固定短超时（秒）：TCP/TLS 探测使用，避免长挂起。
CONNECTION_TIMEOUT = 3
#  DNS 固定短超时（秒）：getaddrinfo 独立超时（不依赖 TCP 超时）。
DNS_TIMEOUT = 2

#  受控 DNS 解析器类型：(host, port) -> infos（与 socket.getaddrinfo 兼容的
#  子集；测试注入 fake 解析器验证固定超时与脱敏语义，不访问真实 DNS）。
GetAddrInfo = Callable[[str, int], list]


class ModelStorageConnectionError(Exception):
    """连接测试失败：携带稳定阶段与错误码，不泄露凭据。"""

    def __init__(self, stage: str, code: str):
        # 消息只包含阶段与稳定错误码，不携带 endpoint/凭据/bucket 等敏感信息。
        super().__init__(f"model_storage_connection_test {stage} failed: {code}")
        self.stage = stage
        self.code = code


class _RedirectBlocked(Exception):
    """内部标记：服务端返回了被禁止的重定向（目标未经验证，不得跟随）。"""

    _model_storage_redirect = True


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


@dataclass(frozen=True)
class VerifiedEndpoint:
    """受控解析结果：TCP 连接目标与 TLS/SNI 语义分离。

    ``verified_ip`` 必须用于本次连接的全部 TCP 连接（DNS 固定，防
    rebinding）；``host``/``port`` 用于 URL、Host 头与 TLS SNI/证书校验，
    两者不得混用。
    """

    scheme: str
    host: str
    port: int
    verified_ip: str

    def netloc(self) -> str:
        # IPv6 字面量需要方括号，否则 netloc 无法被 URL 解析器还原。
        if ":" in self.host and not self.host.startswith("["):
            return f"[{self.host}]:{self.port}"
        return f"{self.host}:{self.port}"


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def resolve_verified_endpoint(
    endpoint: str,
    *,
    getaddrinfo: Optional[GetAddrInfo] = None,
    timeout: Optional[float] = None,
) -> VerifiedEndpoint:
    """受控 DNS 解析：固定短超时，返回安全校验通过的连接目标。

    拒绝 link-local、云元数据及解析后落入这些范围的目标；允许 loopback 与
    RFC1918。**可测试的固定短超时**：``getaddrinfo`` 可注入（测试注入固定
    延迟/异常的解析器），超时统一为 ``timeout``（默认 :data:`DNS_TIMEOUT`），
    超时映射为稳定脱敏错误码 ``dns_resolution_timeout``（不泄露主机名/端点
    细节）。解析失败或不安全时抛出 :class:`ModelStorageConnectionError`。
    返回的 ``verified_ip`` 必须由调用方用于同一次连接的全部 TCP 连接，
    避免校验与连接分别解析（DNS rebinding）。
    """
    parsed = validate_endpoint_url(endpoint)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    # host 本身可能是字面 IP（含 IPv6），也需按同样规则校验。
    if _is_ip_literal(host):
        if _is_disallowed_address(ipaddress.ip_address(host)):
            raise ModelStorageConnectionError("dns", "s3_forbidden_address")
        return VerifiedEndpoint(parsed.scheme, host, port, host)
    resolver = getaddrinfo if getaddrinfo is not None else _default_getaddrinfo
    effective_timeout = DNS_TIMEOUT if timeout is None else timeout
    try:
        infos = _getaddrinfo_with_timeout(resolver, host, port, effective_timeout)
    except socket.timeout:
        #  固定短超时：稳定脱敏错误码，不携带主机名/端点细节。
        raise ModelStorageConnectionError("dns", "dns_resolution_timeout") from None
    except OSError:
        raise ModelStorageConnectionError("dns", "dns_resolution_failed") from None
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
        return VerifiedEndpoint(parsed.scheme, host, port, addr)
    # 所有解析结果都不安全（例如全部落入 link-local/元数据）时拒绝。
    raise ModelStorageConnectionError("dns", "s3_forbidden_address")


def _default_getaddrinfo(host: str, port: int) -> list:
    return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)


def _getaddrinfo_with_timeout(
    resolver: GetAddrInfo, host: str, port: int, timeout: float
) -> list:
    """固定短超时的 getaddrinfo 包装（跨平台、可测试，不依赖 signal）。

    ``socket.getaddrinfo`` 不支持 timeout 参数。这里使用**独立 daemon 线程**
    执行解析器，调用线程以有界 ``queue.Queue.get(timeout)`` 等待结果：
    超时或解析器异常统一映射为 ``socket.timeout`` / ``OSError``，再由
    :func:`resolve_verified_endpoint` 映射为稳定脱敏错误码
    ``dns_resolution_timeout`` / ``dns_resolution_failed``。

    不覆盖全局 ``signal``/``timer``：在 FastAPI 非主线程（uvicorn worker、
    TestClient portal thread）同样真正约束慢解析器，且不受 executor
    shutdown 影响（daemon 线程不阻塞进程退出，队列等待立即返回）。
    """
    import queue as _queue
    import threading

    result: "_queue.Queue" = _queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            result.put_nowait(("ok", resolver(host, port)))
        except socket.timeout:
            result.put_nowait(("timeout", None))
        except OSError:
            result.put_nowait(("oserror", None))
        except Exception:
            # 非 OSError/timeout 的解析器异常按解析失败处理（脱敏，不携带
            # 原始异常文本）。
            result.put_nowait(("oserror", None))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        status, value = result.get(timeout=max(0.001, float(timeout)))
    except _queue.Empty:
        raise socket.timeout("dns resolution timeout") from None
    if status == "ok":
        return value
    if status == "timeout":
        raise socket.timeout("dns resolution timeout")
    raise OSError("dns resolution failed")


def resolve_verified_address(endpoint: str) -> str:
    """向后兼容入口：返回受控解析通过的首个安全 IP。"""
    return resolve_verified_endpoint(endpoint).verified_ip


def _virtual_style_host_safe(host: str, bucket: str) -> bool:
    """virtual-hosted 风格的 ``{bucket}.{host}`` 是否可做 SNI/证书校验。

    bucket 含 "."（或 host 为 IP 字面量）时该组合不是合法主机名，证书校验
    与 DNS 固定都无法正确表达，必须回退 path style。
    """
    if not host or _is_ip_literal(host):
        return False
    if not bucket or "." in bucket or ":" in bucket or len(bucket) < 3:
        return False
    return not _is_ip_literal(bucket)


def _stage(
    ok: bool, error_code: Optional[str] = None
) -> ModelStorageConnectionStagePublic:
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


def build_pinned_http_client(
    *,
    verified: VerifiedEndpoint,
    resolver: Callable[[str], Optional[str]],
    tls_verify: bool,
) -> "object":
    """构建**DNS 固定**的 urllib3 连接池管理器。

    连接池使用的连接类覆写 ``_new_conn``：TCP 一律连向 ``resolver`` 给出的
    已验证 IP（防 DNS rebinding），而 urllib3/http.client 的 Host 头、TLS
    SNI 与证书主机名校验仍基于原始主机名（``verified.host``）。重试关闭、
    连接池自身不做重定向（重定向由 client 层拦截，见 :func:`_forbid_redirects`）。
    """
    import urllib3
    from urllib3.util import Retry, Timeout

    base_http = urllib3.connection.HTTPConnection
    base_https = urllib3.connection.HTTPSConnection

    class _PinnedHTTPConnection(base_http):
        def _new_conn(self):  # type: ignore[override]
            ip = resolver((self._dns_host or "").rstrip("."))
            if ip is None:
                raise socket.gaierror("pinned host not verified")
            # socket.create_connection 不支持 socket_options，手动设置。
            sock = socket.create_connection(
                (ip, self.port),
                self.timeout,
                source_address=self.source_address,
            )
            for opt in self.socket_options or ():
                sock.setsockopt(*opt)
            return sock

    class _PinnedHTTPSConnection(base_https):
        def _new_conn(self):  # type: ignore[override]
            ip = resolver((self._dns_host or "").rstrip("."))
            if ip is None:
                raise socket.gaierror("pinned host not verified")
            sock = socket.create_connection(
                (ip, self.port),
                self.timeout,
                source_address=self.source_address,
            )
            for opt in self.socket_options or ():
                sock.setsockopt(*opt)
            return sock

    http_pool = type(
        "PinnedHTTPConnectionPool",
        (urllib3.HTTPConnectionPool,),
        {"ConnectionCls": _PinnedHTTPConnection},
    )
    https_pool = type(
        "PinnedHTTPSConnectionPool",
        (urllib3.HTTPSConnectionPool,),
        {"ConnectionCls": _PinnedHTTPSConnection},
    )

    kwargs = dict(
        timeout=Timeout(connect=CONNECTION_TIMEOUT, read=CONNECTION_TIMEOUT),
        maxsize=2,
        retries=Retry(total=0, redirect=False),
    )
    if verified.scheme == "https":
        kwargs["cert_reqs"] = "CERT_REQUIRED" if tls_verify else "CERT_NONE"
        if not tls_verify:
            # 仅当显式允许时才跳过证书主机名校验；默认必须通过校验。
            kwargs["assert_hostname"] = False
    manager = urllib3.PoolManager(**kwargs)
    # pool_classes_by_scheme 是实例属性（__init__ 中赋值），必须在实例化后
    # 替换，使 http/https 池都使用 DNS 固定连接类。
    manager.pool_classes_by_scheme = {"http": http_pool, "https": https_pool}
    return manager


def _probe(
    verified: VerifiedEndpoint,
    *,
    tls_verify: bool,
    probe_host: Optional[str] = None,
) -> None:
    """固定短超时的 TCP/TLS 探测：TCP 连向已验证 IP，SNI/证书校验用主机名。

    不发送任何 HTTP 请求（自然不存在重定向）；失败抛出带稳定错误码的
    :class:`ModelStorageConnectionError`。
    """
    target = probe_host or verified.host
    try:
        connection = socket.create_connection(
            (verified.verified_ip, verified.port), timeout=CONNECTION_TIMEOUT
        )
    except OSError:
        raise ModelStorageConnectionError("tcp", "tcp_connection_failed") from None
    try:
        if verified.scheme == "https":
            # 统一路径：server_hostname 使用原始主机名（IP 字面量或域名），
            # ssl 模块自动处理域名 SAN 匹配与 IP SAN 匹配；check_hostname
            # 在 tls_verify=False 时由 _create_unverified_context 关闭。
            # 注意 server_hostname 不带方括号（即使 IPv6）。
            context = (
                ssl.create_default_context()
                if tls_verify
                else ssl._create_unverified_context()
            )
            with context.wrap_socket(connection, server_hostname=target):
                pass
    except ssl.SSLCertVerificationError:
        raise ModelStorageConnectionError(
            "tls", "tls_certificate_verify_failed"
        ) from None
    except Exception:
        raise ModelStorageConnectionError("tls", "tls_handshake_failed") from None
    finally:
        connection.close()


def _classify(exc: Exception) -> str:
    """把底层异常映射为稳定脱敏错误码；绝不携带原始异常文本。"""
    if isinstance(exc, ModelStorageConnectionError):
        return exc.code
    if _is_redirect_error(exc):
        return "s3_redirect_forbidden"
    if _is_auth_failure(exc):
        return "s3_authentication_failed"
    return "s3_request_failed"


def _is_redirect_error(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    if isinstance(status, int) and 300 <= status < 400:
        return True
    return getattr(exc, "_model_storage_redirect", False) is True


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
    已验证 IP 交给 client 连接池用于本次连接的全部 TCP 连接（DNS 固定），
    Host/SNI/证书校验仍用原始主机名；client 层禁止重定向。
    每一步失败都报告稳定脱敏错误码，删除失败不得折叠为成功；
    已写入的临时对象在 ``finally`` 中尽力清理。
    """
    # access_key/secret_key 仅透传给 client_factory，绝不进入结果/日志。
    # 阶段 1：受控 DNS。
    try:
        verified = resolve_verified_endpoint(endpoint)
    except ModelStorageConnectionError as exc:
        return _connection_result(exc.code)

    # 实际连接语义：连接测试统一 path style（请求 host 恒为原始 host，与
    # SNI/证书校验目标一致）；入参 virtual 风格在 IP 字面量或点号桶名等无法
    # 表达 SNI/证书语义的场景下回退为 path style。
    virtual = bool(use_virtual_hosted_style)
    if virtual and not _virtual_style_host_safe(verified.host, bucket):
        virtual = False
    # 连接池 DNS 固定表：只允许连向受控解析已验证的 IP；path style 下请求
    # host 恒等于原始 host，固定表只需该条目，其他任何 host（包括重定向
    # 目标或 virtual 派生 host）都无法建立 TCP 连接。
    hosts = {verified.host: verified.verified_ip}

    # 阶段 1：TCP/TLS 探测（TCP 到已验证 IP，SNI/证书校验用原始主机名，
    # 与实际请求 host 一致；不发送 HTTP 请求，天然无重定向）。
    try:
        _probe(verified, tls_verify=tls_verify)
    except ModelStorageConnectionError as exc:
        return _connection_result(exc.code)

    def _resolver(host: str) -> Optional[str]:
        return hosts.get((host or "").rstrip("."))

    try:
        client = client_factory(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=(verified.scheme == "https") and tls_enabled,
            tls_verify=tls_verify,
            region=region,
            use_virtual_hosted_style=virtual,
            verified=verified,
            resolver=_resolver,
        )
    except ModelStorageConnectionError as exc:
        return _connection_result(exc.code)
    except Exception:
        return _connection_result("s3_client_initialization_failed")

    _forbid_redirects(client)

    def _failure(
        stage: str,
        code: str,
        stages: dict[str, ModelStorageConnectionStagePublic],
    ) -> ModelStorageConnectionTestPublic:
        # 失败点之后尚未执行的阶段标记 not_reached，已执行阶段保持真实结果。
        result = {name: _stage(True) for name in stages}
        result[stage] = _stage(False, code)
        for name in ("bucket", "write", "read", "delete"):
            if name not in result:
                result[name] = _not_reached()
        return _full_result(ok=False, error_code=code, **result)

    stages: dict[str, ModelStorageConnectionStagePublic] = {}
    probe_name: Optional[str] = None
    # 清理责任标记：只有**写成功后**的对象才需要清理；写失败时对象可能
    # 不存在，不做兜底删除（也不重复删除已显式删除的对象）。
    cleanup_pending = False
    try:
        # 阶段 2：Bucket 访问（list）。
        try:
            list(client.list_objects(bucket, prefix=_probe_prefix(prefix)))
        except Exception as exc:
            return _failure("bucket", _classify(exc), stages)
        stages["bucket"] = _stage(True)

        # 阶段 3：写入临时对象。
        probe_name = _probe_object_name(prefix)
        payload = secrets.token_bytes(32)
        try:
            client.put_object(
                bucket,
                probe_name,
                io.BytesIO(payload),
                len(payload),
                content_type="application/octet-stream",
            )
        except Exception as exc:
            probe_name = None  # 写失败：不触发兜底删除。
            return _failure("write", _classify(exc), stages)
        stages["write"] = _stage(True)
        cleanup_pending = True

        # 阶段 4：读取并校验（清理责任在 finally）。
        try:
            response = client.get_object(bucket, probe_name)
            try:
                received = response.read()
            finally:
                _close_response(response)
            if (
                hashlib.sha256(received).hexdigest()
                != hashlib.sha256(payload).hexdigest()
            ):
                stages["read"] = _stage(False, "s3_read_content_mismatch")
                return _full_result(
                    ok=False,
                    bucket=stages["bucket"],
                    write=stages["write"],
                    read=stages["read"],
                    delete=_not_reached(),
                    error_code="s3_read_content_mismatch",
                )
        except Exception as exc:
            return _failure("read", _classify(exc), stages)
        stages["read"] = _stage(True)

        # 阶段 5：删除临时对象；删除失败必须报告，不得折叠为成功。
        try:
            client.remove_object(bucket, probe_name)
        except Exception as exc:
            stages["delete"] = _stage(False, _classify(exc))
            return _full_result(
                ok=False,
                bucket=stages["bucket"],
                write=stages["write"],
                read=stages["read"],
                delete=stages["delete"],
                error_code=stages["delete"].error_code,
            )
        stages["delete"] = _stage(True)
        cleanup_pending = False  # 已显式删除，finally 不再重复清理。
        return _full_result(
            ok=True,
            bucket=stages["bucket"],
            write=stages["write"],
            read=stages["read"],
            delete=stages["delete"],
            error_code=None,
        )
    finally:
        # 尽力清理；清理失败不影响主流程已计算并返回的阶段结果（删除阶段
        # 在主流程显式执行并报告，这里只是异常路径的兜底删除）。
        if cleanup_pending and probe_name is not None:
            try:
                client.remove_object(bucket, probe_name)
            except Exception:
                pass


def _close_response(response) -> None:
    close = getattr(response, "close", None)
    release = getattr(response, "release_conn", None)
    if callable(close):
        close()
    if callable(release):
        release()


def _forbid_redirects(client) -> None:
    """在 client 的底层 ``_url_open`` 前拦截一切 3xx 响应，禁止跟随重定向。

    重定向目标未经受控 DNS 验证，跟随等于把凭据送往任意地址；拦截后以
    稳定错误码失败，响应体/Location 头不进入任何日志或结果。
    """
    url_open = getattr(client, "_url_open", None)
    if not callable(url_open):
        return

    def _guarded(*args, **kwargs):
        response = url_open(*args, **kwargs)
        status = getattr(response, "status", None)
        if isinstance(status, int) and 300 <= status < 400:
            _close_response(response)
            raise _RedirectBlocked()
        return response

    client._url_open = _guarded


def _probe_prefix(prefix: str) -> str:
    return (prefix or "").strip("/")


def _probe_object_name(prefix: str) -> str:
    probe_prefix = _probe_prefix(prefix)
    name = f"_connection-tests/probe-{secrets.token_hex(12)}"
    return f"{probe_prefix}/{name}" if probe_prefix else name


def _is_auth_failure(exc: Exception) -> bool:
    code = str(getattr(exc, "code", "")).lower()
    # 只按结构化 ``code`` 判定，不解析异常文本，避免把凭据/endpoint 带进日志。
    return code in {
        "accessdenied",
        "invalidaccesskeyid",
        "signaturedoesnotmatch",
    }
