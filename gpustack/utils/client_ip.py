import ipaddress
from typing import Any, Iterable, Optional


def get_client_ip(
    request: Any, trusted_proxy_cidrs: Optional[Iterable[str]] = None
) -> Optional[str]:
    client_host = request.client.host if request.client else None
    if not _is_trusted_proxy(client_host, trusted_proxy_cidrs):
        return client_host

    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        ip = _first_forwarded_for_ip(forwarded_for)
        if ip:
            return ip

    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()

    forwarded = request.headers.get("forwarded")
    if forwarded:
        ip = _forwarded_header_ip(forwarded)
        if ip:
            return ip

    return client_host


def _is_trusted_proxy(
    client_host: Optional[str], trusted_proxy_cidrs: Optional[Iterable[str]]
) -> bool:
    if not client_host or not trusted_proxy_cidrs:
        return False

    try:
        client_ip = ipaddress.ip_address(client_host)
    except ValueError:
        return False

    for cidr in trusted_proxy_cidrs:
        try:
            if client_ip in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            continue

    return False


def _first_forwarded_for_ip(header_value: str) -> Optional[str]:
    for item in header_value.split(","):
        ip = item.strip()
        if ip:
            return _clean_forwarded_ip(ip)
    return None


def _forwarded_header_ip(header_value: str) -> Optional[str]:
    for part in header_value.split(";"):
        key, separator, value = part.strip().partition("=")
        if separator and key.lower() == "for":
            return _clean_forwarded_ip(value)
    return None


def _clean_forwarded_ip(value: str) -> str:
    value = value.strip().strip('"')
    if value.startswith("[") and "]" in value:
        return value[1 : value.index("]")]
    if ":" in value and value.count(":") == 1:
        return value.split(":", 1)[0]
    return value
