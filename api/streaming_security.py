import ipaddress
from typing import List, Optional, Union


IPNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


def parse_allowed_networks(raw_value: Optional[str]) -> List[IPNetwork]:
    """
    Parse comma-separated IPs/CIDRs into network objects.

    Invalid entries are ignored so a single bad token does not break startup.
    """
    if not raw_value:
        return []

    networks: List[IPNetwork] = []
    for token in raw_value.split(","):
        value = token.strip()
        if not value:
            continue
        try:
            if "/" in value:
                networks.append(ipaddress.ip_network(value, strict=False))
            else:
                ip_obj = ipaddress.ip_address(value)
                suffix = "/32" if ip_obj.version == 4 else "/128"
                networks.append(ipaddress.ip_network(f"{value}{suffix}", strict=False))
        except ValueError:
            continue
    return networks


def is_ip_allowed(client_ip: Optional[str], allowed_networks: List[IPNetwork]) -> bool:
    """Return True when allowlist is empty or client_ip belongs to one of the networks."""
    if not allowed_networks:
        return True
    if not client_ip:
        return False
    try:
        ip_obj = ipaddress.ip_address(client_ip)
    except ValueError:
        return False
    return any(ip_obj in network for network in allowed_networks)
