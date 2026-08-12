"""
Real-Time Street Traffic Detection and Classification
CS5330 Pattern Recognition & Computer Vision
Drew Hill & Abhiram Banda

video_source.py

"""
import socket
import subprocess
import threading
from pathlib import Path
from urllib.parse import urlsplit

import configs


def get_source(raw):
    """


    """
    if raw == "relay":
        address = find_phone_address()
        if address is None:
            raise SystemExit("No phone link-local address found - are you on the hotspot?")
        print(f"relay: auto-detected {address}")
        start_relay(address)
        return "http://127.0.0.1:8080/video"
    return resolve_source(raw)


def resolve_source(raw):
    """


    """
    if "://" in raw:
        return raw
    if Path(raw).exists():
        return str(Path(raw).resolve())
    if raw.isdigit():
        return int(raw)
    return f"http://{raw}:{8080}/{"video"}"


def is_network_source(source):
    """


    """
    return isinstance(source, str) and "://" in source


def is_file_source(source):
    """


    """
    return isinstance(source, str) and "://" not in source


def _host_port(source):
    """


    """
    parts = urlsplit(source)
    host = parts.hostname
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None:
        port = {"http": 80, "https": 443, "rtsp": 554}.get((parts.scheme or "").lower())
    return host, port


def check_connection(source, timeout=configs.CONNECTION_TIMEOUT_S):
    """


    """
    host, port = _host_port(source)
    if host is None or port is None or "%" in host:
        return True, ""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, ""
    except ConnectionRefusedError:
        return False, (
            f"Could not connect to {source}\n"
            f"Connection refused at {host}:{port} - the host is reachable but nothing is "
            f"listening. Check that the camera app's server is started and that --port matches."
        )
    except (socket.timeout, OSError):
        return False, (
            f"Could not reach {source}\n"
            f"Host {host}:{port} is unreachable - the phone and laptop are probably not on the "
            f"same network, or the network blocks device-to-device traffic (AP/client isolation).\n"
            f"Verify with: ping {host}"
        )


def find_phone_address():
    """


    """
    output = subprocess.run(["netstat", "-rn"], capture_output=True, text=True).stdout
    for line in output.splitlines():
        cols = line.split()
        if len(cols) >= 2 and cols[0] == "default" and cols[1].startswith("fe80::") and "%" in cols[1]:
            addr, _, zone = cols[1].partition("%")
            if addr != "fe80::":
                return cols[1]
    return None


def start_relay(address, phone_port=8080, local_port=8080):
    """


    """
    host, _, iface = address.partition("%")
    scope = socket.if_nametoindex(iface)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", local_port))
    srv.listen(5)

    def serve():
        while True:
            client, _ = srv.accept()
            threading.Thread(target=handle, args=(client, host, phone_port, scope), daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()


def pump(src, dst):
    """


    """
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def handle(client, host, port, scope):
    """


    """
    up = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        up.connect((host, port, 0, scope))
    except OSError as e:
        print(f"upstream connect failed: {e}")
        client.close()
        return
    threading.Thread(target=pump, args=(client, up), daemon=True).start()
    pump(up, client)
