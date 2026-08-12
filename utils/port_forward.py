"""Auto-manage oc port-forward for EvalHub access from local notebooks."""

import atexit
import os
import signal
import socket
import subprocess
import time


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


_port_forward_proc = None


def ensure_evalhub_port_forward(
    namespace: str = "demo",
    service: str = "evalhub",
    local_port: int = 8443,
    remote_port: int = 8443,
) -> str:
    """Ensure oc port-forward to EvalHub is running.

    If localhost:{local_port} is already reachable, does nothing.
    Otherwise starts `oc port-forward` as a background subprocess.

    Returns the EvalHub base URL (e.g. "https://localhost:8443").
    """
    global _port_forward_proc

    if _is_port_open("localhost", local_port):
        print(f"EvalHub already reachable at localhost:{local_port}")
        return f"https://localhost:{local_port}"

    if _port_forward_proc and _port_forward_proc.poll() is not None:
        print(f"Previous port-forward died (exit={_port_forward_proc.returncode}). Restarting...")
        _port_forward_proc = None

    print(f"Starting port-forward: svc/{service} {local_port}:{remote_port} (ns={namespace})")
    _port_forward_proc = subprocess.Popen(
        ["oc", "port-forward", "-n", namespace, f"svc/{service}",
         f"{local_port}:{remote_port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(15):
        time.sleep(1)
        if _is_port_open("localhost", local_port):
            print(f"Port-forward ready (pid={_port_forward_proc.pid})")
            atexit.register(_stop_port_forward)
            return f"https://localhost:{local_port}"

    raise RuntimeError(
        f"Port-forward failed to start within 15s. "
        f"Run manually: oc port-forward -n {namespace} svc/{service} {local_port}:{remote_port}"
    )


def resolve_evalhub_url(namespace: str = "demo") -> str:
    """Return the EvalHub URL, skipping port-forward when an external URL is configured.

    Priority:
    1. EVALHUB_URL env var pointing to an external Route → use directly
    2. EVALHUB_URL pointing to svc.cluster.local → use directly (in-cluster workbench)
    3. Otherwise → start oc port-forward and return localhost URL
    """
    url = os.getenv("EVALHUB_URL", "").rstrip("/")
    if url and "svc.cluster.local" not in url and not url.startswith("https://localhost"):
        print(f"Using external EvalHub URL: {url}")
        return url
    if url and "svc.cluster.local" in url:
        print(f"Using in-cluster EvalHub URL: {url}")
        return url
    return ensure_evalhub_port_forward(namespace=namespace)


def _stop_port_forward():
    global _port_forward_proc
    if _port_forward_proc and _port_forward_proc.poll() is None:
        _port_forward_proc.send_signal(signal.SIGTERM)
        _port_forward_proc.wait(timeout=5)
        _port_forward_proc = None
