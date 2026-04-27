import os
import signal
import subprocess
import sys
import time
import requests
import logging

PORT = 15301
URL = f"http://127.0.0.1:{PORT}"
PID_FILE = "memk_daemon.pid"

logger = logging.getLogger("memk.manager")

def start():
    """Start the daemon if not already running."""
    if is_running():
        print(f"Daemon already running at {URL}")
        return

    print("Starting MemoryKernel Daemon (this may take a few seconds to load the model)...")
    
    # Run uvicorn in a separate process
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "memk.server.daemon:app", "--port", str(PORT), "--host", "127.0.0.1", "--no-access-log"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    )
    
    with open(PID_FILE, "w") as f:
        f.write(str(process.pid))

    # Wait for startup
    timeout = 30
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"{URL}/health", timeout=1)
            if resp.status_code == 200:
                print(f"Daemon started successfully (PID: {process.pid})")
                return
        except:
            time.sleep(0.5)
    
    print("Warning: Daemon startup is taking longer than expected. Check logs if it fails.")

def stop():
    """Stop the daemon via API."""
    if not is_running():
        print("Daemon is not running.")
        return

    try:
        requests.post(f"{URL}/shutdown")
        print("Shutdown command sent.")
    except:
        print("Failed to send shutdown command. Force killing...")
        if os.path.exists(PID_FILE):
            with open(PID_FILE, "r") as f:
                pid = int(f.read())
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Killed process {pid}")
            except:
                pass
    
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

def is_running():
    """Check if the daemon is responsive."""
    try:
        resp = requests.get(f"{URL}/health", timeout=0.5)
        return resp.status_code == 200
    except:
        return False

def get_status():
    if is_running():
        try:
            resp = requests.get(f"{URL}/health").json()
            return f"RUNNING (v{resp.get('version', resp.get('engine', 'unknown'))})"
        except:
            return "RUNNING (unresponsive)"
    return "STOPPED"
