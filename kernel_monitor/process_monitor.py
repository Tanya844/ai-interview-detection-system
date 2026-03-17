# code for monitoring system processes and activities

import psutil
import pyperclip
import time
import json
from datetime import datetime

# List of suspicious AI-related process names
SUSPICIOUS_PROCESSES = [
    "chatgpt", "copilot", "gemini", "claude", "perplexity",
    "notion", "grammarly", "jasper", "writesonic",
    "chrome", "firefox", "msedge", "brave",  # browsers (flag if open)
]

def get_running_processes():
    """Returns list of currently running process names (lowercase)."""
    running = []
    for proc in psutil.process_iter(['name']):
        try:
            running.append(proc.info['name'].lower())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return running

def detect_suspicious_apps(running_processes):
    """Check if any suspicious AI tools are running."""
    found = []
    for proc in running_processes:
        for sus in SUSPICIOUS_PROCESSES:
            if sus in proc:
                found.append(proc)
                break
    return list(set(found))

def monitor_clipboard(duration_seconds=5, interval=0.5):
    """
    Monitor clipboard for sudden large text changes (possible AI paste).
    Returns list of clipboard events detected.
    """
    events = []
    last_clipboard = ""
    start = time.time()

    try:
        last_clipboard = pyperclip.paste()
    except:
        last_clipboard = ""

    while time.time() - start < duration_seconds:
        time.sleep(interval)
        try:
            current = pyperclip.paste()
        except:
            current = ""

        if current != last_clipboard and len(current) > 30:
            events.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "clipboard_length": len(current),
                "preview": current[:80]  # first 80 chars only
            })
        last_clipboard = current

    return events

def run_kernel_monitor(duration=60):
    """
    Run full kernel monitor for given duration (seconds).
    Returns a dict of findings.
    """
    print("[Kernel Monitor] Starting...")
    
    # Check processes
    running = get_running_processes()
    suspicious = detect_suspicious_apps(running)
    
    # Monitor clipboard
    print("[Kernel Monitor] Watching clipboard...")
    clipboard_events = monitor_clipboard(duration_seconds=duration)

    result = {
        "suspicious_processes": suspicious,
        "clipboard_events": clipboard_events,
        "clipboard_paste_count": len(clipboard_events),
        "cheating_flag": len(suspicious) > 0 or len(clipboard_events) > 2
    }

    print("[Kernel Monitor] Done.")
    print(json.dumps(result, indent=2))
    return result

# Run standalone test
if __name__ == "__main__":
    result = run_kernel_monitor(duration=10)  # 10 sec test