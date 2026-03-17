import threading
import time
from process_monitor import run_kernel_monitor
from typing_monitor import start_typing_monitor, stop_typing_monitor

def run_member4_monitoring(interview_duration=60):
    """
    Run both monitors during interview.
    Returns combined cheating signals.
    """
    results = {}

    # Start typing monitor in background
    start_typing_monitor()

    # Run kernel monitor in parallel thread
    kernel_result = {}
    def kernel_thread():
        kernel_result.update(run_kernel_monitor(duration=interview_duration))

    t = threading.Thread(target=kernel_thread)
    t.start()

    # Wait for interview to finish
    time.sleep(interview_duration)

    # Stop typing monitor
    typing_result = stop_typing_monitor()
    t.join()

    # Combine signals
    cheating_score = 0
    if kernel_result.get("cheating_flag"):
        cheating_score += 50
    if typing_result.get("cheating_flag"):
        cheating_score += 30
    if kernel_result.get("clipboard_paste_count", 0) > 2:
        cheating_score += 20

    results = {
        "kernel_monitor": kernel_result,
        "typing_monitor": typing_result,
        "cheating_score": min(cheating_score, 100),   # max 100
        "verdict": "SUSPICIOUS" if cheating_score >= 50 else "CLEAN"
    }

    print("\n===== MEMBER 4 FINAL REPORT =====")
    import json
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    run_member4_monitoring(interview_duration=15)  # 15 sec test


## 4. `requirements.txt`

psutil
pyperclip
pynput