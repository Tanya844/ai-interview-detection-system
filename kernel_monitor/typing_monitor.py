import time
import threading
import json
from pynput import keyboard

# Stores typing data
typing_data = {
    "total_keystrokes": 0,
    "paste_events": 0,
    "keystroke_times": [],   # time between each key press
    "start_time": None,
    "end_time": None,
}

_last_key_time = None
_monitoring = False
_listener = None

def on_press(key):
    global _last_key_time

    if not _monitoring:
        return

    now = time.time()

    # Detect Ctrl+V (paste) on Windows/Linux
    # Detect Cmd+V (paste) on Mac
    try:
        if key == keyboard.Key.ctrl or key == keyboard.Key.cmd:
            pass  # modifier key, skip
        
        # Check for paste combo
        # pynput handles this through HotKey separately
    except:
        pass

    # Record keystroke timing
    if _last_key_time is not None:
        gap = now - _last_key_time
        typing_data["keystroke_times"].append(round(gap, 4))

    _last_key_time = now
    typing_data["total_keystrokes"] += 1


def start_typing_monitor():
    """Start monitoring keyboard in background thread."""
    global _monitoring, _listener
    _monitoring = True
    typing_data["start_time"] = time.time()

    _listener = keyboard.Listener(on_press=on_press)
    _listener.start()
    print("[Typing Monitor] Started.")


def stop_typing_monitor():
    """Stop monitoring and return analysis."""
    global _monitoring, _listener
    _monitoring = False
    typing_data["end_time"] = time.time()

    if _listener:
        _listener.stop()

    return analyze_typing()


def detect_paste_events(answer_text, original_keystroke_count):
    """
    If final answer length >> keystrokes typed, likely copy-pasted.
    Simple heuristic: if typed chars < 40% of answer length = suspicious.
    """
    if len(answer_text) == 0:
        return False
    ratio = original_keystroke_count / max(len(answer_text), 1)
    return ratio < 0.4   # less than 40% typed manually = paste suspected


def analyze_typing():
    """Analyze collected typing data and return cheating signals."""
    times = typing_data["keystroke_times"]
    
    avg_gap = sum(times) / len(times) if times else 0
    
    # Unusually large gap = sudden text dump (possible paste)
    large_gaps = [t for t in times if t > 3.0]  # gap > 3 seconds
    
    result = {
        "total_keystrokes": typing_data["total_keystrokes"],
        "average_keystroke_gap_sec": round(avg_gap, 4),
        "large_gaps_count": len(large_gaps),  # suspicious pauses
        "paste_suspected": len(large_gaps) > 2 or typing_data["total_keystrokes"] < 10,
        "cheating_flag": len(large_gaps) > 2
    }

    print("[Typing Monitor] Analysis:")
    print(json.dumps(result, indent=2))
    return result


# Standalone test
if __name__ == "__main__":
    print("Type something for 15 seconds...")
    start_typing_monitor()
    time.sleep(15)
    result = stop_typing_monitor()