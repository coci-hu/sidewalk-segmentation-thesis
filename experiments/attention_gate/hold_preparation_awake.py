"""Temporary wake request across preparation subprocess transitions only."""
import ctypes
import json
import os
import time
from pathlib import Path

root = Path(__file__).resolve().parent / "step_05_2026-10-14/loader_optimization_001"
function = ctypes.windll.kernel32.SetThreadExecutionState
function.argtypes, function.restype = [ctypes.c_uint], ctypes.c_uint
held = bool(function(0x80000001))
try:
    with (root / "preparation_wake_guard.json").open("x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "held": held, "max_seconds": 3600,
                   "release_file": "RELEASE_PREPARATION_WAKE"}, f)
    started = time.monotonic()
    while held and time.monotonic() - started < 3600 and not (root / "RELEASE_PREPARATION_WAKE").exists():
        time.sleep(5)
finally:
    if held:
        function(0x80000000)
