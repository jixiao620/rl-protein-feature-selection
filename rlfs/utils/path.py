# rlfs/utils/paths.py
from __future__ import annotations

import os
import time
from typing import Optional


def make_run_dir(base_dir: str = "runs", run_name: Optional[str] = None) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}_{run_name}" if run_name else timestamp
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
