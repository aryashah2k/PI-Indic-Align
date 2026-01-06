import json
import os
import re
import sys
import time
import uuid
import socket
import random
import logging
import warnings
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

ISO_UTC_FMT = "%Y%m%dT%H%M%SZ"


def utc_timestamp() -> str:
    return time.strftime(ISO_UTC_FMT, time.gmtime())


def set_warnings_policy(as_errors: bool = True) -> None:
    if as_errors:
        warnings.filterwarnings("error")
    else:
        warnings.filterwarnings("ignore", category=UserWarning)


def setup_logging(log_dir: str, level: str = "INFO") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("pi_align_indic")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers = []

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logger.level)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, f"run_{utc_timestamp()}.log"))
    fh.setLevel(logger.level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None and hasattr(torch, "manual_seed"):
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]


def get_device(preference: str = "auto") -> str:
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"  # type: ignore[attr-defined]
    # auto
    return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"  # type: ignore[attr-defined]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at {path}:{ln}: {e}")
    return data


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(obj), f, ensure_ascii=False, indent=2, allow_nan=False)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(sanitize_for_json(r), ensure_ascii=False, allow_nan=False) + "\n")


def detect_lang_script_from_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    # Expect pattern: nirdesha_<lang>_<script>.jsonl
    m = re.match(r"^nirdesha_([a-z]{3})_([A-Za-z]+)\.jsonl$", os.path.basename(filename))
    if not m:
        return None, None
    return m.group(1), m.group(2)


def run_id(task: str) -> str:
    return f"{task}-{utc_timestamp()}-{uuid.uuid4().hex[:8]}"


def system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": sys.version.split(" ")[0],
        "time_utc": utc_timestamp(),
    }
    # Try git hash if available
    try:
        gh = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        info["git_hash"] = gh
    except Exception:
        info["git_hash"] = None
    try:
        import importlib.metadata as im

        pkgs = {}
        for pkg in [
            "sentence-transformers",
            "torch",
            "numpy",
            "scikit-learn",
            "ranx",
            "rank-bm25",
            "netcal",
            "rapidfuzz",
        ]:
            try:
                pkgs[pkg] = im.version(pkg)
            except Exception:
                pkgs[pkg] = None
        info["packages"] = pkgs
    except Exception:
        pass

    if torch is not None:
        try:
            info["cuda_available"] = torch.cuda.is_available()  # type: ignore[attr-defined]
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                info["cuda_device_count"] = torch.cuda.device_count()  # type: ignore[attr-defined]
                info["cuda_device"] = torch.cuda.get_device_name(0)  # type: ignore[attr-defined]
        except Exception:
            pass
    return info


def ensure_dirs(base_results: str, base_visualizations: str, base_logs: str) -> None:
    os.makedirs(base_results, exist_ok=True)
    os.makedirs(base_visualizations, exist_ok=True)
    os.makedirs(base_logs, exist_ok=True)


def save_config_snapshot(dst_dir: str, config: Dict[str, Any]) -> None:
    write_json(os.path.join(dst_dir, "config_snapshot.json"), config)


def iso_pair(a: str, b: str) -> str:
    return f"{a}->{b}"


def result_filename(task: str, backbone: str, lang_or_pair: str, split_tag: str, timestamp: str) -> str:
    return f"{task}-{backbone}-{lang_or_pair}-{split_tag}-{timestamp}.json"


def write_visualization_json(path: str, metric: str, axes: Dict[str, Any], data: Any, notes: str = "") -> None:
    payload = {
        "metric": metric,
        "axes": axes,
        "data": data,
        "notes": notes,
    }
    write_json(path, payload)


def sanitize_for_json(obj: Any) -> Any:
    # Recursively replace NaN/Inf with None to create strict JSON
    import math

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_for_json(x) for x in obj)
    return obj
