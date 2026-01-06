import os
import json
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

from .utils import read_jsonl, detect_lang_script_from_filename, write_json


@dataclass
class Record:
    id: str
    persona: str
    instruction: str
    description: str
    lang: str
    script: str
    source_file: str


class DataLoader:
    def __init__(self, data_dir: str, logs_dir: str):
        self.data_dir = data_dir
        self.logs_dir = logs_dir
        self.issues: List[Dict[str, Any]] = []
        self.records: List[Record] = []
        self.by_lang: Dict[str, List[Record]] = {}
        self.by_id: Dict[str, List[Record]] = {}

    def _log_issue(self, file: str, idx: int, reason: str, payload: Dict[str, Any]) -> None:
        self.issues.append({
            "file": file,
            "index": idx,
            "reason": reason,
            "payload": payload,
        })

    def _validate_row(self, row: Dict[str, Any]) -> Optional[Tuple[str, str, str, str]]:
        try:
            rid = str(row["id"]) if "id" in row else None
            p = row.get("indian_context_persona")
            ins = row.get("indian_context_instruction")
            desc = row.get("description", "")
            if not (rid and isinstance(p, str) and isinstance(ins, str)):
                return None
            return rid, p, ins, desc
        except Exception:
            return None

    def load(self) -> None:
        for fn in sorted(os.listdir(self.data_dir)):
            if not fn.endswith(".jsonl"):
                continue
            lang, script = detect_lang_script_from_filename(fn)
            path = os.path.join(self.data_dir, fn)
            if lang is None or script is None:
                # skip unknown patterns but log
                self._log_issue(path, -1, "filename_pattern_mismatch", {"filename": fn})
                continue
            rows = read_jsonl(path)
            for i, row in enumerate(rows):
                valid = self._validate_row(row)
                if valid is None:
                    self._log_issue(path, i, "missing_required_fields", row)
                    continue
                rid, p, ins, desc = valid
                rec = Record(
                    id=rid,
                    persona=p,
                    instruction=ins,
                    description=desc,
                    lang=lang,
                    script=script,
                    source_file=fn,
                )
                self.records.append(rec)
                self.by_lang.setdefault(lang, []).append(rec)
                self.by_id.setdefault(rid, []).append(rec)

        # Save issues log
        if self.issues:
            write_json(os.path.join(self.logs_dir, "data_issues.json"), self.issues)

    def cross_lingual_map(self) -> Dict[str, Dict[str, Record]]:
        # id -> lang -> record
        mp: Dict[str, Dict[str, Record]] = {}
        for rid, recs in self.by_id.items():
            mp[rid] = {r.lang: r for r in recs}
        return mp

    def languages(self) -> List[str]:
        return sorted(self.by_lang.keys())
