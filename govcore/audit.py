"""A tamper-evident audit trail for retrieval decisions.

CPS 230 is about operational risk: when something goes wrong you have to be
able to reconstruct what the system did and show the record was not edited
after the fact. So every answer-or-abstain decision writes one record carrying
who asked, what was retrieved and at what scores, whether the system answered or
abstained and why, which sources it cited, and how many passages were withheld
on access grounds.

The records form a hash chain: each one commits to the hash of the previous
record, so removing or altering any entry breaks every entry after it and
``verify`` catches it. This is a genuine integrity check, not encryption and not
a substitute for a real append-only store, but it makes silent tampering
detectable, which is the property an auditor actually asks for.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

GENESIS = "0" * 64

# Fields that are hashed into the chain. entry_hash itself is excluded.
_CHAINED = ("seq", "decision_id", "ts", "principal", "clearance", "question",
            "decision", "abstain_reason", "retrieved", "cited",
            "access_withheld", "pii_redactions", "model", "corpus",
            "prev_hash")


def _canonical(entry: dict) -> str:
    payload = {k: entry.get(k) for k in _CHAINED}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _hash_entry(entry: dict) -> str:
    return hashlib.sha256(_canonical(entry).encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only, hash-chained decision log. Optionally mirrored to a JSONL
    file so it survives a restart."""

    def __init__(self, path: Optional[str] = None, clock=None):
        self.path = path
        self._clock = clock or (lambda: time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._entries: list[dict] = []
        if path and os.path.exists(path):
            self._load(path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    self._entries.append(json.loads(line))

    @property
    def last_hash(self) -> str:
        return self._entries[-1]["entry_hash"] if self._entries else GENESIS

    def record(self, *, principal: str, clearance: str, question: str,
               decision: str, abstain_reason: str = "",
               retrieved: Optional[list] = None, cited: Optional[list] = None,
               access_withheld: int = 0, pii_redactions: int = 0,
               model: str = "", corpus: str = "") -> dict:
        """Append one decision record and return it (including its hash)."""
        entry = {
            "seq": len(self._entries),
            "decision_id": uuid.uuid4().hex[:12],
            "ts": self._clock(),
            "principal": principal,
            "clearance": clearance,
            "question": question,
            "decision": decision,
            "abstain_reason": abstain_reason,
            "retrieved": retrieved or [],
            "cited": cited or [],
            "access_withheld": access_withheld,
            "pii_redactions": pii_redactions,
            "model": model,
            "corpus": corpus,
            "prev_hash": self.last_hash,
        }
        entry["entry_hash"] = _hash_entry(entry)
        self._entries.append(entry)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def entries(self) -> list[dict]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def verify(self) -> tuple[bool, str]:
        """Recompute the chain. Returns (ok, detail). Detects any altered,
        inserted, removed, or reordered record."""
        prev = GENESIS
        for i, entry in enumerate(self._entries):
            if entry.get("seq") != i:
                return False, "sequence break at index %d" % i
            if entry.get("prev_hash") != prev:
                return False, "broken link at seq %d" % i
            if _hash_entry(entry) != entry.get("entry_hash"):
                return False, "altered content at seq %d" % i
            prev = entry["entry_hash"]
        return True, "intact: %d record(s) verified" % len(self._entries)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e, ensure_ascii=False) for e in self._entries)
