"""A reference PII detector and redactor.

This is a regex and checksum layer, not a full data-loss-prevention product. It
catches the structured identifiers that leak most often through a RAG answer:
email addresses, phone numbers, card-like numbers (validated with the Luhn
check so random digit runs are not flagged), Australian Tax File Number-shaped
numbers, and IP addresses. It does not attempt names or addresses, which need a
trained model; claiming otherwise would be dishonest. Its job here is defence in
depth: redact identifiers out of retrieved context before the model ever sees
them, and record how many were removed in the audit trail.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?61|0)\d(?:[ \-]?\d){7,9}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)")
_TFN = re.compile(r"(?<!\d)\d{3}[ \-]?\d{3}[ \-]?\d{3}(?!\d)")
_IP = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")


@dataclass
class Finding:
    kind: str
    value: str
    start: int
    end: int


def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 13:
        return False
    total, parity = 0, len(nums) % 2
    for i, d in enumerate(nums):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def detect(text: str) -> list[Finding]:
    """Find structured PII. Order matters: more specific patterns first so a
    card number is not also reported as a phone number."""
    found: list[Finding] = []
    taken: list[tuple[int, int]] = []

    def _overlaps(s: int, e: int) -> bool:
        return any(s < te and e > ts for ts, te in taken)

    def _scan(rx, kind, validate=None):
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            if _overlaps(s, e):
                continue
            val = m.group()
            if validate and not validate(val):
                continue
            found.append(Finding(kind, val, s, e))
            taken.append((s, e))

    _scan(_EMAIL, "email")
    _scan(_CARD, "card", _luhn_ok)
    _scan(_TFN, "tfn")
    _scan(_IP, "ip")
    _scan(_PHONE, "phone")
    found.sort(key=lambda f: f.start)
    return found


def redact(text: str, kinds=None) -> tuple[str, list[Finding]]:
    """Replace detected PII with [REDACTED:KIND] markers. Returns the cleaned
    text and the findings removed."""
    findings = detect(text)
    if kinds is not None:
        findings = [f for f in findings if f.kind in kinds]
    out, last = [], 0
    for f in findings:
        out.append(text[last:f.start])
        out.append("[REDACTED:%s]" % f.kind.upper())
        last = f.end
    out.append(text[last:])
    return "".join(out), findings


def redact_chunks(chunks, kinds=None):
    """Return copies of chunks with PII redacted from their text, plus the total
    number of identifiers removed. Originals are left untouched so the audit log
    can still reference them."""
    import copy
    total = 0
    cleaned = []
    for ch in chunks:
        red, findings = redact(ch.text, kinds)
        if findings:
            ch = copy.copy(ch)
            ch.text = red
            total += len(findings)
        cleaned.append(ch)
    return cleaned, total
