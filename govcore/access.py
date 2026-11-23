"""Access-scoped retrieval: a reference access-control model for RAG.

The point Sasi raised is that access control is only half the job. Keeping a
user away from a document is one control; the harder control is what the model
is allowed to SAY once a document is in reach. This module handles the first
half honestly, and the pipeline wires it to the second half: when the only
passages that would answer a question sit above the caller's clearance, the
retriever hands back nothing and the system abstains instead of answering from
data the caller was never cleared to see.

This is a reference model, not a production entitlements engine. It encodes a
classic Bell-LaPadula "no read up" rule over an ordered set of classification
levels, plus optional need-to-know compartments. Real deployments resolve
clearances from an identity provider; here a small fixed roster of roles stands
in so the behaviour is visible and testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional


class Classification(IntEnum):
    """Ordered sensitivity levels. Higher means more restricted."""
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3

    @classmethod
    def parse(cls, value) -> "Classification":
        if isinstance(value, Classification):
            return value
        if isinstance(value, bool):  # guard: bool is an int subclass
            return cls.PUBLIC
        if isinstance(value, int):
            return cls(value) if value in cls._value2member_map_ else cls.RESTRICTED
        if isinstance(value, str):
            key = value.strip().upper()
            if key in cls.__members__:
                return cls[key]
            if key.isdigit():
                return cls.parse(int(key))
        # Unknown label: fail closed to the most restrictive level.
        return cls.RESTRICTED

    @property
    def label(self) -> str:
        return self.name.capitalize()


@dataclass(frozen=True)
class Principal:
    """Who is asking. A role name, a clearance level, and the need-to-know
    compartments they hold."""
    role: str
    clearance: Classification = Classification.PUBLIC
    compartments: frozenset = field(default_factory=frozenset)

    @property
    def label(self) -> str:
        comp = (" + {%s}" % ", ".join(sorted(self.compartments))
                if self.compartments else "")
        return "%s (%s%s)" % (self.role, self.clearance.label, comp)


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str


def can_access(principal: Principal, classification, compartment: Optional[str] = None
               ) -> AccessDecision:
    """Decide whether a principal may read an item. No read up: the principal's
    clearance must meet or exceed the item's classification, and they must hold
    any compartment the item requires."""
    level = Classification.parse(classification)
    if principal.clearance < level:
        return AccessDecision(
            False, "clearance %s below required %s"
            % (principal.clearance.label, level.label))
    if compartment and compartment not in principal.compartments:
        return AccessDecision(False, "missing need-to-know compartment %s" % compartment)
    return AccessDecision(True, "cleared at %s" % level.label)


def chunk_classification(chunk) -> Classification:
    """Read a chunk's classification, defaulting to PUBLIC when unset."""
    raw = getattr(chunk, "classification", None)
    if raw is None:
        raw = getattr(chunk, "meta", {}).get("classification", 0)
    return Classification.parse(raw)


def chunk_compartment(chunk) -> Optional[str]:
    comp = getattr(chunk, "compartment", None)
    if comp is None:
        comp = getattr(chunk, "meta", {}).get("compartment")
    return comp or None


def allow_predicate(principal: Principal) -> Callable[[object], bool]:
    """Build a chunk -> bool filter for the retriever from a principal."""
    def _allow(chunk) -> bool:
        return can_access(principal, chunk_classification(chunk),
                          chunk_compartment(chunk)).allowed
    return _allow


# A small fixed roster so the access behaviour is visible in the demo and tests.
# In a real system these come from the identity provider, not a dict.
ROLES = {
    "public_user": Principal("public_user", Classification.PUBLIC),
    "staff": Principal("staff", Classification.INTERNAL),
    "risk_officer": Principal("risk_officer", Classification.CONFIDENTIAL,
                              frozenset({"RISK"})),
    "executive": Principal("executive", Classification.RESTRICTED,
                           frozenset({"RISK", "RETAIL", "FINANCE"})),
}

DEFAULT_ROLE = "executive"


def get_principal(role: Optional[str]) -> Principal:
    """Look up a roster role; unknown or empty falls back to the most-cleared
    principal so the ungoverned path behaves exactly as before."""
    if not role:
        return ROLES[DEFAULT_ROLE]
    return ROLES.get(role, ROLES[DEFAULT_ROLE])
