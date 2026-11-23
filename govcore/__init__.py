"""govcore: the governance layer on top of the RAG pipeline.

Three concerns kept separate from retrieval so each can be read and tested on
its own: access-scoped retrieval (access.py), a tamper-evident decision log
(audit.py), and reference PII redaction (pii.py). The pipeline composes them;
none of them know about each other.
"""
from .access import (AccessDecision, Classification, Principal, ROLES,
                     allow_predicate, can_access, chunk_classification,
                     chunk_compartment, get_principal)
from .audit import AuditLog
from .pii import Finding, detect, redact, redact_chunks

__all__ = [
    "AccessDecision", "Classification", "Principal", "ROLES",
    "allow_predicate", "can_access", "chunk_classification",
    "chunk_compartment", "get_principal",
    "AuditLog",
    "Finding", "detect", "redact", "redact_chunks",
]
