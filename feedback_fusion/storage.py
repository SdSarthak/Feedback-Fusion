"""Optional MongoDB persistence.

The original app opened a MongoClient at import time and called
`insert_many` unconditionally, so every analysis crashed on machines where
`mongod` was not running. Storage is now best-effort: if Mongo is
unreachable the analysis still completes and the UI reports that results
were not persisted.
"""

import re

from .config import settings

_client = None
_client_checked = False

# Mongo collection names cannot contain '$' or NUL and must be non-empty.
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class StorageResult:
    """Outcome of a persistence attempt."""

    def __init__(self, stored, collection=None, count=0, reason=None):
        self.stored = stored
        self.collection = collection
        self.count = count
        self.reason = reason

    def __bool__(self):
        return self.stored

    def __repr__(self):
        if self.stored:
            return f"<StorageResult stored {self.count} into {self.collection}>"
        return f"<StorageResult skipped: {self.reason}>"


def safe_collection_name(prefix, identifier):
    """Build a Mongo-safe collection name from arbitrary user input."""
    slug = _UNSAFE_RE.sub("_", str(identifier or "").strip()).strip("_")
    slug = slug[:80] or "unknown"
    return f"{prefix}_{slug}"


def get_client():
    """Return a MongoClient, or None when Mongo is unavailable/disabled.

    The connection is probed once with a short timeout; pymongo otherwise
    defers connection errors until the first operation, which would surface
    as a confusing 30-second hang inside the UI.
    """
    global _client, _client_checked
    if not settings.storage_enabled:
        return None
    if _client_checked:
        return _client

    _client_checked = True
    try:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError

        candidate = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=1500)
        candidate.admin.command("ping")
        _client = candidate
    except Exception:
        _client = None
    return _client


def store_results(results, prefix, identifier, text_field="text"):
    """Persist analysis results, returning a StorageResult either way."""
    if not results:
        return StorageResult(False, reason="nothing to store")

    if not settings.storage_enabled:
        return StorageResult(False, reason="storage disabled via STORAGE_ENABLED")

    client = get_client()
    if client is None:
        return StorageResult(
            False, reason=f"could not reach MongoDB at {settings.mongo_uri}"
        )

    collection_name = safe_collection_name(prefix, identifier)
    records = [
        r.as_record(text_field) if hasattr(r, "as_record") else dict(r) for r in results
    ]
    try:
        client[settings.mongo_db][collection_name].insert_many(records)
    except Exception as exc:
        return StorageResult(False, collection_name, reason=str(exc))

    return StorageResult(True, collection_name, count=len(records))


def reset_client_cache():
    """Clear the memoised client. Used by tests."""
    global _client, _client_checked
    _client = None
    _client_checked = False
