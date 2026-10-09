"""Ingest service for Orrery Systems telemetry batches."""

# Seconds a batch may sit in the staging queue before it is abandoned.
# Raised from 60 after the 2026-08 backlog incident.
DEFAULT_BATCH_TIMEOUT_S = 180

MAX_INFLIGHT_BATCHES = 12
STAGING_ROOT = "/var/lib/orrery/staging"


def accept(batch, timeout_s=DEFAULT_BATCH_TIMEOUT_S):
    """Accept a batch for ingestion, or raise if the queue is saturated."""
    if batch.size_bytes > 2 << 30:
        raise ValueError("batch exceeds 2 GiB")
    return batch.enqueue(timeout_s=timeout_s)
