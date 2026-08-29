"""Lazy registry for benchmark-native backends."""

from __future__ import annotations

from typing import Any

from .base import BenchmarkBackend


def create_backend(slug: str, config: dict[str, Any]) -> BenchmarkBackend:
    kwargs = dict(config.get("backend_kwargs", {}))
    if slug == "bfcl":
        from .bfcl import BFCLBackend

        return BFCLBackend(**kwargs)
    if slug == "browsecompplus":
        from .browsecompplus import BrowseCompPlusBackend

        return BrowseCompPlusBackend(**kwargs)
    raise ValueError(f"No benchmark-native backend registered for {slug!r}")
