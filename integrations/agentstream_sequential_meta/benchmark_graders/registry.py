"""Lazy private-grader registry."""

from __future__ import annotations

from typing import Any

from .base import PrivateGrader


def create_grader(slug: str, config: dict[str, Any]) -> PrivateGrader:
    kwargs = dict(config.get("grader_kwargs", {}))
    if slug == "bfcl":
        from .bfcl import BFCLGrader

        subset = config.get("backend_kwargs", {}).get("subset", "multi_turn_base")
        return BFCLGrader(subset=str(subset), **kwargs)
    if slug == "browsecompplus":
        from .browsecompplus import BrowseCompPlusGrader

        assets_dir = config.get("backend_kwargs", {}).get("assets_dir")
        if assets_dir is not None:
            kwargs.setdefault("assets_dir", assets_dir)
        return BrowseCompPlusGrader(**kwargs)
    raise ValueError(f"No private grader registered for {slug!r}")
