"""Installed Phase 8 SDK must match the declared structured-output contract."""

from __future__ import annotations

import inspect
import re
from importlib.metadata import version
from pathlib import Path

from anthropic.resources.messages import Messages


def test_declared_anthropic_pin_matches_installed_stable_output_config_api():
    """A clean install cannot silently select an SDK without stable output_config."""
    requirements = (Path(__file__).resolve().parents[1] / "requirements-ml.txt").read_text(
        encoding="utf-8"
    )
    match = re.search(r"(?m)^anthropic==([^\s#]+)", requirements)

    assert match is not None
    assert match.group(1) == version("anthropic")
    assert "output_config" in inspect.signature(Messages.create).parameters
