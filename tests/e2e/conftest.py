# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration for e2e tests.

Registers the markers used by tests in this directory so pytest does not
emit PytestUnknownMarkWarning when running standalone.
"""

# Third Party
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end tests that boot real services (vllm serve, etc.). "
        "Slow; opt in with `pytest -m e2e` or run the file directly.",
    )
    config.addinivalue_line(
        "markers",
        "gpu: requires a CUDA-capable GPU. Skipped automatically when no GPU "
        "is visible to torch.cuda.",
    )
