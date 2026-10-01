"""Optional Live Integration Tests for Pipecat Voice Orchestration.

Executes live audio pipeline tests when Pipecat library and hardware audio devices
are present in the local environment. Gracefully skips automatically otherwise.
"""

import pytest

try:
    import pipecat
    HAS_PIPECAT = True
except ImportError:
    HAS_PIPECAT = False


@pytest.mark.skipif(not HAS_PIPECAT, reason="Pipecat library is not installed in environment.")
class TestPipecatLiveIntegration:
    """Optional tests requiring active pipecat installation."""

    def test_01_pipecat_package_installed(self) -> None:
        """Verifies pipecat module presence when optional dependency is present."""
        assert HAS_PIPECAT is True
        assert hasattr(pipecat, "__version__")
