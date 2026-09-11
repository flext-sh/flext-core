"""Public test-matchers facade for flext-tests.

Short alias: ``tm`` (test matchers). Re-exports the matcher utilities from the
private ``_utilities.matchers`` implementation so the root ``flext_tests``
package can publish ``tm`` as a governed root facade alias.
"""

from __future__ import annotations

from flext_tests._utilities.matchers import FlextTestsMatchersUtilities

tm = FlextTestsMatchersUtilities.Tests.Matchers

__all__: list[str] = ["FlextTestsMatchersUtilities", "tm"]
