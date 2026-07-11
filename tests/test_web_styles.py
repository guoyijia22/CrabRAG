from pathlib import Path
import re


def test_primary_page_actions_override_the_generic_button_background():
    styles = Path("apps/web/src/styles.css").read_text(encoding="utf-8")

    assert re.search(
        r"[^{}]*\.page-actions\s+button\.primary-button[^{}]*\{[^}]*"
        r"background:\s*var\(--accent\)[^}]*color:\s*white",
        styles,
        re.DOTALL,
    )
