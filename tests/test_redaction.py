from __future__ import annotations

from analysis.redaction import (
    RedactionSettings,
    TextRedactor,
    redact_pii,
    redact_sensitive_terms,
)


def test_redact_pii_email_and_phone():
    text = "Contact me at jane.doe@example.com or +1-202-555-0147."
    redacted = redact_pii(text)
    assert "jane.doe@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted


def test_redact_sensitive_terms():
    text = "The launch code is ORCHID and the project name is Zephyr."
    redacted = redact_sensitive_terms(text, ["orchid", "zephyr"])
    assert "ORCHID" not in redacted
    assert redacted.count("[REDACTED_TERM]") == 2


def test_text_redactor_combines_rules():
    settings = RedactionSettings(
        enable_redaction=True, redact_pii=True, sensitive_terms=["apollo"]
    )
    redactor = TextRedactor(settings)
    text = "Email apollo lead at lead@company.com for details."
    result = redactor.redact(text)
    assert "lead@company.com" not in result
    assert "[REDACTED_EMAIL]" in result
    assert "apollo" not in result.lower()
