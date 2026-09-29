from edge_memory.policy import pii


def test_detects_email():
    assert "email" in pii.detect("contact me at jane.doe@example.com")


def test_detects_phone_and_ssn():
    cats = pii.detect("call 415-555-0134 or ref 123-45-6789")
    assert "phone" in cats
    assert "ssn" in cats


def test_clean_text_has_no_pii():
    assert not pii.contains_pii("torque spec for pump X-200 is 45 Nm")


def test_redaction_masks_email():
    assert "[EMAIL]" in pii.redact("mail a@b.com now")
    assert "a@b.com" not in pii.redact("mail a@b.com now")
