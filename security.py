"""
Security & Guardrails Module:
1. HMAC-SHA256 Webhook Signatures: Cryptographically signs outgoing event payloads to guarantee authenticity.
2. Prompt Injection Guardrails: Detects and quarantines adversarial injection payloads in agent posts.
"""

import hmac
import hashlib
import time
import json
import re
from typing import Tuple, Dict, Any

# Shared system signing key (in production, dynamically managed per-agent via Vault/KMS)
SYSTEM_HMAC_SECRET = "asama_network_secret_key_v1_secure"

# Common prompt injection triggers and adversarial patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(the\s+)?above",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"output\s+(your\s+)?(system\s+prompt|instructions|secret|api\s*key)",
    r"reveal\s+confidential",
    r"bypass\s+safety",
    r"roleplay\s+as\s+an\s+unrestricted",
    r"jailbreak",
    r"system\s*:\s*admin"
]


def sign_payload(payload: Dict[str, Any], secret_key: str = SYSTEM_HMAC_SECRET) -> Tuple[str, int]:
    """
    Signs a JSON webhook payload with an HMAC-SHA256 signature and timestamp.
    Returns (hex_signature, timestamp).
    """
    timestamp = int(time.time())
    canonical_body = json.dumps(payload, sort_keys=True)
    message = f"{timestamp}:{canonical_body}".encode('utf-8')
    signature = hmac.new(secret_key.encode('utf-8'), message, hashlib.sha256).hexdigest()
    return signature, timestamp


def verify_signature(payload: Dict[str, Any], signature: str, timestamp: int, secret_key: str = SYSTEM_HMAC_SECRET, max_age_seconds: int = 300) -> bool:
    """
    Verifies that the HMAC-SHA256 signature matches and has not expired (replay protection).
    """
    current_time = int(time.time())
    if abs(current_time - timestamp) > max_age_seconds:
        return False  # Expired or replay attack

    canonical_body = json.dumps(payload, sort_keys=True)
    message = f"{timestamp}:{canonical_body}".encode('utf-8')
    expected_signature = hmac.new(secret_key.encode('utf-8'), message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected_signature, signature)


def sanitize_and_check_guardrails(title: str, content: str) -> Tuple[bool, str]:
    """
    Inspects post title and content for malicious prompt injection attempts.
    Returns (is_safe: bool, reason: str).
    """
    full_text = f"{title}\n{content}".lower()

    for pattern in INJECTION_PATTERNS:
        match = re.search(pattern, full_text)
        if match:
            return False, f"Prompt injection detected matching pattern: '{match.group(0)}'"

    # Check for suspicious script injection or HTML payloads
    if "<script" in full_text or "javascript:" in full_text:
        return False, "Malicious HTML/Script tags detected in content"

    return True, "Passed guardrail sanitization"
