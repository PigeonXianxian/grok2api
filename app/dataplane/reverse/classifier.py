"""Reverse pipeline result classifier.

Maps upstream HTTP status codes and response bodies to a ResultCategory.
Improved CF challenge detection: checks HTML body patterns for real CF pages,
not just keyword matching that can false-positive on JSON error responses.
"""

from typing import Any

from app.dataplane.reverse.protocol.xai_usage import is_invalid_credentials_body

from .types import ResultCategory

# Cloudflare challenge HTML signatures — checked case-insensitively
_CF_SIGNATURES = (
    "cf-challenge",
    "cf-browser-verification",
    "just a moment",
    "checking your browser",
    "cf-wrapper",
    "/cdn-cgi/challenge-platform",
)


def _is_cloudflare_challenge(body: str) -> bool:
    """Check if response body indicates a real Cloudflare challenge page.

    Only matches when the body contains HTML-specific CF patterns,
    avoiding false positives on JSON API error responses that may
    contain the word 'cloudflare' in their error message.
    """
    if not body:
        return False
    lower = body.lower()
    if "text/html" not in lower[:2000] and not any(
        sig in lower for sig in _CF_SIGNATURES
    ):
        return False
    return any(sig in lower for sig in _CF_SIGNATURES)


def classify_result(
    status_code: int,
    body: str = "",
    *,
    payload: Any = None,
) -> ResultCategory:
    """Classify an upstream response into a ResultCategory.

    ``body`` is the raw response body (or first ~400 chars for error responses).
    ``payload`` is the parsed JSON, if available.
    """
    if status_code == 200:
        return ResultCategory.SUCCESS

    if status_code == 429:
        return ResultCategory.RATE_LIMITED

    if status_code == 401:
        return ResultCategory.AUTH_FAILURE

    if status_code == 400 and is_invalid_credentials_body(body):
        return ResultCategory.AUTH_FAILURE

    if status_code == 403:
        # Known blocked/invalid account markers take precedence.
        if is_invalid_credentials_body(body):
            return ResultCategory.AUTH_FAILURE
        # Check body for real Cloudflare challenge page (HTML patterns).
        if _is_cloudflare_challenge(body):
            return ResultCategory.FORBIDDEN
        # Generic 403: permission error, suspension, WAF — treat as forbidden
        # but NOT as a CF challenge, avoiding unnecessary clearance invalidation.
        return ResultCategory.FORBIDDEN

    if status_code == 404:
        return ResultCategory.NOT_FOUND

    if status_code >= 500:
        return ResultCategory.UPSTREAM_5XX

    return ResultCategory.UNKNOWN


__all__ = ["classify_result", "_is_cloudflare_challenge"]
