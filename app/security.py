from __future__ import annotations

from hmac import compare_digest

from app.config import Settings


def is_authorized_api_request(
    config: Settings,
    authorization: str | None,
) -> bool:
    token = config.api_auth_token
    if not token:
        return True

    scheme, separator, provided_token = (authorization or "").partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not provided_token:
        return False
    return compare_digest(provided_token, token)
