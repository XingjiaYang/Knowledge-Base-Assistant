from __future__ import annotations

from hmac import compare_digest

from app.config import Settings


def bearer_token(authorization: str | None) -> str | None:
    scheme, separator, provided_token = (authorization or "").partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not provided_token:
        return None
    return provided_token


def is_authorized_api_request(
    config: Settings,
    authorization: str | None,
) -> bool:
    token = config.api_auth_token
    if not token:
        return True

    provided_token = bearer_token(authorization)
    if provided_token is None:
        return False
    return compare_digest(provided_token, token)
