from __future__ import annotations


def bearer_token(authorization: str | None) -> str | None:
    scheme, separator, provided_token = (authorization or "").partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not provided_token:
        return None
    return provided_token
