from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.session_store import (  # noqa: E402
    clean_session_title,
    hash_password,
    hash_token,
    normalize_username,
    parse_bootstrap_users,
    verify_password,
)


def assert_username_normalization() -> None:
    if normalize_username(" Alice.Admin ") != "alice.admin":
        raise AssertionError("Usernames should be normalized for login.")

    try:
        normalize_username("bad user")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid usernames should be rejected.")

    print("Username normalization -> ok")


def assert_bootstrap_user_parsing() -> None:
    users = parse_bootstrap_users("admin:secret; analyst:another")
    if users != [("admin", "secret"), ("analyst", "another")]:
        raise AssertionError("Bootstrap users should parse username:password pairs.")

    try:
        parse_bootstrap_users("missing-password")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid bootstrap users should be rejected.")

    print("Bootstrap user parsing -> ok")


def assert_password_hashing() -> None:
    password_hash = hash_password("correct horse battery staple")
    if not verify_password("correct horse battery staple", password_hash):
        raise AssertionError("Password hash should verify the original password.")
    if verify_password("wrong", password_hash):
        raise AssertionError("Password hash should reject wrong passwords.")

    print("Password hashing -> ok")


def assert_token_hashing() -> None:
    token_hash = hash_token("session-token")
    if token_hash == "session-token" or len(token_hash) != 64:
        raise AssertionError("Session tokens should be stored as SHA-256 hashes.")

    print("Token hashing -> ok")


def assert_session_title_cleanup() -> None:
    title = clean_session_title("  First\nquestion about Qdrant  ", 20)
    if title != "First question about":
        raise AssertionError("Session titles should be collapsed and bounded.")

    print("Session title cleanup -> ok")


def main() -> None:
    assert_username_normalization()
    assert_bootstrap_user_parsing()
    assert_password_hashing()
    assert_token_hashing()
    assert_session_title_cleanup()


if __name__ == "__main__":
    main()
