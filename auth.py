"""Simple local authentication for the InventoryLens MVP."""

import hashlib
import hmac
import os
import re
import sqlite3

from db import connect, get_or_create_business, now

PBKDF2_ITERATIONS = 200_000


def hash_password(password, salt=None):
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password, stored_hash):
    try:
        salt_hex, digest_hex = stored_hash.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return hmac.compare_digest(actual, expected)


def create_user(business_id, username, password, role):
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,40}", username):
        raise ValueError("Username must be 3-40 letters, numbers, '.', '_' or '-'.")
    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters.")
    if role not in {"owner", "employee"}:
        raise ValueError("Role must be owner or employee.")
    with connect() as db:
        db.execute(
            """
            INSERT INTO users (business_id, username, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (business_id, username, hash_password(password), role, now()),
        )


def get_user(username):
    with connect() as db:
        row = db.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row) if row else None


def authenticate(username, password):
    user = get_user(username.strip().lower())
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return {key: value for key, value in user.items() if key != "password_hash"}


def require_user(user_id, owner=False):
    with connect() as db:
        user = db.execute(
            "SELECT id, business_id, username, role FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if user is None:
        raise PermissionError("Please sign in again.")
    if owner and user["role"] != "owner":
        raise PermissionError("Only an owner can perform this action.")
    return dict(user)


def seed_demo_users():
    business = get_or_create_business()
    for username, password, role in (
        ("owner", "owner123", "owner"), ("employee", "employee123", "employee"),
    ):
        if not get_user(username):
            try:
                create_user(business["id"], username, password, role)
            except sqlite3.IntegrityError:
                if not get_user(username):
                    raise
    return business
