from dataclasses import dataclass

import jwt
from fastapi import Request

from app.core.config import Settings
from app.core.exceptions import AuthError


@dataclass
class Auth:
    user_id: str
    token: str


def authenticate(request: Request, settings: Settings) -> Auth:
    """Validate the Go backend's HS256 access token locally (same JWT_SECRET)."""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise AuthError("missing bearer token")
    token = header.removeprefix("Bearer ").strip()
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            issuer="bookly",
            options={"require": ["exp", "sub"]},
        )
    except jwt.InvalidTokenError as exc:
        raise AuthError(f"invalid token: {exc}") from exc
    return Auth(user_id=claims["sub"], token=token)
