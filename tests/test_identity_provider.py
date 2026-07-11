from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.exceptions import PyJWKClientConnectionError, PyJWKClientError

from services.rag_api.identity import (
    CompositeIdentityProvider,
    IdentityAuthenticationError,
    IdentityProviderUnavailable,
    InternalTokenSet,
    OidcJwtIdentityProvider,
    OidcSettings,
)
from services.rag_api.security import PrincipalContext


ISSUER = "https://id.example.test/tenant"
AUDIENCE = "crabrag"


class StaticJwksClient:
    def __init__(self, key) -> None:
        self.key = key
        self.tokens: list[str] = []

    def get_signing_key_from_jwt(self, token: str):
        self.tokens.append(token)
        return SimpleNamespace(key=self.key)


class StaticProvider:
    def __init__(self, principal: PrincipalContext) -> None:
        self.principal = principal
        self.calls = 0

    def authenticate(self, _headers) -> PrincipalContext:
        self.calls += 1
        return self.principal


@pytest.fixture()
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _settings() -> OidcSettings:
    return OidcSettings(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_url=f"{ISSUER}/.well-known/jwks.json",
        algorithms=("RS256",),
        roles_claim="roles",
        groups_claim="groups",
        permission_revision_claim="permission_revision",
        admin_claim="crabrag_admin",
    )


def _token(signing_key, **overrides) -> str:
    now = datetime.now(timezone.utc)
    claims = {
        "sub": "alice",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "nbf": now - timedelta(seconds=1),
        "exp": now + timedelta(minutes=5),
        "roles": ["reviewer", "admin"],
        "groups": ["north"],
        "permission_revision": "42",
        "crabrag_admin": True,
    }
    claims.update(overrides)
    return jwt.encode(claims, signing_key, algorithm="RS256", headers={"kid": "key-1"})


def test_oidc_provider_validates_jwt_and_maps_principal(signing_key):
    client = StaticJwksClient(signing_key.public_key())
    provider = OidcJwtIdentityProvider(_settings(), jwks_client=client)
    token = _token(signing_key)

    principal = provider.authenticate({"Authorization": f"Bearer {token}"})

    assert principal == PrincipalContext(
        subject="alice",
        roles=("admin", "reviewer"),
        groups=("north",),
        permission_revision="42",
        can_manage_index=True,
    )
    assert client.tokens == [token]


@pytest.mark.parametrize(
    ("claim", "value"),
    [
        ("iss", "https://attacker.invalid"),
        ("aud", "other-service"),
        ("exp", datetime.now(timezone.utc) - timedelta(seconds=1)),
        ("nbf", datetime.now(timezone.utc) + timedelta(minutes=5)),
    ],
)
def test_oidc_provider_rejects_invalid_registered_claims(signing_key, claim, value):
    provider = OidcJwtIdentityProvider(
        _settings(),
        jwks_client=StaticJwksClient(signing_key.public_key()),
    )

    with pytest.raises(IdentityAuthenticationError):
        provider.authenticate({"authorization": f"Bearer {_token(signing_key, **{claim: value})}"})


def test_oidc_provider_rejects_alg_none_before_jwks_lookup(signing_key):
    client = StaticJwksClient(signing_key.public_key())
    provider = OidcJwtIdentityProvider(_settings(), jwks_client=client)
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {"sub": "alice", "iss": ISSUER, "aud": AUDIENCE, "exp": now + timedelta(minutes=5)},
        key="",
        algorithm="none",
        headers={"kid": "key-1"},
    )

    with pytest.raises(IdentityAuthenticationError):
        provider.authenticate({"authorization": f"Bearer {token}"})

    assert client.tokens == []


def test_oidc_provider_rejects_algorithm_outside_allowlist_before_jwks_lookup():
    shared_secret = b"shared-secret-with-at-least-32-bytes"
    client = StaticJwksClient(shared_secret)
    provider = OidcJwtIdentityProvider(_settings(), jwks_client=client)
    now = datetime.now(timezone.utc)
    token = jwt.encode(
        {"sub": "alice", "iss": ISSUER, "aud": AUDIENCE, "exp": now + timedelta(minutes=5)},
        key=shared_secret,
        algorithm="HS256",
        headers={"kid": "key-1"},
    )

    with pytest.raises(IdentityAuthenticationError):
        provider.authenticate({"authorization": f"Bearer {token}"})

    assert client.tokens == []


def test_oidc_provider_rejects_wrong_signature(signing_key):
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    provider = OidcJwtIdentityProvider(
        _settings(),
        jwks_client=StaticJwksClient(other_key.public_key()),
    )

    with pytest.raises(IdentityAuthenticationError):
        provider.authenticate({"authorization": f"Bearer {_token(signing_key)}"})


@pytest.mark.parametrize(
    ("error", "expected_error"),
    [
        (PyJWKClientError("unknown kid"), IdentityAuthenticationError),
        (PyJWKClientConnectionError("jwks unavailable"), IdentityProviderUnavailable),
    ],
)
def test_oidc_provider_fails_closed_when_jwks_lookup_fails(signing_key, error, expected_error):
    class FailingJwksClient:
        def get_signing_key_from_jwt(self, token):
            raise error

    provider = OidcJwtIdentityProvider(_settings(), jwks_client=FailingJwksClient())

    with pytest.raises(expected_error):
        provider.authenticate({"authorization": f"Bearer {_token(signing_key)}"})


@pytest.mark.parametrize("missing_claim", ["sub", "exp"])
def test_oidc_provider_rejects_missing_required_claim(signing_key, missing_claim):
    now = datetime.now(timezone.utc)
    claims = {"sub": "alice", "iss": ISSUER, "aud": AUDIENCE, "exp": now + timedelta(minutes=5)}
    del claims[missing_claim]
    token = jwt.encode(claims, signing_key, algorithm="RS256", headers={"kid": "key-1"})
    provider = OidcJwtIdentityProvider(
        _settings(),
        jwks_client=StaticJwksClient(signing_key.public_key()),
    )

    with pytest.raises(IdentityAuthenticationError):
        provider.authenticate({"authorization": f"Bearer {token}"})


def test_bearer_failure_never_falls_back_to_trusted_gateway(signing_key):
    oidc = OidcJwtIdentityProvider(
        _settings(),
        jwks_client=StaticJwksClient(signing_key.public_key()),
    )
    local = StaticProvider(
        PrincipalContext("local-admin", ("admin",), (), "1", can_manage_index=True)
    )
    provider = CompositeIdentityProvider(oidc=oidc, local=local)

    with pytest.raises(IdentityAuthenticationError):
        provider.authenticate({"authorization": "Bearer not-a-jwt"})

    assert local.calls == 0


def test_local_identity_provider_remains_compatible_when_oidc_is_disabled():
    expected = PrincipalContext("local-admin", ("admin",), (), "1", True)
    local = StaticProvider(expected)
    provider = CompositeIdentityProvider(oidc=None, local=local)

    assert provider.authenticate({}) == expected
    assert local.calls == 1


def test_enterprise_mode_without_bearer_is_anonymous(signing_key):
    provider = CompositeIdentityProvider(
        oidc=OidcJwtIdentityProvider(
            _settings(),
            jwks_client=StaticJwksClient(signing_key.public_key()),
        ),
        local=StaticProvider(PrincipalContext("local-admin", (), (), "1", True)),
    )

    assert provider.authenticate({}) == PrincipalContext.anonymous()


def test_internal_token_set_accepts_current_and_unexpired_previous():
    now = datetime(2026, 7, 12, 0, 0, tzinfo=timezone.utc)
    tokens = InternalTokenSet(
        current="current-token",
        previous="previous-token",
        previous_valid_until=now + timedelta(minutes=5),
    )

    assert tokens.is_valid("current-token", now=now)
    assert tokens.is_valid("previous-token", now=now)
    assert not tokens.is_valid("wrong-token", now=now)
    assert not tokens.is_valid("previous-token", now=now + timedelta(minutes=5))


def test_internal_token_set_rejects_malformed_previous_expiry(monkeypatch):
    monkeypatch.setenv("CRABRAG_INTERNAL_TOKEN", "current-token")
    monkeypatch.setenv("CRABRAG_INTERNAL_TOKEN_PREVIOUS", "previous-token")
    monkeypatch.setenv("CRABRAG_INTERNAL_TOKEN_PREVIOUS_VALID_UNTIL", "not-a-time")

    with pytest.raises(IdentityProviderUnavailable):
        InternalTokenSet.from_environment()
