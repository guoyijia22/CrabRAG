from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Protocol

import jwt
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError, PyJWKClientConnectionError, PyJWKClientError

from services.rag_api.security import PrincipalContext, principal_from_headers


class IdentityAuthenticationError(RuntimeError):
    """The supplied identity credential is invalid."""


class IdentityProviderUnavailable(RuntimeError):
    """The configured identity provider cannot safely authenticate requests."""


class IdentityProvider(Protocol):
    def authenticate(self, headers: Mapping[str, str]) -> PrincipalContext: ...


class JwksClient(Protocol):
    def get_signing_key_from_jwt(self, token: str): ...


@dataclass(frozen=True)
class OidcSettings:
    issuer: str
    audience: str
    jwks_url: str
    algorithms: tuple[str, ...] = ("RS256",)
    roles_claim: str = "roles"
    groups_claim: str = "groups"
    permission_revision_claim: str = "permission_revision"
    admin_claim: str = "crabrag_admin"
    timeout_seconds: float = 5.0

    @classmethod
    def from_environment(cls) -> "OidcSettings | None":
        issuer = os.getenv("CRABRAG_OIDC_ISSUER", "").strip()
        audience = os.getenv("CRABRAG_OIDC_AUDIENCE", "").strip()
        jwks_url = os.getenv("CRABRAG_OIDC_JWKS_URL", "").strip()
        if not any((issuer, audience, jwks_url)):
            return None
        if not all((issuer, audience, jwks_url)):
            raise IdentityProviderUnavailable("OIDC 配置不完整，已拒绝身份认证")
        algorithms = _csv_tuple(os.getenv("CRABRAG_OIDC_ALGORITHMS", "RS256"))
        if not algorithms:
            raise IdentityProviderUnavailable("OIDC 算法白名单为空，已拒绝身份认证")
        try:
            timeout = float(os.getenv("CRABRAG_OIDC_TIMEOUT_SECONDS", "5"))
        except ValueError as exc:
            raise IdentityProviderUnavailable("OIDC 超时配置无效，已拒绝身份认证") from exc
        if timeout <= 0:
            raise IdentityProviderUnavailable("OIDC 超时配置必须大于 0")
        return cls(
            issuer=issuer,
            audience=audience,
            jwks_url=jwks_url,
            algorithms=algorithms,
            roles_claim=os.getenv("CRABRAG_OIDC_ROLES_CLAIM", "roles").strip() or "roles",
            groups_claim=os.getenv("CRABRAG_OIDC_GROUPS_CLAIM", "groups").strip() or "groups",
            permission_revision_claim=(
                os.getenv("CRABRAG_OIDC_PERMISSION_REVISION_CLAIM", "permission_revision").strip()
                or "permission_revision"
            ),
            admin_claim=os.getenv("CRABRAG_OIDC_ADMIN_CLAIM", "crabrag_admin").strip() or "crabrag_admin",
            timeout_seconds=timeout,
        )


class TrustedGatewayIdentityProvider:
    def __init__(self, internal_token: str | None) -> None:
        self._internal_token = internal_token

    def authenticate(self, headers: Mapping[str, str]) -> PrincipalContext:
        return principal_from_headers(headers, internal_token=self._internal_token)


class OidcJwtIdentityProvider:
    def __init__(self, settings: OidcSettings, *, jwks_client: JwksClient | None = None) -> None:
        self.settings = settings
        self._jwks_client = jwks_client or PyJWKClient(
            settings.jwks_url,
            cache_keys=True,
            timeout=settings.timeout_seconds,
        )

    def authenticate(self, headers: Mapping[str, str]) -> PrincipalContext:
        token = _bearer_token(headers)
        if token is None:
            raise IdentityAuthenticationError("缺少 Bearer Token")
        try:
            header = jwt.get_unverified_header(token)
        except InvalidTokenError as exc:
            raise IdentityAuthenticationError("Bearer Token 格式无效") from exc
        algorithm = str(header.get("alg") or "")
        if algorithm not in self.settings.algorithms:
            raise IdentityAuthenticationError("Bearer Token 签名算法不受信任")
        if not str(header.get("kid") or "").strip():
            raise IdentityAuthenticationError("Bearer Token 缺少 kid")
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token).key
        except PyJWKClientConnectionError as exc:
            raise IdentityProviderUnavailable("OIDC JWKS 服务不可用，已拒绝身份认证") from exc
        except PyJWKClientError as exc:
            raise IdentityAuthenticationError("Bearer Token 的签名密钥不受信任") from exc
        except Exception as exc:  # noqa: BLE001
            raise IdentityProviderUnavailable("OIDC JWKS 解析失败，已拒绝身份认证") from exc
        try:
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=list(self.settings.algorithms),
                audience=self.settings.audience,
                issuer=self.settings.issuer,
                options={"require": ["sub", "exp"]},
            )
        except InvalidTokenError as exc:
            raise IdentityAuthenticationError("Bearer Token 验证失败") from exc
        subject = str(claims.get("sub") or "").strip()
        if not subject:
            raise IdentityAuthenticationError("Bearer Token 缺少有效主体")
        return PrincipalContext(
            subject=subject,
            roles=_claim_tuple(claims.get(self.settings.roles_claim)),
            groups=_claim_tuple(claims.get(self.settings.groups_claim)),
            permission_revision=(
                str(claims.get(self.settings.permission_revision_claim) or "").strip() or "1"
            ),
            can_manage_index=_claim_bool(claims.get(self.settings.admin_claim)),
        )


class CompositeIdentityProvider:
    def __init__(self, *, oidc: OidcJwtIdentityProvider | None, local: IdentityProvider) -> None:
        self._oidc = oidc
        self._local = local

    def authenticate(self, headers: Mapping[str, str]) -> PrincipalContext:
        if _bearer_token(headers) is not None:
            if self._oidc is None:
                raise IdentityAuthenticationError("未配置 OIDC，无法验证 Bearer Token")
            return self._oidc.authenticate(headers)
        if self._oidc is not None:
            return PrincipalContext.anonymous()
        return self._local.authenticate(headers)


def get_identity_provider() -> CompositeIdentityProvider:
    environment = tuple(
        os.getenv(name, "")
        for name in (
            "CRABRAG_INTERNAL_TOKEN",
            "CRABRAG_OIDC_ISSUER",
            "CRABRAG_OIDC_AUDIENCE",
            "CRABRAG_OIDC_JWKS_URL",
            "CRABRAG_OIDC_ALGORITHMS",
            "CRABRAG_OIDC_TIMEOUT_SECONDS",
            "CRABRAG_OIDC_ROLES_CLAIM",
            "CRABRAG_OIDC_GROUPS_CLAIM",
            "CRABRAG_OIDC_PERMISSION_REVISION_CLAIM",
            "CRABRAG_OIDC_ADMIN_CLAIM",
        )
    )
    return _cached_identity_provider(environment)


@lru_cache(maxsize=8)
def _cached_identity_provider(_environment: tuple[str, ...]) -> CompositeIdentityProvider:
    settings = OidcSettings.from_environment()
    return CompositeIdentityProvider(
        oidc=OidcJwtIdentityProvider(settings) if settings else None,
        local=TrustedGatewayIdentityProvider(os.getenv("CRABRAG_INTERNAL_TOKEN")),
    )


def _bearer_token(headers: Mapping[str, str]) -> str | None:
    normalized = {str(key).lower(): str(value) for key, value in headers.items()}
    value = normalized.get("authorization", "").strip()
    if not value:
        return None
    scheme, separator, token = value.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        raise IdentityAuthenticationError("Authorization 必须使用 Bearer Token")
    return token.strip()


def _claim_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return _csv_tuple(value)
    if isinstance(value, (list, tuple, set)):
        return tuple(sorted({str(item).strip() for item in value if str(item).strip()}))
    return ()


def _claim_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(sorted({item.strip() for item in value.split(",") if item.strip()}))
