"""xAI OAuth 登录端点 — /admin/api/xai/oauth/*"""
import secrets
import time
from typing import Any

import orjson
from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import Response
from pydantic import BaseModel

from app.platform.errors import ValidationError
from app.platform.logging.logger import logger
from app.dataplane.reverse.protocol.xai_oauth import (
    generate_pkce_codes,
    discover_oidc,
    build_authorize_url,
    exchange_code_for_tokens,
)

if hasattr(Request, "app"):  # type-check guard
    from app.control.account.repository import AccountRepository
    from app.control.account.refresh import AccountRefreshService

router = APIRouter(tags=["Admin - xAI OAuth"])
public_router = APIRouter(tags=["xAI OAuth"])  # 无认证，用于浏览器回调

_TAG = "Admin - xAI OAuth"

_PENDING_OAUTH: dict[str, Any] = {}


def _get_repo(request: Request) -> "AccountRepository":
    return request.app.state.repository


def _get_refresh_svc(request: Request) -> "AccountRefreshService":
    return request.app.state.refresh_service


def _json(data: Any, status_code: int = 200) -> Response:
    return Response(
        content=orjson.dumps(data),
        media_type="application/json",
        status_code=status_code,
    )


# ── 端点 ──────────────────────────────────────────────────────────────────


class XAIOAuthURLRequest(BaseModel):
    pool: str = "super"
    proxy_url: str = ""


@router.post("/xai/oauth/url")
async def oauth_url(req: XAIOAuthURLRequest):
    """生成 xAI OAuth 授权 URL。"""
    pkce = generate_pkce_codes()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    discovery = await discover_oidc()

    auth_url = build_authorize_url(
        discovery.authorization_endpoint,
        pkce.code_challenge, state, nonce,
    )

    _PENDING_OAUTH[state] = {
        "pkce": pkce,
        "pool": req.pool,
        "proxy_url": req.proxy_url,
        "created_at": time.time(),
    }

    return _json({
        "url": auth_url,
        "state": state,
        "message": "打开此 URL 完成授权后，调用 POST /xai/oauth/callback",
    })


class XAIOAuthRawExchange(BaseModel):
    """直接交换：code + code_verifier → token。不走 PENDING_OAUTH。"""
    code: str
    code_verifier: str
    pool: str = "super"


@router.post("/xai/oauth/exchange")
async def oauth_raw_exchange(
    req: XAIOAuthRawExchange,
    repo: "AccountRepository" = Depends(_get_repo),
):
    """直接用 code + code_verifier 交换 token 并导入。"""
    from app.dataplane.reverse.protocol.xai_oauth import (
        exchange_code_for_tokens, PKCECodes,
    )
    from app.control.account.xai_oauth_store import import_oauth_token
    from app.control.account.commands import AccountUpsert

    pkce = PKCECodes(
        code_verifier=req.code_verifier,
        code_challenge="",  # 不需要 challenge
    )
    bundle = await exchange_code_for_tokens(req.code, pkce)
    ext = await import_oauth_token(bundle.token_data, pool=req.pool)

    email = bundle.token_data.email or "grok-cli"
    await repo.upsert(AccountUpsert(
        token=bundle.token_data.access_token,
        account_id=f"xai:{email}", pool=req.pool, ext=ext,
    ))

    return _json({
        "status": "success",
        "email": email,
        "access_token": bundle.token_data.access_token[:40] + "...",
        "refresh_token": bundle.token_data.refresh_token[:40] + "...",
        "expires_in": bundle.token_data.expires_in,
    })


class XAIOAuthFromSSORequest(BaseModel):
    """用 SSO token 自动完成 OAuth 流程获取 Grok CLI token。"""
    token: str
    pool: str = "super"
    service: str = "grok-cli"  # "grok-cli" | "build-cli"


@router.post("/xai/oauth/from-sso")
async def oauth_from_sso(
    req: XAIOAuthFromSSORequest,
    repo: "AccountRepository" = Depends(_get_repo),
    refresh_svc: "AccountRefreshService" = Depends(_get_refresh_svc),
):
    """用已有的 Grok SSO token 自动完成 OAuth 流程，获取 api.x.ai 的 Bearer token。

    支持 Super/Heavy 账号，支持标准 Grok CLI 和 Build CLI 两种服务。
    通过 SSO cookie 自动完成 OAuth consent（无需浏览器手动操作）。
    """
    pro_token = req.token.strip()
    if pro_token.startswith("sso="):
        pro_token = pro_token[4:]

    from app.dataplane.reverse.protocol.xai_oauth import (
        generate_pkce_codes, build_authorize_url, discover_oidc,
        exchange_code_for_tokens, REDIRECT_URI,
    )
    from app.control.account.xai_oauth_store import import_oauth_token
    from app.control.account.commands import AccountUpsert

    pkce = generate_pkce_codes()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    discovery = await discover_oidc()

    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        return _json({"status": "error", "message": "curl_cffi not available"}, 500)

    import urllib.parse

    consent_params = {
        "response_type": "code", "client_id": "b1a00492-073a-47ea-816f-4c329264a828",
        "redirect_uri": REDIRECT_URI,
        "scope": "openid profile email offline_access grok-cli:access api:access",
        "code_challenge": pkce.code_challenge, "code_challenge_method": "S256",
        "state": state, "nonce": nonce, "plan": "generic", "referrer": "cli-proxy-api",
    }
    consent_url = f"https://accounts.x.ai/oauth2/consent?{urllib.parse.urlencode(consent_params)}"

    session = curl_requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })
    session.cookies.set("sso", pro_token, domain=".x.ai", path="/")
    session.cookies.set("sso-rw", pro_token, domain=".x.ai", path="/")

    logger.info("xai oauth from-sso: navigating to consent page service={}", req.service)

    r = session.get(consent_url, impersonate="chrome136", allow_redirects=False)
    if r.status_code in (301, 302, 303, 307, 308):
        loc = r.headers.get("Location", "")
        if "callback" in loc and "code=" in loc:
            parsed = urllib.parse.urlparse(loc)
            qs = urllib.parse.parse_qs(parsed.query)
            code = qs.get("code", [""])[0]
            logger.info("xai oauth from-sso: got code directly from redirect")
        elif "sign-in" in loc:
            return _json({
                "status": "need_login",
                "message": "SSO cookie not accepted by accounts.x.ai. This account may be Google-registered and cannot be used for OAuth.",
                "redirect_to": loc[:200],
            }, 400)
        else:
            return _json({"status": "unexpected_redirect", "location": loc[:200]}, 400)
    elif r.status_code == 200:
        if "consent" in r.url:
            logger.info("xai oauth from-sso: on consent page, posting allow")
            r2 = session.post(consent_url, data={"consent": "allow"},
                            impersonate="chrome136", allow_redirects=False)
            if r2.status_code in (301, 302, 303, 307, 308):
                loc = r2.headers.get("Location", "")
                if "callback" in loc and "code=" in loc:
                    parsed = urllib.parse.urlparse(loc)
                    qs = urllib.parse.parse_qs(parsed.query)
                    code = qs.get("code", [""])[0]
                else:
                    return _json({"status": "unexpected_redirect", "location": loc[:200]}, 400)
            else:
                return _json({"status": "consent_failed", "http": r2.status_code}, 400)
        else:
            return _json({
                "status": "need_login",
                "message": "Not on consent page — SSO cookie may not work for this account type",
            }, 400)
    else:
        return _json({"status": "error", "http": r.status_code}, 400)

    if not code:
        return _json({"status": "no_code", "message": "Failed to extract authorization code"}, 400)

    logger.info("xai oauth from-sso: got code, exchanging for tokens service={}", req.service)

    bundle = await exchange_code_for_tokens(code, pkce)
    pool = req.pool
    if bundle.token_data.email and ("@x.ai" in bundle.token_data.email or "@xai" in bundle.token_data.email.lower()):
        pool = "super"

    ext = await import_oauth_token(bundle.token_data, pool=pool)

    access_token = bundle.token_data.access_token
    email = bundle.token_data.email or "xai-oauth"
    account_id = f"xai:{email}"

    await repo.upsert_accounts([AccountUpsert(
        token=access_token,
        account_id=account_id,
        pool=pool,
        ext=ext,
    )])

    if refresh_svc is not None:
        await refresh_svc.refresh_on_demand()

    logger.info("xai oauth from-sso success: email={} pool={} service={}", email, pool, req.service)

    return _json({
        "status": "success",
        "message": f"Grok CLI token obtained: {email} (pool={pool}, service={req.service})",
        "email": email,
        "pool": pool,
        "service": req.service,
        "access_token": access_token[:30] + "...",
        "expires_in": bundle.token_data.expires_in,
        "refresh_token": bundle.token_data.refresh_token[:30] + "..." if bundle.token_data.refresh_token else "",
    })


class XAIOAuthCallbackRequest(BaseModel):
    code: str
    state: str


# ── 回调重定向端点（从 forwarder 302 过来，GET + query params）───────


@public_router.get("/xai/oauth/callback-redirect")
async def oauth_callback_redirect(
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
):
    """接收 OAuth 回调（从 forwarder 端口 56121 302 重定向过来）。"""
    if error:
        return Response(
            content=f"<h1>Login Failed</h1><p>{error}</p>",
            media_type="text/html", status_code=400,
        )
    if not code or not state:
        return Response(
            content="<h1>Login Failed</h1><p>Missing code or state</p>",
            media_type="text/html", status_code=400,
        )

    from app.control.account.xai_oauth_store import import_oauth_token
    from app.control.account.commands import AccountUpsert

    pending = _PENDING_OAUTH.pop(state, None)
    if pending is None:
        return Response(
            content="<h1>Login Failed</h1><p>Unknown or expired state. Please restart the OAuth flow.</p>",
            media_type="text/html", status_code=400,
        )
    if time.time() - pending["created_at"] > 600:
        return Response(
            content="<h1>Login Failed</h1><p>OAuth state expired (>10 min). Please restart.</p>",
            media_type="text/html", status_code=400,
        )

    try:
        from app.dataplane.reverse.protocol.xai_oauth import exchange_code_for_tokens
        bundle = await exchange_code_for_tokens(code, pending["pkce"])
        ext = await import_oauth_token(bundle.token_data, pool=pending["pool"])

        access_token = bundle.token_data.access_token
        email = bundle.token_data.email or "xai-oauth"
        account_id = f"xai:{email}"

        repo = request.app.state.repository
        await repo.upsert(AccountUpsert(
            token=access_token, account_id=account_id, pool=pending["pool"], ext=ext,
        ))

        refresh_svc = request.app.state.refresh_service
        if refresh_svc is not None:
            await refresh_svc.refresh_on_demand()

        logger.info("xai oauth callback-redirect success: email={}", email)
        return Response(
            content=f"<h1>Login Successful!</h1><p>Grok CLI token obtained: {email}</p><p>You can close this window.</p>",
            media_type="text/html",
        )
    except Exception as exc:
        logger.exception("xai oauth callback-redirect exchange failed")
        return Response(
            content=f"<h1>Login Failed</h1><p>{exc}</p>",
            media_type="text/html", status_code=500,
        )


@router.post("/xai/oauth/callback")
async def oauth_callback(
    req: XAIOAuthCallbackRequest,
    repo: "AccountRepository" = Depends(_get_repo),
    refresh_svc: "AccountRefreshService" = Depends(_get_refresh_svc),
):
    """用 OAuth code 交换 token 并导入账号系统。"""
    from app.control.account.xai_oauth_store import import_oauth_token
    from app.control.account.commands import AccountUpsert

    pending = _PENDING_OAUTH.pop(req.state, None)
    if pending is None:
        raise ValidationError("Unknown or expired state", param="state")
    if time.time() - pending["created_at"] > 600:
        raise ValidationError("OAuth state expired (>10 min)", param="state")

    bundle = await exchange_code_for_tokens(req.code, pending["pkce"])
    ext = await import_oauth_token(bundle.token_data, pool=pending["pool"])

    access_token = bundle.token_data.access_token
    email = bundle.token_data.email or "xai-oauth"
    account_id = f"xai:{email}"

    await repo.upsert(AccountUpsert(
        token=access_token,
        account_id=account_id,
        pool=pending["pool"],
        ext=ext,
    ))

    if refresh_svc is not None:
        await refresh_svc.refresh_on_demand()

    logger.info("xai oauth account imported: email={} pool={}", email, pending["pool"])
    return _json({
        "status": "success",
        "message": f"xAI OAuth 账号 {email} 已导入",
        "email": email,
        "pool": pending["pool"],
        "expires_in": bundle.token_data.expires_in,
    })


@router.post("/xai/oauth/import-file")
async def oauth_import_file(
    filepath: str = Body(..., embed=True),
    pool: str = Body("super", embed=True),
    repo: "AccountRepository" = Depends(_get_repo),
    refresh_svc: "AccountRefreshService" = Depends(_get_refresh_svc),
):
    """从 CPA 导出的 xAI token JSON 文件导入。"""
    from pathlib import Path
    from app.control.account.xai_oauth_store import import_oauth_from_file
    from app.control.account.commands import AccountUpsert

    fp = Path(filepath).expanduser().resolve()
    if not fp.exists():
        raise ValidationError(f"File not found: {fp}", param="filepath")

    ext = await import_oauth_from_file(fp, pool=pool)
    if ext is None:
        raise ValidationError("Failed to parse OAuth token file", param="filepath")

    email = ext.get("xai_oauth_email", "unknown")
    access_token = ext.get("xai_oauth_access_token", "")
    account_id = f"xai:{email}"

    await repo.upsert(AccountUpsert(
        token=access_token,
        account_id=account_id,
        pool=pool,
        ext=ext,
    ))

    if refresh_svc is not None:
        await refresh_svc.refresh_on_demand()

    logger.info("xai oauth file imported: email={} pool={}", email, pool)
    return _json({
        "status": "success",
        "message": f"xAI OAuth token 文件已导入: {email}",
        "email": email,
        "pool": pool,
    })
