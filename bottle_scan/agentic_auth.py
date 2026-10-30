"""Sign in with ChatGPT for the experiment supervisor; no robot imports.

Public interfaces accept the flat ``provider`` settings dictionary. Credentials
live outside the repository and never enter an experiment artifact. Sources:
https://developers.openai.com/siwc/token-sharing-open-source/sign-in
https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import http.server
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import time
import urllib.parse
import uuid
import webbrowser

AUTH_BASE = "https://auth.openai.com"
API_BASE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
REPO_ROOT = Path(__file__).resolve().parents[1]


class AuthError(RuntimeError):
    """Safe to display: messages contain no response bodies or credentials."""


def new_session(settings):
    try:
        import requests
    except ImportError:
        raise AuthError("缺少 requests；安装 requirements.agentic.txt 后重试。") from None
    client = requests.Session()
    proxy = settings.get("proxy")
    client.trust_env = proxy is None
    if proxy:
        client.proxies.update({"http": str(proxy), "https": str(proxy)})
    return client


def _request(client, method, url, **kwargs):
    try:
        return getattr(client, method)(url, allow_redirects=False,
                                      timeout=kwargs.pop("timeout", (10, 30)), **kwargs)
    except Exception:
        # requests' exception text may contain proxy credentials or request URLs.
        raise AuthError("ChatGPT 网络请求失败或超时；本轮暂停，请检查代理和连接。") from None


def _json_response(response, operation):
    try:
        if response.status_code != 200:
            raise AuthError(f"{operation}失败（HTTP {int(response.status_code)}）；请检查授权或稍后重试。")
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except AuthError:
        raise
    except Exception:
        raise AuthError(f"{operation}返回了无效数据。") from None
    finally:
        response.close()


def _external_dir(path):
    path = Path(path).expanduser().absolute()
    if path.is_symlink():
        raise AuthError("认证目录不能是符号链接。")
    path = path.resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise AuthError("认证目录必须位于项目目录之外，不能随实验或 Git 保存。")
    return path


def auth_directory(settings):
    return _external_dir(settings.get("auth_dir") or
                         os.environ.get("BOTTLE_AGENTIC_AUTH_DIR") or
                         Path.home() / ".config" / "bottle-agentic")


def _slot(settings):
    value = settings.get("account", "default")
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise AuthError("account 只能包含字母、数字、下划线和连字符，最多 64 字符。")
    return value


def _record_path(settings):
    return auth_directory(settings) / ("account_" + _slot(settings) + ".json")


def _ensure_private(directory):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)


def _atomic_private(path, value):
    _ensure_private(path.parent)
    if path.is_symlink():
        raise AuthError("拒绝写入符号链接认证文件。")
    fd, temporary = tempfile.mkstemp(prefix=".credentials-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        # Persist the rename as well as the content before a rotating token is used.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _read_private(path):
    if not path.exists() and not path.is_symlink():
        return {}
    if path.is_symlink():
        raise AuthError("拒绝读取符号链接认证文件。")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
        raise AuthError("认证文件必须是仅当前用户可读写的普通文件（chmod 600）。")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise AuthError("认证文件不属于当前用户。")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except Exception:
        raise AuthError("认证文件损坏；请重新执行 agentic login。") from None


@contextmanager
def _account_lock(settings):
    import fcntl
    directory = auth_directory(settings)
    _ensure_private(directory)
    path = directory / ("account_" + _slot(settings) + ".lock")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _legacy_registration(settings):
    if _slot(settings) != "default":
        return {}
    configured = settings.get("identity_dir") or os.environ.get("AGENTIC_IDENTITY_DIR")
    choices = [Path(configured).expanduser()] if configured else [
        REPO_ROOT.parent / ".siwc-robot-probe", Path.home() / ".siwc-robot-probe"]
    for directory in choices:
        directory = _external_dir(directory)
        if not (directory / "client_id").is_file():
            continue
        try:
            client_id = (directory / "client_id").read_text().strip()
            host_id = (directory / "host_id").read_text().strip()
        except OSError:
            raise AuthError("原 Robot Supervisor Probe 注册缺少 client_id 或 host_id。") from None
        if client_id == "dynamic_agent_client" or not client_id or not host_id:
            raise AuthError("原 Robot Supervisor Probe 注册无效。")
        return {"client_id": client_id, "ext_agent_host_id": host_id}
    if configured:
        raise AuthError("identity_dir 中未找到原 Robot Supervisor Probe 注册。")
    return {}


def _registration(settings):
    record = _read_private(_record_path(settings))
    if record:
        return record
    record = _legacy_registration(settings)
    host_path = auth_directory(settings) / "host.json"
    host = _read_private(host_path)
    if not record.get("ext_agent_host_id"):
        record["ext_agent_host_id"] = host.get("ext_agent_host_id") or "urn:uuid:" + str(uuid.uuid4())
    if not host:
        _atomic_private(host_path, {"ext_agent_host_id": record["ext_agent_host_id"]})
    return record


def _official_auth_url(url):
    parts = urllib.parse.urlsplit(str(url))
    if parts.scheme != "https" or parts.netloc != "auth.openai.com" or parts.fragment:
        raise AuthError("OIDC 元数据包含非官方认证地址，已停止。")
    return str(url)


def _metadata(client):
    meta = _json_response(_request(client, "get", AUTH_BASE + "/.well-known/openid-configuration"), "OIDC discovery")
    if meta.get("issuer") != AUTH_BASE:
        raise AuthError("OIDC issuer 不匹配。")
    _official_auth_url(meta.get("jwks_uri"))
    return meta


def _validate_id_token(token, client_id, client, *, nonce=None, subject=None):
    try:
        import jwt
    except ImportError:
        raise AuthError("缺少 PyJWT[crypto]；安装 requirements.agentic.txt 后重试。") from None
    metadata = _metadata(client)
    jwks = _json_response(_request(client, "get", metadata["jwks_uri"]), "OIDC JWKS")
    try:
        header = jwt.get_unverified_header(token)
        if header.get("alg") != "RS256" or not header.get("kid"):
            raise ValueError()
        keys = [key for key in jwks["keys"] if key.get("kid") == header["kid"]
                and key.get("kty") == "RSA" and key.get("use", "sig") == "sig"
                and key.get("alg", "RS256") == "RS256"]
        if len(keys) != 1:
            raise ValueError()
        identity = jwt.decode(token, jwt.PyJWK.from_dict(keys[0]).key,
                              algorithms=["RS256"], audience=client_id, issuer=AUTH_BASE,
                              options={"require": ["exp", "iat", "sub", "aud", "iss"]})
        if not isinstance(identity.get("sub"), str) or not identity["sub"]:
            raise ValueError()
        if nonce is not None and not secrets.compare_digest(str(identity.get("nonce", "")), nonce):
            raise ValueError()
        if subject is not None and identity["sub"] != subject:
            raise ValueError()
        # OIDC multiple audiences require the authorized party to be this client.
        if isinstance(identity["aud"], list) and len(identity["aud"]) > 1 and identity.get("azp") != client_id:
            raise ValueError()
        if identity.get("azp", client_id) != client_id:
            raise ValueError()
        return identity
    except Exception:
        raise AuthError("OIDC 身份验证失败（签名、算法、账号、有效期或 nonce 不匹配）。") from None


def _timestamp(value):
    try:
        if isinstance(value, str) and not value.replace(".", "", 1).isdigit():
            result = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        else:
            result = float(value)
        return result if math.isfinite(result) else 0.0
    except (ValueError, TypeError, OverflowError):
        return 0.0


def _token_record(tokens, registration, client, *, nonce=None, refresh=False):
    try:
        expires = float(tokens["expires_in"])
        if not math.isfinite(expires) or expires <= 0:
            raise ValueError()
        if str(tokens.get("token_type", "")).lower() != "bearer":
            raise ValueError()
        for field in ("access_token", "refresh_token"):
            if not isinstance(tokens.get(field), str) or not tokens[field]:
                raise ValueError()
        scopes = tokens.get("scope")
        if scopes is None and refresh:
            scopes = " ".join(registration.get("scopes", []))
        scopes = sorted(set(scopes.split()))
        if "chatgpt.tokens.use.direct" not in scopes:
            raise AuthError("账号未授予 ChatGPT 套餐调用权限；请重新登录并授权。")
    except AuthError:
        raise
    except Exception:
        raise AuthError("令牌响应缺少有效凭据、轮换刷新令牌或到期信息。") from None
    identity = None
    if tokens.get("id_token"):
        identity = _validate_id_token(tokens["id_token"], registration["client_id"], client,
                                      nonce=nonce, subject=registration.get("subject"))
    elif not refresh:
        raise AuthError("登录未返回 OIDC ID token。")
    result = dict(registration)
    if identity:
        result.update(subject=identity["sub"], email=identity.get("email", ""), issuer=identity["iss"])
        result["id_token"] = tokens["id_token"]
    result.update(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"],
                  token_type="Bearer", scopes=scopes, expires_at=time.time() + expires,
                  saved_at=datetime.now(timezone.utc).isoformat(),
                  earliest_refresh_at=_timestamp(tokens.get("earliest_refresh_at", 0)))
    return result


def _parse_callback(values, expected_state, saved_client_id):
    for field in ("state", "code", "client_id", "error"):
        if field in values and len(values[field]) != 1:
            raise AuthError("OAuth 回调有重复参数。")
    state = values.get("state", [""])[0]
    if not secrets.compare_digest(str(state), expected_state):
        raise AuthError("OAuth state 不匹配；请重新登录。")
    if "error" in values:
        raise AuthError("浏览器未授权此次登录；请重新登录。")
    client_id = values.get("client_id", [saved_client_id])[0]
    code = values.get("code", [None])[0]
    if not code or not client_id or client_id == "dynamic_agent_client":
        raise AuthError("OAuth 回调缺少授权码或已注册 client_id。")
    if saved_client_id and saved_client_id != client_id:
        raise AuthError("OAuth 回调 client_id 与已选账号不匹配。")
    return code, client_id


def login(settings, say=print):
    """Interactive local-loopback login. Prints no ID/access/refresh token.

    On a headless host forward the configured callback port with SSH before
    opening the displayed URL on the laptop. Each ``account`` is a separate slot.
    """
    port = int(settings.get("callback_port", 8766))
    timeout = float(settings.get("login_timeout_s", 300))
    if not 0 <= port <= 65535 or not math.isfinite(timeout) or not 1 <= timeout <= 1800:
        raise AuthError("回调端口或登录等待时间无效。")
    # Serialize an interactive re-login against refresh/logout for this account.
    with _account_lock(settings):
        registration = _registration(settings)
        state, nonce, verifier = (secrets.token_urlsafe(n) for n in (32, 32, 64))
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")

        class Callback(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path != "/auth/callback":
                    self.send_error(404)
                    return
                values = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
                try:
                    result = _parse_callback(values, state, registration.get("client_id"))
                except AuthError as error:
                    self.server.result = error
                    self.send_response(400)
                else:
                    self.server.result = result
                    self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(b"Return to the experiment terminal to see the sign-in result.\n")

            def log_message(self, *_args):
                pass

        try:
            server = http.server.HTTPServer(("127.0.0.1", port), Callback)
        except OSError:
            raise AuthError("无法绑定 127.0.0.1 登录回调端口；请检查占用或修改 callback_port。") from None
        client = None
        try:
            client = new_session(settings)
            redirect = f"http://127.0.0.1:{server.server_port}/auth/callback"
            params = {"client_id": registration.get("client_id") or "dynamic_agent_client",
                      "ext_agent_host_id": registration["ext_agent_host_id"],
                      "response_type": "code", "redirect_uri": redirect,
                      "scope": SCOPES, "resource": API_BASE, "state": state, "nonce": nonce,
                      "code_challenge_method": "S256", "code_challenge": challenge}
            if not registration.get("client_id"):
                params["agent_name_hint"] = "agentic explore data collect"
            # Deliberately omit id_token_hint: this URL is printed on headless
            # terminals. The user can select the correct account in the browser.
            url = AUTH_BASE + "/api/accounts/authorize?" + urllib.parse.urlencode(params)
            say("Continue with ChatGPT：请在浏览器登录。")
            say(f"远程运行时先在笔记本转发端口：ssh -N -L {server.server_port}:127.0.0.1:{server.server_port} 6f5090")
            say(url)
            if settings.get("open_browser", False):
                webbrowser.open(url)
            deadline = time.monotonic() + timeout
            while not hasattr(server, "result") and time.monotonic() < deadline:
                server.timeout = min(1.0, max(.01, deadline - time.monotonic()))
                server.handle_request()
            result = getattr(server, "result", None)
            if result is None:
                raise AuthError("等待浏览器登录超时；没有修改原有账号凭据。")
            if isinstance(result, AuthError):
                raise result
            code, issued_client = result
            tokens = _json_response(_request(client, "post", AUTH_BASE + "/api/accounts/oauth/token",
                data={"grant_type": "authorization_code", "client_id": issued_client,
                      "code": code, "code_verifier": verifier, "redirect_uri": redirect,
                      "resource": API_BASE}), "OAuth 换令牌")
            candidate = dict(registration, client_id=issued_client)
            validated = _token_record(tokens, candidate, client, nonce=nonce)
            _atomic_private(_record_path(settings), validated)
            say("ChatGPT 登录和 OIDC 验证通过；凭据已保存到项目之外。")
        finally:
            server.server_close()
            if client is not None:
                client.close()
    return auth_status(settings)


def auth_status(settings):
    """Local-only status; never refresh, inspect another app's token, or connect."""
    record = _read_private(_record_path(settings))
    return {"backend": "chatgpt", "account": _slot(settings),
            "registered": bool(record.get("client_id")),
            "signed_in": bool(record.get("access_token") and record.get("refresh_token")),
            "email": record.get("email", ""),
            "expired": _timestamp(record.get("expires_at", 0)) <= time.time(),
            "expires_at": record.get("expires_at"),
            "plan_permission": "chatgpt.tokens.use.direct" in record.get("scopes", []),
            "credentials_directory": str(auth_directory(settings))}


def access_token(settings, client=None, *, force_refresh=False):
    """Return bearer credential to the HTTP transport only; caller must not log."""
    owned = client is None
    client = client or new_session(settings)
    try:
        with _account_lock(settings):
            record = _read_private(_record_path(settings))
            if not record.get("refresh_token") or not record.get("access_token"):
                raise AuthError("尚未登录 ChatGPT；请先执行 agentic login。")
            if "chatgpt.tokens.use.direct" not in record.get("scopes", []):
                raise AuthError("当前登录缺少 ChatGPT 套餐调用权限；请重新登录。")
            now = time.time()
            expires = _timestamp(record.get("expires_at"))
            if not force_refresh and expires > now + 90:
                return record["access_token"]
            earliest = _timestamp(record.get("earliest_refresh_at"))
            if now < earliest:
                if not force_refresh and expires > now + 5:
                    return record["access_token"]
                raise AuthError("令牌尚未到可刷新时间；本轮暂停，请稍后重试。")
            tokens = _json_response(_request(client, "post", AUTH_BASE + "/api/accounts/oauth/token",
                data={"grant_type": "refresh_token", "client_id": record["client_id"],
                      "refresh_token": record["refresh_token"], "resource": API_BASE}), "OAuth 刷新")
            updated = _token_record(tokens, record, client, refresh=True)
            _atomic_private(_record_path(settings), updated)
            return updated["access_token"]
    finally:
        if owned:
            client.close()


def logout(settings):
    """Revoke this renewable session and clear tokens, retaining registration."""
    with _account_lock(settings):
        record = _read_private(_record_path(settings))
        if not record:
            return {"signed_out": True, "remote_revocation": "no_local_session"}
        revoked = not bool(record.get("refresh_token"))
        client = new_session(settings)
        try:
            if record.get("refresh_token"):
                for attempt in range(2):
                    try:
                        metadata = _metadata(client)
                        endpoint = _official_auth_url(metadata["revocation_endpoint"])
                        response = _request(client, "post", endpoint, data={
                            "token": record["refresh_token"], "token_type_hint": "refresh_token",
                            "client_id": record["client_id"]})
                        status = response.status_code
                        response.close()
                        revoked = status == 200
                        if revoked or status < 500:
                            break
                    except (AuthError, KeyError):
                        revoked = False
                    if attempt == 0:
                        time.sleep(.5)
        finally:
            client.close()
            keep = {key: value for key, value in record.items()
                    if key in ("client_id", "ext_agent_host_id", "subject", "issuer", "email")}
            _atomic_private(_record_path(settings), keep)
        result = {"signed_out": True, "remote_revocation": "confirmed" if revoked else "unconfirmed"}
        if not revoked:
            result["message"] = "本地凭据已清除；远端注销未确认，可在 ChatGPT 设置中断开该应用。"
        return result


def list_accounts(settings):
    """Enumerate locally saved slots without exposing any token or subject."""
    directory = auth_directory(settings)
    return [auth_status(dict(settings, account=path.stem[len("account_"):]))
            for path in sorted(directory.glob("account_*.json"))]
