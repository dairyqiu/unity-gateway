"""Claude Desktop / Cowork launcher: route the GUI through Databricks AI Gateway.

Claude Code is a CLI ucode spawns, so its refresh proxy lives exactly as long as
the child process. Claude Desktop (and its Cowork mode) is a standalone GUI ucode
cannot parent, and its gateway config is a static file. Writing tokens into that
file would leave them to go stale — the Databricks OAuth token expires in ~1h. So
`ug claude-cowork` keeps the same refresh-proxy model that `ug claude` uses:

  1. Establish both auth sessions — the Databricks OAuth session (so the proxy can
     mint swap credentials) and the Anthropic subscription OAuth (via
     `claude setup-token`, whose long-lived token the proxy relays).
  2. Start the loopback refresh proxy. It live-refreshes the Databricks credential
     into the `X-Databricks-AI-Gateway-Token` swap header and stamps the fixed
     Anthropic `Authorization` + the `Databricks-Model-Provider-Service` routing
     header on every request. The Desktop client's own credential/auth-scheme is
     irrelevant: the proxy drops it and owns the upstream auth.
  3. Write a Claude Desktop gateway config pointing at that proxy (holding NO real
     tokens, only a loopback URL) and register it in ``configLibrary/_meta.json`` as
     an entry and the applied config — a file alone is invisible to Desktop. The
     prior applied config is restored on exit.
  4. Stay in the foreground to keep the proxy alive while Desktop is used.

EXPERIMENTAL. The Desktop gateway-config schema (the Claude-3p ``configLibrary``
entry + ``_meta.json`` registry) is not a public contract; the keys written here
mirror what Desktop authors and may drift across releases. ``modelDiscoveryEnabled``
is false and models are listed explicitly, because the relayed ``GET /v1/models``
probe is not served until the companion gateway route change ships.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from ucode import gateway_proxy
from ucode.config_io import backup_existing_file, read_json_safe, write_json_file
from ucode.constants import LOOPBACK_HOST, MODEL_PROVIDER_SERVICE_HEADER
from ucode.databricks import get_databricks_token
from ucode.managed_files import OS, current_os
from ucode.telemetry import agent_version, ucode_version
from ucode.ui import print_note, print_success, print_warning

# The Anthropic subscription OAuth, when generated headlessly. `claude setup-token`
# mints a long-lived (~1yr) token and prints it; we also accept it pre-supplied via
# this env var (the same one Claude Code reads) to skip the browser flow in CI.
CLAUDE_CODE_OAUTH_TOKEN_ENV_VAR = "CLAUDE_CODE_OAUTH_TOKEN"
# Anthropic OAuth tokens are `sk-ant-oat…`; match defensively so we can pull the
# token out of `claude setup-token` output regardless of surrounding prose.
_OAUTH_TOKEN_RE = re.compile(r"sk-ant-[A-Za-z0-9._\-]+")
# Client auth headers Desktop may send with its (now-unused) configured key. The
# proxy owns the upstream auth, so drop them: a stray `x-api-key` carrying the
# OAuth would be rejected by Anthropic alongside the injected `Authorization`.
_STRIP_CLIENT_AUTH_HEADERS = frozenset({"x-api-key", "authorization"})
# Placeholder written as the Desktop config's key. The proxy overwrites
# `Authorization` and drops `x-api-key`, so no real credential lives on disk.
_CONFIG_KEY_PLACEHOLDER = "ug-refresh-proxy-injects-credentials"
# Desktop's registry file inside configLibrary/: `{appliedId, entries: [{id, name}]}`.
# A config file is invisible to Desktop until it is listed in `entries`, and inactive
# until it is the `appliedId`. ug registers its entry and applies it, then restores the
# prior appliedId on exit so Desktop doesn't stay pinned to a dead-proxy config.
_META_FILENAME = "_meta.json"
_ENTRY_NAME = "Unity Gateway"
# Desktop writes its own configLibrary files mode 0600; match that.
_CONFIG_FILE_MODE = 0o600


def _app_support_dir() -> Path:
    """Claude Desktop's per-OS application-support root for the gateway
    (``Claude-3p``) build. Raises on unsupported platforms."""
    system = current_os()
    if system is OS.MACOS:
        return Path.home() / "Library" / "Application Support" / "Claude-3p"
    if system is OS.WINDOWS:
        base = os.environ.get("APPDATA")
        if not base:
            raise RuntimeError("APPDATA is not set; cannot locate Claude Desktop config.")
        return Path(base) / "Claude-3p"
    raise RuntimeError(
        f"`ug claude-cowork` currently supports macOS and Windows only (detected {system.value})."
    )


def _config_library_dir() -> Path:
    return _app_support_dir() / "configLibrary"


def config_entry_id(workspace: str) -> str:
    """Deterministic entry id for ``workspace`` so re-runs reuse one stable entry
    rather than accumulating duplicates."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ug-claude-cowork::{workspace}"))


def config_path(workspace: str) -> Path:
    """Path to the ug-owned Desktop gateway-config entry for ``workspace``."""
    return _config_library_dir() / f"{config_entry_id(workspace)}.json"


def register_active_config(entry_id: str, name: str = _ENTRY_NAME) -> str | None:
    """Register ``entry_id`` in Desktop's ``_meta.json`` and mark it applied.

    Preserves existing entries (e.g. the user's ``Default``) and returns the prior
    ``appliedId`` so the caller can restore it on exit.
    """
    meta_path = _config_library_dir() / _META_FILENAME
    meta = read_json_safe(meta_path) if meta_path.exists() else {}
    prior_applied = meta.get("appliedId")
    entries = meta.get("entries")
    entries = (
        [e for e in entries if isinstance(e, dict) and e.get("id") != entry_id]
        if (isinstance(entries, list))
        else []
    )
    entries.append({"id": entry_id, "name": name})
    meta["entries"] = entries
    meta["appliedId"] = entry_id
    write_json_file(meta_path, meta)
    os.chmod(meta_path, _CONFIG_FILE_MODE)
    return prior_applied if isinstance(prior_applied, str) else None


def restore_active_config(prior_applied: str | None) -> None:
    """Point ``appliedId`` back to ``prior_applied`` (or clear it), so Desktop's
    next restart doesn't boot into the now-dead loopback-proxy config."""
    meta_path = _config_library_dir() / _META_FILENAME
    if not meta_path.exists():
        return
    meta = read_json_safe(meta_path)
    if prior_applied is not None:
        meta["appliedId"] = prior_applied
    else:
        meta.pop("appliedId", None)
    write_json_file(meta_path, meta)
    os.chmod(meta_path, _CONFIG_FILE_MODE)


def render_config(base_url: str, models: list[str]) -> dict:
    """The Desktop gateway config pointing at the loopback proxy.

    Mirrors the schema Desktop authors for a gateway config. Holds no real
    credentials and no ``inferenceCustomHeaders`` — the proxy injects the
    Authorization, swap, and Model-Provider-Service headers per request.
    ``modelDiscoveryEnabled`` is false and ``models`` are listed explicitly,
    because the relayed ``/v1/models`` probe isn't served.
    """
    config: dict = {
        "inferenceProvider": "gateway",
        "inferenceGatewayBaseUrl": base_url,
        "inferenceCredentialKind": "static",
        "inferenceGatewayApiKey": _CONFIG_KEY_PLACEHOLDER,
        "modelDiscoveryEnabled": False,
    }
    if models:
        config["inferenceModels"] = [{"name": model} for model in models]
    return config


def _resolve_anthropic_oauth() -> str:
    """Return the Anthropic subscription OAuth token, launching the auth session
    when needed.

    Prefers a pre-supplied ``CLAUDE_CODE_OAUTH_TOKEN`` (headless / CI); otherwise
    runs ``claude setup-token`` (opens a browser to the Claude subscription) and
    parses its long-lived token from the output.
    """
    preset = os.environ.get(CLAUDE_CODE_OAUTH_TOKEN_ENV_VAR, "").strip()
    if preset:
        return preset
    print_note("Launching Claude subscription sign-in (`claude setup-token`)...")
    try:
        result = subprocess.run(
            ["claude", "setup-token"],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "`claude` was not found on PATH. Install Claude Code "
            "(npm i -g @anthropic-ai/claude-code), or set "
            f"{CLAUDE_CODE_OAUTH_TOKEN_ENV_VAR} to a token from `claude setup-token`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "`claude setup-token` failed; cannot obtain a subscription token."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("`claude setup-token` timed out.") from exc
    match = _OAUTH_TOKEN_RE.search(result.stdout or "")
    if not match:
        raise RuntimeError(
            "Could not read a subscription token from `claude setup-token` output. "
            f"Run it manually and pass the token via {CLAUDE_CODE_OAUTH_TOKEN_ENV_VAR}."
        )
    print_success("Claude subscription authenticated")
    return match.group(0)


def _ensure_databricks_session(workspace: str, profile: str | None) -> None:
    """Make sure a Databricks OAuth session exists so the proxy can mint swap
    credentials, launching `databricks auth login` if the token fetch fails."""
    if shutil.which("databricks") is None:
        raise RuntimeError(
            "The `databricks` CLI was not found on PATH. Install it "
            "(https://docs.databricks.com/dev-tools/cli/install.html), then re-run."
        )
    try:
        get_databricks_token(workspace, profile)
        return
    except (RuntimeError, FileNotFoundError):
        # RuntimeError: no/expired session. FileNotFoundError: the CLI vanished
        # between the which() check and the call. Either way, try an explicit login.
        pass
    print_note(f"Launching Databricks sign-in for {workspace}...")
    cmd = ["databricks", "auth", "login", "--host", workspace]
    if profile:
        cmd += ["--profile", profile]
    try:
        subprocess.run(cmd, check=True, timeout=300)
    except FileNotFoundError as exc:
        raise RuntimeError("`databricks` CLI was not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"`databricks auth login` failed for {workspace}.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("`databricks auth login` timed out.") from exc
    # Confirm the session now mints a token, so a stale/half-finished login
    # surfaces here rather than as a silent proxy 401 later.
    get_databricks_token(workspace, profile)


def launch(
    workspace: str,
    profile: str | None,
    provider: str,
    models: list[str] | None = None,
) -> None:
    """Configure Claude Desktop for ``provider`` and run the refresh proxy.

    Blocks until interrupted: the proxy must outlive the call, since Desktop is a
    separate long-lived GUI process (not a child we exec).
    """
    anthropic_oauth = _resolve_anthropic_oauth()
    _ensure_databricks_session(workspace, profile)

    extra_headers = {
        gateway_proxy.AUTHORIZATION_HEADER: f"Bearer {anthropic_oauth}",
        MODEL_PROVIDER_SERVICE_HEADER: provider,
        "User-Agent": f"ucode/{ucode_version()} claude-cowork/{agent_version('claude')}",
    }
    server, cache, client = gateway_proxy.start_proxy(
        workspace,
        profile,
        port=0,  # let the OS assign a free loopback port; the config uses the bound one
        token_header=gateway_proxy.AI_GATEWAY_TOKEN_HEADER,
        force_refresh_near_expiry=False,
        extra_headers=extra_headers,
        strip_client_headers=_STRIP_CLIENT_AUTH_HEADERS,
    )
    bound_port = server.server_address[1]
    base_url = f"http://{LOOPBACK_HOST}:{bound_port}"

    path = config_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_existing_file(path, path.with_suffix(".json.ug-backup"))
    write_json_file(path, render_config(base_url, models or []))
    os.chmod(path, _CONFIG_FILE_MODE)
    # Register + activate the entry (a file alone is invisible to Desktop); remember
    # the prior applied config so we can hand it back when the proxy stops.
    prior_applied = register_active_config(config_entry_id(workspace))

    print_success(f"Gateway refresh proxy running at {base_url}")
    print_note(
        f"Registered + applied the '{_ENTRY_NAME}' gateway config. Fully quit Claude Desktop "
        "(Cmd-Q) and reopen it to pick it up — it reads the config only at startup. Keep this "
        f"command running while you use Desktop; closing it stops the proxy.\nConfig: {path}"
    )
    try:
        # Serve on the main thread until Ctrl-C. serve_forever() blocks in select()
        # and unwinds only via shutdown() or the KeyboardInterrupt SIGINT raises — it
        # does NOT wake on unrelated signals (SIGCHLD/SIGPIPE/SIGWINCH), unlike
        # signal.pause(), which would otherwise drop the proxy on the first inference.
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cache.stop()
        server.shutdown()
        client.close()
        # Hand the applied config back so Desktop's next restart doesn't boot into
        # this now-dead loopback proxy.
        restore_active_config(prior_applied)
        print_warning(
            "Gateway refresh proxy stopped; Claude Desktop can no longer reach the gateway. "
            "Restored the previously-applied config."
        )
