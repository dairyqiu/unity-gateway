"""Tests for the Claude Desktop / Cowork launcher config + auth resolution."""

from __future__ import annotations

import subprocess

import pytest

from ucode.agents import claude_desktop
from ucode.managed_files import OS


class TestRenderConfig:
    def test_points_at_proxy_and_holds_no_real_token(self):
        config = claude_desktop.render_config("http://127.0.0.1:51234", [])
        assert config["inferenceProvider"] == "gateway"
        assert config["inferenceGatewayBaseUrl"] == "http://127.0.0.1:51234"
        assert config["coworkTabEnabled"] is True
        # The proxy injects credentials per request; the file must not carry a real one.
        assert config["inferenceGatewayApiKey"] == claude_desktop._CONFIG_KEY_PLACEHOLDER
        assert "sk-ant" not in config["inferenceGatewayApiKey"]

    def test_models_populate_inference_models_to_skip_discovery(self):
        config = claude_desktop.render_config(
            "http://127.0.0.1:1", ["claude-opus-4-8", "claude-sonnet-4-5"]
        )
        assert config["inferenceModels"] == [
            {"name": "claude-opus-4-8"},
            {"name": "claude-sonnet-4-5"},
        ]

    def test_no_models_omits_inference_models(self):
        assert "inferenceModels" not in claude_desktop.render_config("http://127.0.0.1:1", [])


class TestConfigPath:
    def test_macos_path_is_stable_per_workspace(self, monkeypatch, tmp_path):
        monkeypatch.setattr(claude_desktop, "current_os", lambda: OS.MACOS)
        monkeypatch.setenv("HOME", str(tmp_path))
        p1 = claude_desktop.config_path("https://ws.example.com")
        p2 = claude_desktop.config_path("https://ws.example.com")
        assert p1 == p2  # deterministic
        assert p1 != claude_desktop.config_path("https://other.example.com")
        assert p1.parent.name == "configLibrary"
        assert p1.suffix == ".json"
        assert "Claude-3p" in str(p1)

    def test_unsupported_os_raises(self, monkeypatch):
        monkeypatch.setattr(claude_desktop, "current_os", lambda: OS.LINUX)
        with pytest.raises(RuntimeError, match="macOS and Windows"):
            claude_desktop.config_path("https://ws.example.com")


class TestResolveAnthropicOauth:
    def test_prefers_preset_env_and_skips_browser(self, monkeypatch):
        monkeypatch.setenv(claude_desktop.CLAUDE_CODE_OAUTH_TOKEN_ENV_VAR, "sk-ant-oat-preset")

        def _fail(*_a, **_k):  # pragma: no cover - must not run
            raise AssertionError("setup-token should not be invoked when the env var is set")

        monkeypatch.setattr(subprocess, "run", _fail)
        assert claude_desktop._resolve_anthropic_oauth() == "sk-ant-oat-preset"

    def test_parses_token_from_setup_token_output(self, monkeypatch):
        monkeypatch.delenv(claude_desktop.CLAUDE_CODE_OAUTH_TOKEN_ENV_VAR, raising=False)

        def _fake_run(*_a, **_k):
            return subprocess.CompletedProcess(
                args=["claude", "setup-token"],
                returncode=0,
                stdout="Your token:\n  sk-ant-oat01-abcDEF_-123  \nKeep it secret.\n",
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", _fake_run)
        assert claude_desktop._resolve_anthropic_oauth() == "sk-ant-oat01-abcDEF_-123"

    def test_raises_when_no_token_in_output(self, monkeypatch):
        monkeypatch.delenv(claude_desktop.CLAUDE_CODE_OAUTH_TOKEN_ENV_VAR, raising=False)

        def _fake_run(*_a, **_k):
            return subprocess.CompletedProcess(
                args=["claude", "setup-token"], returncode=0, stdout="no token here", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", _fake_run)
        with pytest.raises(RuntimeError, match="Could not read a subscription token"):
            claude_desktop._resolve_anthropic_oauth()

    def test_raises_when_claude_missing(self, monkeypatch):
        monkeypatch.delenv(claude_desktop.CLAUDE_CODE_OAUTH_TOKEN_ENV_VAR, raising=False)

        def _fake_run(*_a, **_k):
            raise FileNotFoundError("claude")

        monkeypatch.setattr(subprocess, "run", _fake_run)
        with pytest.raises(RuntimeError, match="was not found on PATH"):
            claude_desktop._resolve_anthropic_oauth()


class TestEnsureDatabricksSession:
    def test_missing_databricks_cli_raises_actionable_error(self, monkeypatch):
        # A missing `databricks` binary must surface a clear install hint, not a
        # raw FileNotFoundError traceback out of get_databricks_token.
        monkeypatch.setattr(claude_desktop.shutil, "which", lambda _name: None)
        with pytest.raises(RuntimeError, match="`databricks` CLI was not found"):
            claude_desktop._ensure_databricks_session("https://ws.example.com", None)
