"""Claude CUJs for Tests-table cases 13, 15, 17, 19, 21, and 23."""

import pytest
from utils.terminal import AgentTerminal

pytestmark = pytest.mark.claude

CLAUDE_NATIVE_MODEL_FAMILIES = ("Opus", "Sonnet", "Haiku")


def _configure_hosted(session, workspace):
    session.run(
        "configure",
        "--agents",
        "claude",
        "--workspace",
        workspace,
        "--skip-upgrade",
        "--disable-databricks-ai-tools",
        timeout=240,
    )


def _assert_scoped_models_in_picker(session, screen, expected_ids):
    models = session.claude_gateway_models()
    assert [model.get("id") for model in models] == expected_ids, models
    display_names = [model.get("display_name") for model in models]
    assert all(isinstance(name, str) and name for name in display_names), models
    for display_name in display_names:
        assert display_name in screen, screen


def _assert_native_models_in_picker(screen):
    for family in CLAUDE_NATIVE_MODEL_FAMILIES:
        assert family in screen, screen


def _assert_no_claude_owned_gateway_cache_after_launch(session):
    """Check Claude Code's own cache only after its picker process has exited."""
    assert not (session.home / ".claude/cache/gateway-models.json").exists()


@pytest.mark.live
@pytest.mark.tui
def test_case_13_configured_claude_reuses_saved_model_location(
    live_session, workspace, parent_schema, claude_parent_model
):
    """Scenario: configure Claude with --model-location, then launch without options.

    Expected: the saved parent supplies Claude's discovered model catalog.
    """
    session = live_session
    session.run(
        "configure",
        "--agents",
        "claude",
        "--workspace",
        workspace,
        "--model-location",
        parent_schema,
        "--skip-upgrade",
        "--disable-databricks-ai-tools",
        timeout=240,
    )

    command = [str(session.binary), "claude"]
    with AgentTerminal(session, "claude", command, "case-13-saved-location") as tui:
        tui.boot()
        screen = tui.open_model_picker()
        tui.exit_normally()

    _assert_scoped_models_in_picker(session, screen, [claude_parent_model])


@pytest.mark.live
@pytest.mark.tui
def test_case_15_fresh_claude_uses_system_models_when_discovery_disabled(live_session, workspace):
    """Scenario: launch fresh Claude with UG_ENABLE_MODEL_DISCOVERY=0.

    Expected: ug uses its discovered system.ai family models without a gateway catalog.
    """
    session = live_session
    session.env["UG_ENABLE_MODEL_DISCOVERY"] = "0"
    command = [str(session.binary), "claude", "--workspace", workspace]
    with AgentTerminal(session, "claude", command, "case-15-system-models") as tui:
        tui.boot()
        screen = tui.open_model_picker()
        tui.exit_normally()

    models = session.workspace_state()["claude_models"]
    assert models
    assert all(model.startswith("system.ai.") for model in models.values())
    for model_id in models.values():
        assert model_id in screen, screen
    _assert_no_claude_owned_gateway_cache_after_launch(session)


@pytest.mark.parametrize("configured", [True, False], ids=["configured", "fresh"])
@pytest.mark.live
@pytest.mark.tui
def test_case_17_configured_claude_provider_discovers_models_by_default(
    live_session, workspace, claude_provider, claude_provider_model, configured
):
    """Scenario: launch configured and fresh Claude with --provider and no opt-in flag.

    Expected: provider discovery is automatic and supplies its exact picker catalog.
    """
    session = live_session
    if configured:
        _configure_hosted(session, workspace)

    args = [] if configured else ["--workspace", workspace]
    command = [str(session.binary), "claude", *args, "--provider", claude_provider]
    with AgentTerminal(session, "claude", command, "case-17-provider-default") as tui:
        tui.boot()
        screen = tui.open_model_picker()
        tui.exit_normally()

    _assert_scoped_models_in_picker(session, screen, [claude_provider_model])


@pytest.mark.parametrize("configured", [True, False], ids=["configured", "fresh"])
@pytest.mark.live
@pytest.mark.tui
def test_case_19_configured_claude_model_location_overrides_saved_setup(
    live_session, workspace, parent_schema, claude_parent_model, configured
):
    """Scenario: launch configured and fresh Claude with --model-location.

    Expected: the explicit parent supplies its exact picker catalog in both variants.
    """
    session = live_session
    if configured:
        _configure_hosted(session, workspace)

    args = [] if configured else ["--workspace", workspace]
    command = [str(session.binary), "claude", *args, "--model-location", parent_schema]
    with AgentTerminal(session, "claude", command, "case-19-location-default") as tui:
        tui.boot()
        screen = tui.open_model_picker()
        tui.exit_normally()

    _assert_scoped_models_in_picker(session, screen, [claude_parent_model])


@pytest.mark.parametrize("configured", [True, False], ids=["configured", "fresh"])
@pytest.mark.live
@pytest.mark.tui
def test_case_21_claude_provider_uses_native_models_when_discovery_disabled(
    live_session, workspace, claude_provider, configured
):
    """Scenario: launch configured and fresh Claude with discovery off and --provider.

    Expected: both use native aliases; Claude Code creates no cache after picker launch.
    """
    session = live_session
    if configured:
        _configure_hosted(session, workspace)
    session.env["UG_ENABLE_MODEL_DISCOVERY"] = "0"

    args = [] if configured else ["--workspace", workspace]
    command = [str(session.binary), "claude", *args, "--provider", claude_provider]
    with AgentTerminal(session, "claude", command, "case-21-provider-disabled") as tui:
        tui.boot()
        screen = tui.open_model_picker()
        tui.exit_normally()

    _assert_no_claude_owned_gateway_cache_after_launch(session)
    _assert_native_models_in_picker(screen)


@pytest.mark.parametrize("configured", [True, False], ids=["configured", "fresh"])
@pytest.mark.live
@pytest.mark.tui
def test_case_23_claude_location_uses_native_models_when_discovery_disabled(
    live_session, workspace, parent_schema, configured
):
    """Scenario: launch configured and fresh Claude with discovery off and a parent.

    Expected: both use native aliases; Claude Code creates no cache after picker launch.
    """
    session = live_session
    if configured:
        _configure_hosted(session, workspace)
    session.env["UG_ENABLE_MODEL_DISCOVERY"] = "0"

    args = [] if configured else ["--workspace", workspace]
    command = [str(session.binary), "claude", *args, "--model-location", parent_schema]
    with AgentTerminal(session, "claude", command, "case-23-location-disabled") as tui:
        tui.boot()
        screen = tui.open_model_picker()
        tui.exit_normally()

    models = session.workspace_state()["claude_models"]
    assert models
    assert all(model.startswith("system.ai.") for model in models.values())
    _assert_no_claude_owned_gateway_cache_after_launch(session)
    _assert_native_models_in_picker(screen)
