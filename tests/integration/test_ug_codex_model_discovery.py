"""Codex CUJs for Tests-table cases 14, 16, 18, 20, 22, and 24."""

import pytest

pytestmark = pytest.mark.codex


def _configure_hosted(session, workspace):
    session.run(
        "configure",
        "--agents",
        "codex",
        "--workspace",
        workspace,
        "--skip-upgrade",
        "--disable-databricks-ai-tools",
        timeout=240,
    )


@pytest.mark.live
def test_case_14_configured_codex_reuses_saved_model_location(
    live_session, workspace, parent_schema, codex_parent_model
):
    """Scenario: configure Codex with --model-location, then launch without options.

    Expected: the saved parent supplies Codex's discovered model catalog.
    """
    session = live_session
    session.run(
        "configure",
        "--agents",
        "codex",
        "--workspace",
        workspace,
        "--model-location",
        parent_schema,
        "--skip-upgrade",
        "--disable-databricks-ai-tools",
        timeout=240,
    )

    models = session.codex_model_ids(["app-server", "--listen", "stdio://"])

    assert models == [codex_parent_model]


@pytest.mark.live
def test_case_16_fresh_codex_uses_system_models_when_discovery_disabled(live_session, workspace):
    """Scenario: launch fresh Codex with UG_ENABLE_MODEL_DISCOVERY=0.

    Expected: ug uses its discovered system.ai models without a scoped catalog.
    """
    session = live_session
    session.env["UG_ENABLE_MODEL_DISCOVERY"] = "0"
    models = session.codex_model_ids(
        ["--workspace", workspace, "--", "app-server", "--listen", "stdio://"]
    )

    discovered = session.workspace_state()["codex_models"]
    assert models
    assert discovered
    assert all(model.startswith("system.ai.") for model in discovered)
    assert not list((session.home / ".ucode").glob("codex-model-catalog-*.json"))


@pytest.mark.parametrize("configured", [True, False], ids=["configured", "fresh"])
@pytest.mark.live
def test_case_18_configured_codex_provider_discovers_models_by_default(
    live_session, workspace, codex_provider, codex_provider_model, configured
):
    """Scenario: launch configured and fresh Codex with --provider.

    Expected: the explicit provider supplies its exact catalog in both variants.
    """
    session = live_session
    if configured:
        _configure_hosted(session, workspace)

    args = [] if configured else ["--workspace", workspace]
    models = session.codex_model_ids(
        [*args, "--provider", codex_provider, "--", "app-server", "--listen", "stdio://"]
    )

    assert models == [codex_provider_model]


@pytest.mark.parametrize("configured", [True, False], ids=["configured", "fresh"])
@pytest.mark.live
def test_case_20_configured_codex_model_location_overrides_saved_setup(
    live_session, workspace, parent_schema, codex_parent_model, configured
):
    """Scenario: launch configured and fresh Codex with --model-location.

    Expected: the explicit parent supplies its exact catalog in both variants.
    """
    session = live_session
    if configured:
        _configure_hosted(session, workspace)

    args = [] if configured else ["--workspace", workspace]
    models = session.codex_model_ids(
        [
            *args,
            "--model-location",
            parent_schema,
            "--",
            "app-server",
            "--listen",
            "stdio://",
        ]
    )

    assert models == [codex_parent_model]


@pytest.mark.parametrize("configured", [True, False], ids=["configured", "fresh"])
@pytest.mark.live
def test_case_22_codex_provider_uses_native_models_when_discovery_disabled(
    live_session, workspace, codex_provider, configured
):
    """Scenario: launch configured and fresh Codex with discovery off and --provider.

    Expected: both use the native catalog and ug writes no scoped catalog.
    """
    session = live_session
    if configured:
        _configure_hosted(session, workspace)
    session.env["UG_ENABLE_MODEL_DISCOVERY"] = "0"

    args = [] if configured else ["--workspace", workspace]
    models = session.codex_model_ids(
        [*args, "--provider", codex_provider, "--", "app-server", "--listen", "stdio://"]
    )

    assert models
    assert not list((session.home / ".ucode").glob("codex-model-catalog-*.json"))


@pytest.mark.parametrize("configured", [True, False], ids=["configured", "fresh"])
@pytest.mark.live
def test_case_24_codex_location_uses_native_models_when_discovery_disabled(
    live_session, workspace, parent_schema, configured
):
    """Scenario: launch configured and fresh Codex with discovery off and a parent.

    Expected: both use the native catalog and ug writes no scoped catalog.
    """
    session = live_session
    if configured:
        _configure_hosted(session, workspace)
    session.env["UG_ENABLE_MODEL_DISCOVERY"] = "0"

    args = [] if configured else ["--workspace", workspace]
    models = session.codex_model_ids(
        [
            *args,
            "--model-location",
            parent_schema,
            "--",
            "app-server",
            "--listen",
            "stdio://",
        ]
    )

    discovered = session.workspace_state()["codex_models"]
    assert models
    assert discovered
    assert all(model.startswith("system.ai.") for model in discovered)
    assert not list((session.home / ".ucode").glob("codex-model-catalog-*.json"))
