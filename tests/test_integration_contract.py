"""Keep the black-box suite independent of application internals and test doubles."""

import ast
import re
from pathlib import Path


def _markers(nodes):
    return {
        node.attr
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "pytest"
        and node.value.attr == "mark"
    }


def _parametrize_values(node, argument):
    values = []
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call) or len(decorator.args) < 2:
            continue
        function = decorator.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "parametrize"
            and isinstance(function.value, ast.Attribute)
            and function.value.attr == "mark"
            and isinstance(function.value.value, ast.Name)
            and function.value.value.id == "pytest"
        ):
            continue
        if isinstance(decorator.args[0], ast.Constant) and decorator.args[0].value == argument:
            values.append(ast.literal_eval(decorator.args[1]))
    return values


def test_integration_suite_uses_only_public_process_boundaries():
    violations = []
    for path in (Path(__file__).parent / "integration").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            if any(module.split(".")[0] in {"ucode", "mock", "unittest"} for module in modules):
                violations.append(f"{path.name}:{node.lineno}: imports application or test doubles")
            if isinstance(node, ast.Name) and node.id in {
                "monkeypatch",
                "MonkeyPatch",
                "Mock",
                "MagicMock",
                "patch",
                "setattr",
                "delattr",
            }:
                violations.append(f"{path.name}:{node.lineno}: uses {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in {
                "MonkeyPatch",
                "Mock",
                "MagicMock",
                "mock",
                "patch",
                "skip",
                "skipif",
                "xfail",
            }:
                violations.append(f"{path.name}:{node.lineno}: uses {node.attr}")
    assert not violations, "\n".join(violations)


def test_live_integration_cases_belong_to_exactly_one_ci_agent():
    for path in (Path(__file__).parent / "integration").glob("test_*.py"):
        tree = ast.parse(path.read_text())
        module_marks = _markers(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            )
        )
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                marks = module_marks | _markers(node.decorator_list)
                if marks & {"live", "managed"}:
                    assert len(marks & {"claude", "codex"}) == 1, node.name


def test_unmanaged_model_discovery_cases_match_the_tests_table():
    root = Path(__file__).parent / "integration"
    seen = set()
    executions = 0
    for path in root.glob("test_ug_*_model_discovery.py"):
        tree = ast.parse(path.read_text())
        module_marks = _markers(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pytestmark"
                for target in node.targets
            )
        )
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            match = re.match(r"test_case_(\d{2})_", node.name)
            if match is None:
                continue
            case = int(match.group(1))
            seen.add(case)
            marks = module_marks | _markers(node.decorator_list)
            assert marks & {"managed", "live"} == {"live"}, node.name
            has_configured_argument = any(arg.arg == "configured" for arg in node.args.args)
            parametrizations = _parametrize_values(node, "configured")
            if case >= 17:
                assert has_configured_argument, node.name
                assert parametrizations == [[True, False]], node.name
                executions += 2
            else:
                assert not has_configured_argument, node.name
                assert parametrizations == [], node.name
                executions += 1
    assert seen == set(range(13, 25))
    assert executions == 20


def test_smoke_covers_hosted_configuration_and_headless_for_both_agents():
    smoke = set()
    for path in (Path(__file__).parent / "integration").glob("test_*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and "smoke" in _markers(node.decorator_list):
                smoke.add(node.name)
    assert smoke == {
        "test_ug_configure_claude_databricks",
        "test_ug_configure_codex_databricks",
        "test_ug_claude_headless_prompt_argument",
        "test_ug_codex_headless_prompt_argument",
    }


def test_integration_tests_describe_the_scenario_and_expected_result():
    root = Path(__file__).parent / "integration"
    violations = []
    for path in root.rglob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            description = ast.get_docstring(node) or ""
            if "Scenario:" not in description or "Expected:" not in description:
                violations.append(f"{path.name}:{node.lineno}: describe Scenario and Expected")
    assert not violations, "\n".join(violations)
