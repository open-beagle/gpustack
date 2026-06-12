import ast
from pathlib import Path


def test_internal_model_instance_update_route_is_hidden_from_openapi():
    route_file = (
        Path(__file__).parents[2]
        / "gpustack"
        / "routes"
        / "model_instances.py"
    )
    tree = ast.parse(route_file.read_text())

    internal_route_decorators = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue

        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if not isinstance(decorator.func, ast.Attribute):
                continue
            if decorator.func.attr != "put":
                continue
            if not decorator.args:
                continue
            path_arg = decorator.args[0]
            if isinstance(path_arg, ast.Constant) and path_arg.value == "/{id}/internal":
                internal_route_decorators.append(decorator)

    assert len(internal_route_decorators) == 1
    include_in_schema = [
        keyword.value
        for keyword in internal_route_decorators[0].keywords
        if keyword.arg == "include_in_schema"
    ]
    assert len(include_in_schema) == 1
    assert isinstance(include_in_schema[0], ast.Constant)
    assert include_in_schema[0].value is False
