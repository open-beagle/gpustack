from gpustack.utils.command import ensure_bool_parameter


def test_ensure_bool_parameter_adds_default_metrics_flag():
    arguments = ["--host", "0.0.0.0"]

    got = ensure_bool_parameter(arguments, "metrics")

    assert got == ["--host", "0.0.0.0", "--metrics"]
    assert arguments == ["--host", "0.0.0.0"]


def test_ensure_bool_parameter_does_not_duplicate_user_metrics_flag():
    arguments = ["--host", "0.0.0.0", "--metrics"]

    got = ensure_bool_parameter(arguments, "metrics")

    assert got is arguments


def test_ensure_bool_parameter_respects_existing_metrics_parameter():
    arguments = ["--host", "0.0.0.0"]

    got = ensure_bool_parameter(
        arguments,
        "metrics",
        existing_parameters=["--metrics"],
    )

    assert got is arguments


def test_ensure_bool_parameter_respects_existing_metrics_equals_parameter():
    arguments = ["--host", "0.0.0.0"]

    got = ensure_bool_parameter(
        arguments,
        "metrics",
        existing_parameters=["--metrics=true"],
    )

    assert got is arguments
