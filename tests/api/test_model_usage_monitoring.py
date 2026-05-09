from datetime import datetime, timezone

from gpustack.api.middlewares import (
    get_model_usage_context,
    sanitize_error_message,
    sse_event_bytes,
    sse_event_data,
    sse_event_with_data,
)
from gpustack.routes.model_usage import (
    call_csv_fieldnames,
    calls_csv_header,
    rebuild_stats,
    should_use_log_aggregation,
    stat_statement,
)
from gpustack.schemas.model_usage import ModelUsageDailyStat
from gpustack.schemas.model_usage import OperationEnum
from gpustack.utils.envs import get_gpustack_env_list


def test_get_model_usage_context_matches_chat_completion_paths():
    operation, _ = get_model_usage_context("/v1/chat/completions")

    assert operation == OperationEnum.CHAT_COMPLETION


def test_get_model_usage_context_ignores_unknown_paths():
    operation, response_class = get_model_usage_context("/api/v1/dashboard")

    assert operation is None
    assert response_class is None


def test_sanitize_error_message_truncates_long_messages():
    message = "x" * 2048

    assert sanitize_error_message(message) == "x" * 1024


def test_sanitize_error_message_redacts_sensitive_markers():
    assert sanitize_error_message("Authorization: Bearer secret") == "[redacted]"


def test_get_gpustack_env_list_parses_comma_separated_values(monkeypatch):
    monkeypatch.setenv(
        "GPUSTACK_TRUSTED_PROXY_CIDRS",
        "10.0.0.0/24, 192.168.1.0/24,",
    )

    assert get_gpustack_env_list("TRUSTED_PROXY_CIDRS") == [
        "10.0.0.0/24",
        "192.168.1.0/24",
    ]


def test_call_csv_fieldnames_do_not_include_sensitive_prompt_or_secret():
    fieldnames = call_csv_fieldnames()

    assert "prompt" not in fieldnames
    assert "secret_key" not in fieldnames
    assert "hashed_secret_key" not in fieldnames


def test_rebuild_stats_route_is_registered():
    assert rebuild_stats.__name__ == "rebuild_stats"


def test_calls_csv_header_contains_expected_fields_once():
    header = calls_csv_header()

    assert header.count("request_id") == 1
    assert "error_message" in header


def test_non_day_boundary_end_uses_log_aggregation():
    end_at = datetime(2026, 5, 9, 0, 0, tzinfo=timezone.utc)

    assert should_use_log_aggregation(None, end_at, None) is True


def test_day_boundary_end_can_use_stat_aggregation():
    end_at = datetime(2026, 5, 9, 23, 59, 59, 999999, tzinfo=timezone.utc)

    assert should_use_log_aggregation(None, end_at, None) is False


def test_stat_statement_exposes_common_token_aliases():
    statement = stat_statement(ModelUsageDailyStat).subquery()

    assert "prompt_tokens" in statement.c
    assert "completion_tokens" in statement.c
    assert "total_tokens" in statement.c


def test_sse_event_data_extracts_data_with_event_prefix():
    event = 'event: message\ndata: {"usage":{"total_tokens":3}}'

    assert sse_event_data(event) == '{"usage":{"total_tokens":3}}'


def test_sse_event_data_joins_multiple_data_lines():
    event = 'event: message\ndata: first\ndata: second'

    assert sse_event_data(event) == "first\nsecond"


def test_sse_event_bytes_preserves_non_data_event():
    event = 'event: ping\nid: 1'

    assert sse_event_bytes(event) == b'event: ping\nid: 1\n\n'


def test_sse_event_with_data_preserves_event_metadata():
    event = 'event: message\nid: 1\ndata: old'

    assert sse_event_with_data(event, "new") == b'event: message\nid: 1\ndata: new\n\n'
