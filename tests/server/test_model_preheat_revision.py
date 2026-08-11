import pytest

from gpustack.server.model_preheat_revision import (
    ModelPreheatRevisionResolutionError,
    resolve_model_preheat_revision,
)


def test_huggingface_branch_resolves_to_commit_sha():
    class FakeHfApi:
        def repo_info(self, repo_id, revision=None):
            assert (repo_id, revision) == ("org/model", "release")
            return type("Info", (), {"sha": "a" * 40})()

    resolved = resolve_model_preheat_revision(
        "huggingface",
        "org/model",
        "release",
        hf_api_factory=lambda token=None: FakeHfApi(),
    )

    assert resolved == "a" * 40


def test_modelscope_revision_uses_validated_sdk_resolution():
    class FakeHubApi:
        def get_valid_revision(self, model_id, revision=None):
            assert (model_id, revision) == ("org/model", "release")
            return "release-v1"

    resolved = resolve_model_preheat_revision(
        "modelscope",
        "org/model",
        "release",
        modelscope_api_factory=FakeHubApi,
    )

    assert resolved == "release-v1"


def test_revision_resolution_wraps_upstream_error_without_message():
    class FailingApi:
        def repo_info(self, *args, **kwargs):
            raise RuntimeError("token=plain-secret upstream detail")

    with pytest.raises(
        ModelPreheatRevisionResolutionError,
        match="remote_revision_resolution_failed",
    ) as error:
        resolve_model_preheat_revision(
            "huggingface",
            "org/model",
            "release",
            hf_api_factory=lambda token=None: FailingApi(),
        )

    assert "plain-secret" not in str(error.value)


def test_default_revision_is_resolved_for_both_hubs():
    class FakeHfApi:
        def repo_info(self, repo_id, revision=None):
            assert revision is None
            return type("Info", (), {"sha": "b" * 40})()

    class FakeHubApi:
        def get_valid_revision(self, model_id, revision=None):
            assert revision is None
            return "master-fixed"

    assert (
        resolve_model_preheat_revision(
            "huggingface",
            "org/model",
            None,
            hf_api_factory=lambda token=None: FakeHfApi(),
        )
        == "b" * 40
    )
    assert (
        resolve_model_preheat_revision(
            "modelscope",
            "org/model",
            None,
            modelscope_api_factory=FakeHubApi,
        )
        == "master-fixed"
    )
