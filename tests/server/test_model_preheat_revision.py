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


def test_modelscope_revision_uses_commit_when_sdk_proves_it():
    class FakeHubApi:
        def get_valid_revision(self, model_id, revision=None):
            assert (model_id, revision) == ("org/model", "release")
            return "c" * 40

    resolved = resolve_model_preheat_revision(
        "modelscope",
        "org/model",
        "release",
        modelscope_api_factory=FakeHubApi,
    )

    assert resolved == "c" * 40


def test_explicit_commit_revision_does_not_access_hub():
    class ForbiddenApi:
        def __init__(self, *args, **kwargs):
            raise AssertionError("不应访问 Hub")

    commit = "c" * 40
    assert (
        resolve_model_preheat_revision(
            "modelscope",
            "org/model",
            commit,
            modelscope_api_factory=ForbiddenApi,
            modelscope_file_api_factory=ForbiddenApi,
        )
        == commit
    )


def test_modelscope_moving_revision_uses_remote_file_fingerprint():
    class FakeHubApi:
        def get_valid_revision(self, model_id, revision=None):
            return revision or "master"

    class FakeFileApi:
        def __init__(self, blob_id):
            self.blob_id = blob_id

        def list_repo_files(self, model_id, repo_type, *, revision, recursive):
            assert (model_id, repo_type, revision, recursive) == (
                "org/model",
                "model",
                "release",
                True,
            )
            return [
                type(
                    "File",
                    (),
                    {"path": "model.bin", "size": 10, "blob_id": self.blob_id},
                )()
            ]

    first = resolve_model_preheat_revision(
        "modelscope",
        "org/model",
        "release",
        modelscope_api_factory=FakeHubApi,
        modelscope_file_api_factory=lambda: FakeFileApi("a" * 64),
    )
    second = resolve_model_preheat_revision(
        "modelscope",
        "org/model",
        "release",
        modelscope_api_factory=FakeHubApi,
        modelscope_file_api_factory=lambda: FakeFileApi("b" * 64),
    )

    assert first.startswith("modelscope-filelist-v1-")
    assert second.startswith("modelscope-filelist-v1-")
    assert first != second


def test_modelscope_aliases_with_same_files_share_fingerprint():
    class FakeHubApi:
        def get_valid_revision(self, model_id, revision=None):
            return revision

    class FakeFileApi:
        def list_repo_files(self, model_id, repo_type, *, revision, recursive):
            return [
                type(
                    "File",
                    (),
                    {"path": "model.bin", "size": 10, "blob_id": "a" * 64},
                )()
            ]

    resolved = {
        resolve_model_preheat_revision(
            "modelscope",
            "org/model",
            alias,
            modelscope_api_factory=FakeHubApi,
            modelscope_file_api_factory=FakeFileApi,
        )
        for alias in ("release", "stable")
    }

    assert len(resolved) == 1


def test_modelscope_file_fingerprint_rejects_missing_content_digest():
    class FakeHubApi:
        def get_valid_revision(self, model_id, revision=None):
            return revision

    class MissingDigestFileApi:
        def list_repo_files(self, model_id, repo_type, *, revision, recursive):
            return [
                type(
                    "File",
                    (),
                    {"path": "model.bin", "size": 10, "blob_id": None, "lfs": None},
                )()
            ]

    with pytest.raises(
        ModelPreheatRevisionResolutionError,
        match="remote_revision_resolution_failed",
    ):
        resolve_model_preheat_revision(
            "modelscope",
            "org/model",
            "release",
            modelscope_api_factory=FakeHubApi,
            modelscope_file_api_factory=MissingDigestFileApi,
        )


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
            return "d" * 40

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
        == "d" * 40
    )


def test_ollama_uses_trusted_registry_digest():
    assert (
        resolve_model_preheat_revision(
            "ollama_library",
            "llama3:latest",
            "latest",
            ollama_digest_resolver=lambda model_id, revision: "sha256:" + "a" * 64,
        )
        == "sha256:" + "a" * 64
    )


def test_ollama_untrusted_tag_requires_two_phase_snapshot():
    assert (
        resolve_model_preheat_revision(
            "ollama_library",
            "llama3:latest",
            "latest",
            ollama_digest_resolver=lambda model_id, revision: None,
        )
        is None
    )
