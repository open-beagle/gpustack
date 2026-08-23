import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatInventoryManifestStateEnum,
)
from gpustack.server import model_preheat_s3_inventory as inventory_module
from gpustack.server.model_preheat_s3_inventory import ModelPreheatS3Inventory
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
from gpustack.worker.model_preheat.manifest import build_model_preheat_manifest
from gpustack.worker.model_preheat.s3_client import ModelPreheatS3ManifestError


def _identity():
    return ModelPreheatIdentity(
        source="huggingface",
        model_id="org/model",
        revision="resolved-commit",
        file_patterns=("config.json",),
    )


def test_scan_only_accepts_canonical_artifact_manifests(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("model", encoding="utf-8")
    manifest = build_model_preheat_manifest(source, _identity())
    canonical_path = f"storage/{manifest.artifact_prefix('')}/manifest.json"

    class FakeClient:
        def __init__(self):
            self._client = self

        def list_objects(self, bucket, prefix, recursive):
            assert (bucket, prefix, recursive) == ("models", "storage", True)
            return [
                SimpleNamespace(object_name=canonical_path),
                SimpleNamespace(object_name="storage/legacy/ready.json"),
                SimpleNamespace(object_name="storage/partial/files/config.json"),
                SimpleNamespace(object_name="storage/invalid/manifest.json"),
            ]

        def read_artifact_manifest_path(self, bucket, path):
            if path == canonical_path:
                return manifest
            raise ModelPreheatS3ManifestError("invalid_manifest")

        def artifact_manifest_object(self, prefix, value):
            return f"{prefix}/{value.artifact_prefix('')}/manifest.json"

    fake = FakeClient()
    monkeypatch.setattr(
        inventory_module.ModelPreheatS3Client,
        "from_minio",
        lambda **kwargs: fake,
    )

    records, scanned, invalid = inventory_module._scan_profile(
        {
            "endpoint": "https://s3.example.com",
            "bucket": "models",
            "prefix": "storage",
            "tls_enabled": True,
            "tls_verify": True,
            "region": "",
            "use_virtual_hosted_style": True,
            "access_key": "access",
            "secret_key": "secret",
        }
    )

    assert scanned == 2
    assert invalid == 1
    assert [record["artifact_id"] for record in records] == [manifest.artifact_id]
    assert records[0]["resolved_revision"] == "resolved-commit"


def test_refresh_rebuilds_current_version_and_marks_old_version_stale(
    monkeypatch, tmp_path
):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'inventory.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        async with AsyncSession(engine) as session:
            profile = ModelPreheatS3Profile(
                name="profile",
                endpoint="https://s3.example.com",
                bucket="models",
                prefix="storage",
                access_key_encrypted={"ciphertext": "access"},
                secret_key_encrypted={"ciphertext": "secret"},
                encryption_key_version="v1",
                config_version=2,
            )
            session.add(profile)
            await session.flush()
            session.add_all(
                [
                    _artifact(profile.id, 1, "a" * 64),
                    _artifact(profile.id, 2, "b" * 64),
                ]
            )
            profile_id = profile.id
            await session.commit()

        record = _artifact_values("c" * 64)
        record.pop("last_verified_at")
        monkeypatch.setattr(
            inventory_module,
            "_scan_profile",
            lambda snapshot: ([record], 1, 0),
        )
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")
        async with AsyncSession(engine) as session:
            counts = await service.refresh_profile(session, profile_id, 2)
        async with AsyncSession(engine) as session:
            rows = (
                await session.exec(
                    select(ModelPreheatArtifact).order_by(
                        ModelPreheatArtifact.profile_config_version,
                        ModelPreheatArtifact.artifact_id,
                    )
                )
            ).all()
            refreshed_profile = await session.get(ModelPreheatS3Profile, profile_id)
            ever_used_at = refreshed_profile.ever_used_at
        await engine.dispose()
        return counts, rows, ever_used_at

    counts, rows, ever_used_at = asyncio.run(run())

    assert counts == {"scanned": 1, "valid": 1, "invalid": 0, "deleted": 1}
    assert [(row.profile_config_version, row.artifact_id) for row in rows] == [
        (1, "a" * 64),
        (2, "c" * 64),
    ]
    assert rows[0].manifest_state == ModelPreheatInventoryManifestStateEnum.STALE
    assert rows[1].manifest_state == ModelPreheatInventoryManifestStateEnum.VALID
    assert ever_used_at is not None


def test_gc_is_explicitly_unsupported():
    service = ModelPreheatS3Inventory(None)

    try:
        asyncio.run(service.create_gc_job(None, 1, 1))
    except ValueError as exc:
        assert str(exc) == "artifact_gc_not_supported"
    else:
        raise AssertionError("首版不得执行 Artifact GC")


def _artifact(profile_id, version, artifact_id):
    return ModelPreheatArtifact(
        profile_id=profile_id,
        profile_config_version=version,
        **_artifact_values(artifact_id),
    )


def _artifact_values(artifact_id):
    return {
        "artifact_id": artifact_id,
        "source": "huggingface",
        "model_id": "org/model",
        "resolved_revision": "resolved-commit",
        "include_patterns": ["config.json"],
        "exclude_patterns": [],
        "manifest_path": f"storage/huggingface/org/model/{artifact_id}/manifest.json",
        "manifest_digest": "d" * 64,
        "file_count": 1,
        "total_size": 5,
        "manifest_state": ModelPreheatInventoryManifestStateEnum.VALID,
        "last_verified_at": datetime.now(timezone.utc),
    }
