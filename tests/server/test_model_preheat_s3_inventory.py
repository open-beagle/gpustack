import asyncio
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.schemas.model_preheats import (
    ModelPreheatArtifact,
    ModelPreheatInventoryManifestStateEnum,
)
from gpustack.model_preheat_credentials import ModelPreheatCredentialError
from gpustack.server import model_preheat_s3_inventory as inventory_module
from gpustack.server.model_preheat_s3_inventory import (
    InventoryRefreshError,
    ModelPreheatS3Inventory,
)
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
            assert (bucket, prefix, recursive) == ("models", "storage/", True)
            return [
                SimpleNamespace(object_name=canonical_path),
                SimpleNamespace(object_name="storage/legacy/ready.json"),
                SimpleNamespace(object_name="storage/partial/files/config.json"),
                SimpleNamespace(object_name="storage/invalid/manifest.json"),
                SimpleNamespace(object_name="storage-other/foreign/manifest.json"),
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

    records, scanned, invalid, invalid_paths = inventory_module._scan_profile(
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

    assert scanned == 5
    assert invalid == 1
    assert invalid_paths == {"storage/invalid/manifest.json"}
    assert [record["artifact_id"] for record in records] == [manifest.artifact_id]
    assert records[0]["resolved_revision"] == "resolved-commit"


def test_scan_stops_after_a_hard_object_enumeration_limit(monkeypatch):
    class FakeClient:
        def __init__(self):
            self._client = self

        def list_objects(self, bucket, prefix, recursive):
            return [
                SimpleNamespace(object_name=f"storage/file-{index}")
                for index in range(3)
            ]

    monkeypatch.setattr(
        inventory_module.ModelPreheatS3Client,
        "from_minio",
        lambda **kwargs: FakeClient(),
    )
    monkeypatch.setattr(inventory_module, "MAX_OBJECTS_PER_REFRESH", 2)

    try:
        inventory_module._scan_profile(
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
    except ValueError as exc:
        assert str(exc) == "inventory_scan_limit_exceeded"
    else:
        raise AssertionError("枚举超过上限时必须中止扫描")


def test_refresh_rebuilds_current_version_and_marks_old_version_stale(
    monkeypatch, tmp_path
):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'inventory.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                ],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
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
            lambda snapshot: ([record], 1, 0, set()),
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


def test_refresh_discovers_three_sources_from_shared_s3_in_independent_database(
    monkeypatch, tmp_path
):
    async def create_profile(engine, name):
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                ],
            )
        async with AsyncSession(engine) as session:
            profile = ModelPreheatS3Profile(
                name=name,
                endpoint="https://s3.example.com",
                bucket="models",
                prefix="shared",
                access_key_encrypted={"ciphertext": "access"},
                secret_key_encrypted={"ciphertext": "secret"},
                encryption_key_version="v1",
                config_version=1,
            )
            session.add(profile)
            await session.flush()
            profile_id = profile.id
            await session.commit()
            return profile_id

    class SharedFakeS3:
        def __init__(self):
            self._client = self
            self.manifests = {}

        def publish(self, manifest):
            path = f"shared/{manifest.artifact_prefix('')}/manifest.json"
            self.manifests[path] = manifest

        def list_objects(self, bucket, prefix, recursive):
            assert (bucket, prefix, recursive) == ("models", "shared/", True)
            return [SimpleNamespace(object_name=path) for path in self.manifests]

        def read_artifact_manifest_path(self, bucket, path):
            return self.manifests[path]

        def artifact_manifest_object(self, prefix, manifest):
            return f"{prefix}/{manifest.artifact_prefix('')}/manifest.json"

    shared_s3 = SharedFakeS3()
    for source, model_id in (
        ("modelscope", "modelscope/model"),
        ("huggingface", "huggingface/model"),
        ("ollama_library", "ollama/model:latest"),
    ):
        source_dir = tmp_path / source
        source_dir.mkdir()
        (source_dir / "model.bin").write_text(source, encoding="utf-8")
        shared_s3.publish(
            build_model_preheat_manifest(
                source_dir,
                ModelPreheatIdentity(
                    source=source,
                    model_id=model_id,
                    revision="revision-1",
                    file_patterns=("model.bin",),
                ),
            )
        )

    monkeypatch.setattr(
        inventory_module.ModelPreheatS3Client,
        "from_minio",
        lambda **kwargs: shared_s3,
    )

    async def run():
        stack_a = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'a.db'}")
        stack_b = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'b.db'}")
        await create_profile(stack_a, "stack-a")
        profile_id = await create_profile(stack_b, "stack-b")

        service = ModelPreheatS3Inventory(stack_b)
        service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")
        async with AsyncSession(stack_b) as session:
            counts = await service.refresh_profile(session, profile_id, 1)
        async with AsyncSession(stack_b) as session:
            rows = (await session.exec(select(ModelPreheatArtifact))).all()
            profile = await session.get(ModelPreheatS3Profile, profile_id)
        await stack_a.dispose()
        await stack_b.dispose()
        return counts, rows, profile

    counts, rows, profile = asyncio.run(run())

    assert counts["valid"] == 3
    assert {row.source for row in rows} == {
        "modelscope",
        "huggingface",
        "ollama_library",
    }
    assert all(row.created_by_task_id is None for row in rows)
    assert profile.inventory_last_success_at is not None
    assert profile.inventory_last_scan_count == 3


def test_refresh_keeps_existing_artifact_when_manifest_is_invalid(monkeypatch, tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'invalid.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                ],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = _profile()
            session.add(profile)
            await session.flush()
            artifact = _artifact(profile.id, 1, "d" * 64)
            session.add(artifact)
            await session.commit()
            profile_id = profile.id
            manifest_path = artifact.manifest_path
        monkeypatch.setattr(
            inventory_module,
            "_scan_profile",
            lambda snapshot: ([], 1, 1, {manifest_path}),
        )
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")
        async with AsyncSession(engine) as session:
            counts = await service.refresh_profile(session, profile_id, 1)
        async with AsyncSession(engine) as session:
            row = (await session.exec(select(ModelPreheatArtifact))).one()
        await engine.dispose()
        return counts, row

    counts, row = asyncio.run(run())

    assert counts["deleted"] == 0
    assert row.manifest_state == ModelPreheatInventoryManifestStateEnum.INVALID


def test_refresh_records_credential_failure_without_leaking_ciphertext(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'credential.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[ModelPreheatS3Profile.__table__, ModelPreheatArtifact.__table__],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = _profile()
            session.add(profile)
            await session.flush()
            profile_id = profile.id
            await session.commit()
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(
            decrypt=lambda value: (_ for _ in ()).throw(
                ModelPreheatCredentialError("credential_decryption_failed")
            )
        )
        async with AsyncSession(engine) as session:
            with pytest.raises(InventoryRefreshError) as exc_info:
                await service.refresh_profile(session, profile_id, 1)
        async with AsyncSession(engine) as session:
            profile = await session.get(ModelPreheatS3Profile, profile_id)
        await engine.dispose()
        return exc_info.value.code, profile

    error_code, profile = asyncio.run(run())

    assert error_code == "inventory_credential_unavailable"
    assert profile.inventory_last_attempt_at is not None
    assert profile.inventory_last_error_code == "inventory_credential_unavailable"
    assert profile.inventory_refresh_owner is None


def test_refresh_preserves_artifacts_referenced_by_tasks_and_policy(monkeypatch, tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'references.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                ],
            )
            for statement in (
                "CREATE TABLE model_preheat_tasks (id INTEGER PRIMARY KEY, s3_profile_id INTEGER, s3_profile_config_version INTEGER, artifact_id VARCHAR(64))",
                "CREATE TABLE model_storage_sync_tasks (id INTEGER PRIMARY KEY, profile_id INTEGER, profile_config_version INTEGER, artifact_id VARCHAR(64))",
                "CREATE TABLE model_preheat_distribution_policies (id INTEGER PRIMARY KEY, profile_id INTEGER, profile_config_version INTEGER, source_artifact_id INTEGER)",
            ):
                await connection.execute(text(statement))
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = _profile()
            session.add(profile)
            await session.flush()
            artifacts = [_artifact(profile.id, 1, char * 64) for char in "abc"]
            session.add_all(artifacts)
            await session.flush()
            await session.exec(
                text(
                    "INSERT INTO model_preheat_tasks VALUES (1, :profile_id, 1, :artifact_id)"
                ).bindparams(
                    profile_id=profile.id, artifact_id=artifacts[0].artifact_id
                )
            )
            await session.exec(
                text(
                    "INSERT INTO model_storage_sync_tasks VALUES (1, :profile_id, 1, :artifact_id)"
                ).bindparams(
                    profile_id=profile.id, artifact_id=artifacts[1].artifact_id
                )
            )
            await session.exec(
                text(
                    "INSERT INTO model_preheat_distribution_policies VALUES (1, :profile_id, 1, :artifact_id)"
                ).bindparams(profile_id=profile.id, artifact_id=artifacts[2].id)
            )
            await session.commit()
            profile_id = profile.id
        monkeypatch.setattr(inventory_module, "_scan_profile", lambda snapshot: ([], 0, 0, set()))
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")
        async with AsyncSession(engine) as session:
            await service.refresh_profile(session, profile_id, 1)
        async with AsyncSession(engine) as session:
            rows = (await session.exec(select(ModelPreheatArtifact))).all()
        await engine.dispose()
        return rows

    rows = asyncio.run(run())

    assert len(rows) == 3
    assert {row.manifest_state for row in rows} == {
        ModelPreheatInventoryManifestStateEnum.STALE
    }


def test_refresh_rolls_back_when_profile_config_changes_during_scan(monkeypatch, tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'lease.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                ],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = _profile()
            session.add(profile)
            await session.flush()
            session.add(_artifact(profile.id, 1, "a" * 64))
            await session.commit()
            profile_id = profile.id

        started = threading.Event()
        release = threading.Event()

        def scan(snapshot):
            started.set()
            assert release.wait(timeout=2)
            values = _artifact_values("b" * 64)
            values.pop("last_verified_at")
            return ([values], 1, 0, set())

        monkeypatch.setattr(inventory_module, "_scan_profile", scan)
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")
        async with AsyncSession(engine) as old_session:
            refresh = asyncio.create_task(service.refresh_profile(old_session, profile_id, 1))
            await asyncio.to_thread(started.wait)
            async with AsyncSession(engine, expire_on_commit=False) as new_session:
                profile = await new_session.get(ModelPreheatS3Profile, profile_id)
                profile.config_version = 2
                new_session.add(_artifact(profile_id, 2, "c" * 64))
                await new_session.commit()
            release.set()
            with pytest.raises(InventoryRefreshError) as exc_info:
                await refresh
        async with AsyncSession(engine) as session:
            rows = (await session.exec(select(ModelPreheatArtifact))).all()
        await engine.dispose()
        return exc_info.value.code, rows

    error_code, rows = asyncio.run(run())

    assert error_code == "stale_profile_config"
    assert [(row.profile_config_version, row.manifest_state) for row in rows] == [
        (1, ModelPreheatInventoryManifestStateEnum.VALID),
        (2, ModelPreheatInventoryManifestStateEnum.VALID),
    ]


def test_expired_claim_cannot_commit_over_new_claim_from_same_service(
    monkeypatch, tmp_path
):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'claim.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[
                    ModelPreheatS3Profile.__table__,
                    ModelPreheatArtifact.__table__,
                ],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            profile = _profile()
            session.add(profile)
            await session.flush()
            session.add(_artifact(profile.id, 1, "a" * 64))
            await session.commit()
            profile_id = profile.id

        first_started = threading.Event()
        second_started = threading.Event()
        release_first = threading.Event()
        release_second = threading.Event()
        scan_count = 0

        def scan(snapshot):
            nonlocal scan_count
            scan_count += 1
            values = _artifact_values(("b" if scan_count == 1 else "c") * 64)
            values.pop("last_verified_at")
            if scan_count == 1:
                first_started.set()
                assert release_first.wait(timeout=2)
            else:
                second_started.set()
                assert release_second.wait(timeout=2)
            return ([values], 1, 0, set())

        monkeypatch.setattr(inventory_module, "_scan_profile", scan)
        service = ModelPreheatS3Inventory(engine)
        service._cipher = lambda: SimpleNamespace(decrypt=lambda value: "secret")

        async with AsyncSession(engine) as first_session:
            first_refresh = asyncio.create_task(
                service.refresh_profile(first_session, profile_id, 1)
            )
            await asyncio.to_thread(first_started.wait)
            async with AsyncSession(engine) as session:
                await session.exec(
                    update(ModelPreheatS3Profile)
                    .where(ModelPreheatS3Profile.id == profile_id)
                    .values(
                        inventory_refresh_lease_expires_at=datetime.now(timezone.utc)
                        - timedelta(seconds=1)
                    )
                )
                await session.commit()

            async def refresh_second():
                async with AsyncSession(engine) as second_session:
                    return await service.refresh_profile(second_session, profile_id, 1)

            second_refresh = asyncio.create_task(refresh_second())
            await asyncio.to_thread(second_started.wait)
            release_first.set()
            first_error = None
            try:
                await first_refresh
            except InventoryRefreshError as exc:
                first_error = exc.code
            release_second.set()
            second_counts = await second_refresh

        async with AsyncSession(engine) as session:
            rows = (await session.exec(select(ModelPreheatArtifact))).all()
            profile = await session.get(ModelPreheatS3Profile, profile_id)
        await engine.dispose()
        return first_error, second_counts, rows, profile

    first_error, second_counts, rows, profile = asyncio.run(run())

    assert first_error == "stale_profile_config"
    assert second_counts["valid"] == 1
    assert [row.artifact_id for row in rows] == ["c" * 64]
    assert profile.inventory_refresh_owner is None


def test_periodic_refresh_continues_after_one_profile_fails(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'loop.db'}")
        async with engine.begin() as connection:
            await connection.run_sync(
                SQLModel.metadata.create_all,
                tables=[ModelPreheatS3Profile.__table__],
            )
        async with AsyncSession(engine, expire_on_commit=False) as session:
            first = _profile()
            first.name = "first"
            first.inventory_refresh_interval_seconds = 60
            second = _profile()
            second.name = "second"
            second.bucket = "other-models"
            second.inventory_refresh_interval_seconds = 60
            session.add_all([first, second])
            await session.commit()

        service = ModelPreheatS3Inventory(engine)
        calls = []

        async def refresh(session, profile_id, config_version):
            calls.append(profile_id)
            if len(calls) == 1:
                raise InventoryRefreshError("inventory_scan_failed")

        service.create_refresh_job = refresh
        next_refresh_at = {}
        now = datetime(2026, 8, 24, tzinfo=timezone.utc)
        async with AsyncSession(engine) as session:
            await service._refresh_due_profiles(session, next_refresh_at, now)
            await service._refresh_due_profiles(
                session, next_refresh_at, now + timedelta(seconds=61)
            )
        await engine.dispose()
        return calls

    calls = asyncio.run(run())

    assert len(calls) == 2


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


def _profile():
    return ModelPreheatS3Profile(
        name="profile",
        endpoint="https://s3.example.com",
        bucket="models",
        prefix="storage",
        access_key_encrypted={"ciphertext": "access"},
        secret_key_encrypted={"ciphertext": "secret"},
        encryption_key_version="v1",
        config_version=1,
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
