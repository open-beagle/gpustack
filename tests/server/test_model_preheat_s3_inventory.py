import asyncio
import ast
import hashlib
import importlib.util
import io
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    event,
    update,
)
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.schema import CreateTable
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheats import (
    ModelPreheatCachedModel,
    ModelPreheatInventoryGeneration,
    ModelPreheatInventoryGenerationStateEnum,
    ModelPreheatInventoryJob,
    ModelPreheatInventoryJobStateEnum,
    ModelPreheatInventoryManifestStateEnum,
    ModelPreheatInventoryScanSnapshot,
    ModelPreheatInventorySelectionLock,
    ModelPreheatPublicationMarker,
    ModelPreheatBackfillPolicyEnum,
    ModelPreheatExecutionStateEnum,
    ModelPreheatTargetScopeEnum,
    ModelPreheatTask,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerIdentity,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.server.model_preheat_s3_inventory import (
    InventoryRecord,
    InventoryScan,
    InventoryS3Error,
    ModelPreheatS3Inventory,
    MinioInventoryStore,
    ScannedGeneration,
    upsert_verified_publication,
)
from gpustack.server.model_preheat_controller import ReadyProbeResult
from gpustack.schemas.model_preheat_s3_profiles import ModelPreheatS3Profile
from gpustack.worker.model_preheat.identity import ModelPreheatIdentity
from gpustack.worker.model_preheat.manifest import ManifestFile, ModelPreheatManifest
from gpustack.worker.model_preheat.s3_client import ModelPreheatS3Client


class FakeStore:
    def __init__(self, scan):
        self.scan_result = scan
        self.scan_calls = 0
        self.ready = {}
        self.objects = {}
        self.delete_failures = set()
        self.deleted = []
        self.ready_reads = 0

    def scan(self, profile):
        self.scan_calls += 1
        if isinstance(self.scan_result, Exception):
            raise self.scan_result
        return self.scan_result

    def read_ready_reference(self, profile, ready_path):
        self.ready_reads += 1
        return self.ready.get(ready_path)

    def list_generation_objects(self, profile, generation_path):
        return list(self.objects.get(generation_path, ()))

    def iter_generation_objects(self, profile, generation_path):
        yield from self.objects.get(generation_path, ())

    def delete_object(self, profile, object_path):
        if object_path in self.delete_failures:
            raise OSError("secret-access-key")
        for objects in self.objects.values():
            if object_path in objects:
                objects.remove(object_path)
        self.deleted.append(object_path)


class MemoryResponse(io.BytesIO):
    def release_conn(self):
        pass


class MemoryMinio:
    def __init__(self):
        self.objects = {}

    def list_objects(self, bucket, prefix, recursive=True):
        del recursive
        return [
            type("Object", (), {"object_name": name})()
            for (stored_bucket, name), value in self.objects.items()
            if stored_bucket == bucket and name.startswith(prefix)
        ]

    def get_object(self, bucket, name):
        if (bucket, name) not in self.objects:
            raise FileNotFoundError(name)
        return MemoryResponse(self.objects[(bucket, name)])

    def remove_object(self, bucket, name):
        self.objects.pop((bucket, name), None)


def record(
    *,
    state="valid",
    digest="a" * 64,
    generation="preheat-11111111-1111-1111-1111-111111111111",
):
    return InventoryRecord(
        cache_key="c" * 64,
        source="huggingface",
        model_id="org/model",
        resolved_revision="f" * 40,
        include_patterns=(),
        exclude_patterns=("*.bin",),
        generation_id=generation,
        ready_path="model-cache/v1/huggingface/org/model/rev/sel/ready.json",
        manifest_path=f"model-cache/v1/x/generations/{generation}/.gpustack-manifest.json",
        manifest_digest=digest,
        file_count=2,
        total_size=30,
        manifest_state=state,
    )


def generation(name="old", *, referenced=False, fingerprint="a" * 64):
    path = f"model-cache/v1/x/generations/{name}"
    return ScannedGeneration(
        generation_path=path,
        ready_path="model-cache/v1/x/ready.json",
        ready_fingerprint=fingerprint,
        referenced=referenced,
    )


@pytest.fixture
def engine(tmp_path):
    value = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'inventory.db'}",
        poolclass=NullPool,
    )

    async def create():
        async with value.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)

    asyncio.run(create())
    yield value
    asyncio.run(value.dispose())


async def run_refresh(engine, store, profile_id=7):
    service = ModelPreheatS3Inventory(engine, store_factory=lambda profile: store)
    async with AsyncSession(engine) as session:
        job = await service.create_refresh_job(session, profile_id, 1)
    await service.run_job(job.id)
    async with AsyncSession(engine) as session:
        return await session.get(ModelPreheatInventoryJob, job.id)


def test_refresh_upserts_and_invalid_ready_can_recover(engine):
    store = FakeStore(InventoryScan(records=(record(state="invalid"),), generations=()))
    first = asyncio.run(run_refresh(engine, store))
    assert first.state == "ready"

    async def read_state():
        async with AsyncSession(engine) as session:
            item = (await session.exec(select(ModelPreheatCachedModel))).one()
            return item.manifest_state, item.manifest_digest

    assert asyncio.run(read_state()) == (
        ModelPreheatInventoryManifestStateEnum.INVALID,
        "a" * 64,
    )
    store.scan_result = InventoryScan(
        records=(record(state="valid", digest="b" * 64),), generations=()
    )
    asyncio.run(run_refresh(engine, store))
    assert asyncio.run(read_state()) == (
        ModelPreheatInventoryManifestStateEnum.VALID,
        "b" * 64,
    )


def test_inventory_migration_matches_schema_on_all_supported_dialects():
    migration_root = Path(__file__).parents[2] / "gpustack/migrations/versions"
    migration_paths = (
        migration_root / "2026_08_11_1800-c9d0e1f2a3b4_add_preheat_s3_inventory.py",
        migration_root / "2026_08_11_2100-d0e1f2a3b4c5_add_preheat_worker_identity.py",
    )
    migration_columns = {}
    for migration_path in migration_paths:
        tree = ast.parse(migration_path.read_text())
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if not isinstance(call.func, ast.Attribute) or not call.args:
                continue
            if call.func.attr == "create_table" and isinstance(
                call.args[0], ast.Constant
            ):
                migration_columns[call.args[0].value] = {
                    arg.args[0].value
                    for arg in call.args[1:]
                    if isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Attribute)
                    and arg.func.attr == "Column"
                    and arg.args
                    and isinstance(arg.args[0], ast.Constant)
                }
            elif (
                call.func.attr == "add_column"
                and len(call.args) == 2
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[1], ast.Call)
                and call.args[1].args
                and isinstance(call.args[1].args[0], ast.Constant)
            ):
                migration_columns.setdefault(call.args[0].value, set()).add(
                    call.args[1].args[0].value
                )

    models = (
        ModelPreheatCachedModel,
        ModelPreheatInventoryJob,
        ModelPreheatInventoryGeneration,
        ModelPreheatInventorySelectionLock,
        ModelPreheatInventoryScanSnapshot,
        ModelPreheatPublicationMarker,
        ModelPreheatWorkerIdentity,
    )
    for model in models:
        assert migration_columns[model.__tablename__] == set(
            model.__table__.columns.keys()
        )
        for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
            assert str(CreateTable(model.__table__).compile(dialect=dialect))
    assert "lease_expires_at" not in ModelPreheatTask.__table__.columns
    core = (
        migration_root / "2026_08_10_1000-f6a7b8c9d0e1_add_model_preheat_core.py"
    ).read_text()
    successor = migration_paths[1].read_text()
    assert "ix_preheat_worker_uuid_state" not in core
    assert "ix_preheat_worker_uuid_state" in successor


def test_worker_identity_migration_bootstraps_existing_workers_as_unclaimed():
    migration_path = (
        Path(__file__).parents[2]
        / "gpustack/migrations/versions/"
        / "2026_08_11_2100-d0e1f2a3b4c5_add_preheat_worker_identity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_preheat_worker_identity_migration", migration_path
    )
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    statement = migration._bootstrap_existing_workers_statement()

    for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
        sql = str(
            statement.compile(dialect=dialect, compile_kwargs={"literal_binds": True})
        )
        assert "INSERT INTO model_preheat_worker_identities" in sql
        assert "FROM workers" in sql
        assert "worker_id" in sql
        assert "bootstrap_required" in sql

    engine = create_engine("sqlite://")
    metadata = MetaData()
    workers = Table(
        "workers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("worker_uuid", String(256), nullable=False),
    )
    identities = Table(
        "model_preheat_worker_identities",
        metadata,
        Column("worker_id", Integer, nullable=False),
        Column("worker_uuid", String(256), nullable=False),
        Column("token_hash", String(64), nullable=True),
        Column("token_version", Integer, nullable=False),
        Column("bootstrap_required", Boolean, nullable=False),
        Column("expires_at", DateTime, nullable=True),
        Column("revoked_at", DateTime, nullable=True),
        Column("created_at", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            workers.insert(), {"id": 7, "worker_uuid": "upgraded-worker"}
        )
        connection.execute(statement)
        row = connection.execute(identities.select()).mappings().one()
    assert row["worker_id"] == 7
    assert row["worker_uuid"] == "upgraded-worker"
    assert row["bootstrap_required"] is True
    assert row["token_hash"] is None
    assert row["expires_at"] is None


def test_production_scan_marks_tampered_ready_invalid_then_recovers():
    identity = ModelPreheatIdentity("huggingface", "org/model", "commit", [])
    manifest = ModelPreheatManifest(
        identity=identity,
        files=(ManifestFile("config.json", 2, "1" * 64),),
        cache_key="c" * 64,
        selection_digest="d" * 64,
        generation_id="preheat-11111111-1111-1111-1111-111111111111",
    )
    minio = MemoryMinio()
    client = ModelPreheatS3Client(minio)
    manifest_bytes = manifest.to_json_bytes()
    manifest_path = client.manifest_object("cache", manifest)
    ready_path = client.ready_object("cache", manifest)
    valid_ready = client._ready_payload(
        "cache", manifest, hashlib.sha256(manifest_bytes).hexdigest()
    )
    minio.objects[("models", manifest_path)] = manifest_bytes
    minio.objects[("models", ready_path)] = valid_ready
    profile = type("Profile", (), {"bucket": "models", "prefix": "cache"})()
    store = MinioInventoryStore.__new__(MinioInventoryStore)
    store.client = client

    tampered = json.loads(valid_ready)
    tampered["manifest_sha256"] = "0" * 64
    minio.objects[("models", ready_path)] = json.dumps(tampered).encode()
    invalid = store.scan(profile)
    assert invalid.records[0].manifest_state == "invalid"

    minio.objects[("models", ready_path)] = valid_ready
    recovered = store.scan(profile)
    assert recovered.records[0].manifest_state == "valid"
    assert recovered.records[0].manifest_digest == manifest.digest


def test_failed_scan_keeps_last_successful_inventory_and_redacts_error(engine):
    store = FakeStore(InventoryScan(records=(record(),), generations=()))
    asyncio.run(run_refresh(engine, store))
    store.scan_result = InventoryS3Error("plain-secret-key")
    failed = asyncio.run(run_refresh(engine, store))
    assert failed.state == "error"
    assert failed.error_code == "inventory_scan_failed"
    assert "secret" not in (failed.error_message or "")

    async def count_valid():
        async with AsyncSession(engine) as session:
            rows = (await session.exec(select(ModelPreheatCachedModel))).all()
            return len(rows), rows[0].manifest_state

    assert asyncio.run(count_valid()) == (
        1,
        ModelPreheatInventoryManifestStateEnum.VALID,
    )


def test_successful_scan_marks_disappeared_ready_missing(engine):
    store = FakeStore(InventoryScan(records=(record(),), generations=()))
    asyncio.run(run_refresh(engine, store))
    store.scan_result = InventoryScan(records=(), generations=())
    asyncio.run(run_refresh(engine, store))

    async def read_state():
        async with AsyncSession(engine) as session:
            item = (await session.exec(select(ModelPreheatCachedModel))).one()
            return item.manifest_state

    assert asyncio.run(read_state()) == ModelPreheatInventoryManifestStateEnum.MISSING


def test_published_seed_upsert_is_idempotent_and_paths_are_server_derived(engine):
    task = ModelPreheatTask(
        id=41,
        source="huggingface",
        model_id="org/model",
        resolved_revision="f" * 40,
        include_patterns=[],
        exclude_patterns=["*.bin"],
        selection_digest="d" * 64,
        cache_key="c" * 64,
        generation_id="preheat-11111111-1111-1111-1111-111111111111",
        target_scope=ModelPreheatTargetScopeEnum.SEED_WORKER,
        target_worker_uuids=["worker"],
        target_worker_snapshot=[],
        s3_profile_id=7,
        s3_profile_config_version=1,
        s3_profile_snapshot_encrypted={},
        encryption_key_version="v1",
        s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
    )

    async def publish_twice():
        async with AsyncSession(engine) as session:
            session.add(
                ModelPreheatS3Profile(
                    id=7,
                    name="inventory-profile",
                    endpoint="https://s3.example.com",
                    bucket="models",
                    prefix="cache",
                    access_key_encrypted={"ciphertext": "x"},
                    secret_key_encrypted={"ciphertext": "y"},
                    encryption_key_version="v1",
                    config_version=1,
                )
            )
            session.add(task)
            await session.commit()
            await session.refresh(task)
            identity = ModelPreheatIdentity(
                task.source,
                task.model_id,
                task.resolved_revision,
                task.include_patterns,
            )
            client = ModelPreheatS3Client(None)
            selection_prefix = client._selection_prefix(
                "cache", identity, task.selection_digest
            )
            ready = ReadyProbeResult(
                manifest_digest="e" * 64,
                generation_id=task.generation_id,
                ready_path=client._join_object_name(selection_prefix, "ready.json"),
                manifest_path=client._join_object_name(
                    selection_prefix,
                    "generations",
                    task.generation_id,
                    ".gpustack-manifest.json",
                ),
                cache_key=task.cache_key,
                selection_digest=task.selection_digest,
                profile_config_version=1,
                file_count=2,
                total_size=30,
            )
            assert await upsert_verified_publication(
                session,
                task,
                ready,
                expected_attempt=task.attempt,
                expected_profile_version=1,
            )
            assert await upsert_verified_publication(
                session,
                task,
                ready,
                expected_attempt=task.attempt,
                expected_profile_version=1,
            )
            await session.exec(
                update(ModelPreheatTask)
                .where(ModelPreheatTask.id == task.id)
                .values(attempt=2)
            )
            await session.flush()
            await session.refresh(task)
            newer = replace(ready, manifest_digest="f" * 64)
            assert await upsert_verified_publication(
                session,
                task,
                newer,
                expected_attempt=2,
                expected_profile_version=1,
            )
            assert not await upsert_verified_publication(
                session,
                task,
                ready,
                expected_attempt=1,
                expected_profile_version=1,
            )
            forged = ReadyProbeResult(
                **{**ready.__dict__, "ready_path": "access-key/forged.json"}
            )
            assert not await upsert_verified_publication(
                session,
                task,
                forged,
                expected_attempt=task.attempt,
                expected_profile_version=1,
            )
            await session.commit()
        async with AsyncSession(engine) as session:
            return (await session.exec(select(ModelPreheatCachedModel))).all()

    rows = asyncio.run(publish_twice())
    assert len(rows) == 1
    assert rows[0].source_parent_attempt == 2
    assert rows[0].manifest_digest == "f" * 64
    assert rows[0].ready_path.endswith(f"/{'d' * 64}/ready.json")
    assert "access-key" not in rows[0].ready_path


def test_apply_scan_losing_claim_mid_apply_writes_nothing(engine):
    scan = InventoryScan(
        records=(record(digest="b" * 64), record(digest="c" * 64)),
        generations=(),
    )
    scan = InventoryScan(
        records=(scan.records[0], replace(scan.records[1], cache_key="d" * 64)),
        generations=(),
    )
    service = ModelPreheatS3Inventory(engine, apply_batch_size=1)

    async def run():
        async with AsyncSession(engine) as session:
            job = await service.create_refresh_job(session, 7, 1)
            job.state = ModelPreheatInventoryJobStateEnum.RUNNING
            job.claim_token = "old-owner"
            job.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=1)
            job_id = job.id
            session.add(job)
            await session.commit()

        calls = 0

        async def lose_claim(job_id_arg, token):
            nonlocal calls
            calls += 1
            async with AsyncSession(engine) as session:
                await session.exec(
                    update(ModelPreheatInventoryJob)
                    .where(ModelPreheatInventoryJob.id == job_id_arg)
                    .values(claim_token="new-owner")
                )
                await session.commit()
            return False

        service._renew_claim = lose_claim
        await service._apply_scan(job_id, scan, "old-owner")
        async with AsyncSession(engine) as session:
            return (await session.exec(select(ModelPreheatCachedModel))).all(), calls

    rows, renew_calls = asyncio.run(run())
    assert renew_calls == 1
    assert rows == []


def test_refresh_does_not_mark_publication_newer_than_scan_missing(engine):
    store = FakeStore(InventoryScan(records=(record(),), generations=()))
    asyncio.run(run_refresh(engine, store))
    service = ModelPreheatS3Inventory(engine)

    async def publish_after_scan_started():
        scan_started_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        async with AsyncSession(engine) as session:
            job = await service.create_refresh_job(session, 7, 1)
            job.state = ModelPreheatInventoryJobStateEnum.RUNNING
            job.claim_token = "refresh-owner"
            job.scan_started_at = scan_started_at
            job.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
            cached = (await session.exec(select(ModelPreheatCachedModel))).one()
            cached.manifest_digest = "f" * 64
            cached.manifest_state = ModelPreheatInventoryManifestStateEnum.VALID
            cached.last_verified_at = datetime.now(timezone.utc)
            session.add(job)
            session.add(cached)
            job_id = job.id
            await session.commit()
        await service._apply_scan(
            job_id, InventoryScan(records=(), generations=()), "refresh-owner"
        )
        async with AsyncSession(engine) as session:
            return (await session.exec(select(ModelPreheatCachedModel))).one()

    cached = asyncio.run(publish_after_scan_started())
    assert cached.manifest_state == ModelPreheatInventoryManifestStateEnum.VALID
    assert cached.manifest_digest == "f" * 64


def test_missing_cas_uses_persisted_revision_not_same_second_timestamp(engine):
    store = FakeStore(InventoryScan(records=(record(),), generations=()))
    asyncio.run(run_refresh(engine, store))
    service = ModelPreheatS3Inventory(engine)
    same_second = datetime(2026, 8, 11, 12, 0, 0, tzinfo=timezone.utc)

    async def publication_after_snapshot():
        async with AsyncSession(engine) as session:
            job = await service.create_refresh_job(session, 7, 1)
            cached = (await session.exec(select(ModelPreheatCachedModel))).one()
            job.state = ModelPreheatInventoryJobStateEnum.RUNNING
            job.claim_token = "refresh-owner"
            job.scan_started_at = same_second
            job.lease_expires_at = same_second + timedelta(minutes=5)
            session.add(job)
            await session.flush()
            session.add(
                ModelPreheatInventoryScanSnapshot(
                    job_id=job.id,
                    cached_model_id=cached.id,
                    revision=cached.revision,
                )
            )
            job_id = job.id
            cached_id = cached.id
            await session.commit()

        async with AsyncSession(engine) as session:
            await session.exec(
                update(ModelPreheatCachedModel)
                .where(ModelPreheatCachedModel.id == cached_id)
                .values(
                    manifest_digest="f" * 64,
                    manifest_state=ModelPreheatInventoryManifestStateEnum.VALID,
                    last_verified_at=same_second,
                    revision=ModelPreheatCachedModel.revision + 1,
                )
            )
            await session.commit()

        await service._apply_scan(
            job_id, InventoryScan(records=(), generations=()), "refresh-owner"
        )
        async with AsyncSession(engine) as session:
            return (await session.exec(select(ModelPreheatCachedModel))).one()

    cached = asyncio.run(publication_after_snapshot())
    assert cached.revision == 2
    assert cached.manifest_state == ModelPreheatInventoryManifestStateEnum.VALID
    assert cached.manifest_digest == "f" * 64


def test_publication_during_refresh_lock_window_is_not_marked_missing(engine):
    store = FakeStore(InventoryScan(records=(record(),), generations=()))
    asyncio.run(run_refresh(engine, store))
    service = ModelPreheatS3Inventory(engine)
    original_acquire = service.acquire_selection_lock
    published = False

    async def publish_then_acquire(profile_id, selection_key, owner, operation):
        nonlocal published
        if not published:
            published = True
            async with AsyncSession(engine) as session:
                await session.exec(
                    update(ModelPreheatCachedModel)
                    .where(ModelPreheatCachedModel.profile_id == profile_id)
                    .values(
                        manifest_digest="e" * 64,
                        revision=ModelPreheatCachedModel.revision + 1,
                    )
                )
                await session.commit()
        return await original_acquire(profile_id, selection_key, owner, operation)

    service.acquire_selection_lock = publish_then_acquire

    async def run():
        async with AsyncSession(engine) as session:
            job = await service.create_refresh_job(session, 7, 1)
            job.state = ModelPreheatInventoryJobStateEnum.RUNNING
            job.claim_token = "refresh-owner"
            job.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
            cached = (await session.exec(select(ModelPreheatCachedModel))).one()
            session.add(job)
            await session.flush()
            session.add(
                ModelPreheatInventoryScanSnapshot(
                    job_id=job.id,
                    cached_model_id=cached.id,
                    revision=cached.revision,
                )
            )
            job_id = job.id
            await session.commit()
        await service._apply_scan(
            job_id, InventoryScan(records=(), generations=()), "refresh-owner"
        )
        async with AsyncSession(engine) as session:
            return (await session.exec(select(ModelPreheatCachedModel))).one()

    cached = asyncio.run(run())
    assert published is True
    assert cached.revision == 2
    assert cached.manifest_state == ModelPreheatInventoryManifestStateEnum.VALID
    assert cached.manifest_digest == "e" * 64


def test_refresh_record_cas_does_not_overwrite_concurrent_publication(engine):
    store = FakeStore(InventoryScan(records=(record(),), generations=()))
    asyncio.run(run_refresh(engine, store))
    service = ModelPreheatS3Inventory(engine)
    original_acquire = service.acquire_selection_lock

    async def publish_then_acquire(profile_id, selection_key, owner, operation):
        async with AsyncSession(engine) as session:
            await session.exec(
                update(ModelPreheatCachedModel)
                .where(ModelPreheatCachedModel.profile_id == profile_id)
                .values(
                    manifest_digest="e" * 64,
                    revision=ModelPreheatCachedModel.revision + 1,
                )
            )
            await session.commit()
        service.acquire_selection_lock = original_acquire
        return await original_acquire(profile_id, selection_key, owner, operation)

    service.acquire_selection_lock = publish_then_acquire

    async def run():
        async with AsyncSession(engine) as session:
            job = await service.create_refresh_job(session, 7, 1)
            cached = (await session.exec(select(ModelPreheatCachedModel))).one()
            job.state = ModelPreheatInventoryJobStateEnum.RUNNING
            job.claim_token = "refresh-owner"
            job.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
            session.add(job)
            await session.flush()
            session.add(
                ModelPreheatInventoryScanSnapshot(
                    job_id=job.id,
                    cached_model_id=cached.id,
                    revision=cached.revision,
                )
            )
            job_id = job.id
            await session.commit()
        await service._apply_scan(
            job_id,
            InventoryScan(records=(record(digest="a" * 64),), generations=()),
            "refresh-owner",
        )
        async with AsyncSession(engine) as session:
            return (await session.exec(select(ModelPreheatCachedModel))).one()

    cached = asyncio.run(run())
    assert cached.revision == 2
    assert cached.manifest_digest == "e" * 64


def test_apply_scan_losing_selection_lock_before_commit_writes_nothing(engine):
    scan = InventoryScan(records=(record(digest="b" * 64),), generations=())
    service = ModelPreheatS3Inventory(engine, apply_batch_size=1)

    async def run():
        async with AsyncSession(engine) as session:
            job = await service.create_refresh_job(session, 7, 1)
            job.state = ModelPreheatInventoryJobStateEnum.RUNNING
            job.claim_token = "refresh-owner"
            job.scan_started_at = datetime.now(timezone.utc)
            job.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
            session.add(job)
            job_id = job.id
            await session.commit()

        original_renew = service._renew_claim

        async def steal_selection_lock(job_id_arg, token):
            renewed = await original_renew(job_id_arg, token)
            async with AsyncSession(engine) as session:
                await session.exec(
                    update(ModelPreheatInventorySelectionLock)
                    .where(ModelPreheatInventorySelectionLock.owner_token == token)
                    .values(owner_token="new-owner")
                )
                await session.commit()
            return renewed

        service._renew_claim = steal_selection_lock
        await service._apply_scan(job_id, scan, "refresh-owner")
        async with AsyncSession(engine) as session:
            return (await session.exec(select(ModelPreheatCachedModel))).all()

    assert asyncio.run(run()) == []


def test_apply_scan_renews_selection_locks_during_large_acquisition(
    engine, monkeypatch
):
    import gpustack.server.model_preheat_s3_inventory as inventory_module

    records = tuple(
        replace(record(digest=str(index) * 64), cache_key=str(index) * 64)
        for index in range(1, 4)
    )
    service = ModelPreheatS3Inventory(engine)
    renew_calls = 0
    original_acquire = service.acquire_selection_lock
    original_renew = service._renew_apply_selection_locks

    async def slow_acquire(*args, **kwargs):
        acquired = await original_acquire(*args, **kwargs)
        await asyncio.sleep(0.02)
        return acquired

    async def tracked_renew(*args, **kwargs):
        nonlocal renew_calls
        renew_calls += 1
        return await original_renew(*args, **kwargs)

    monkeypatch.setattr(
        inventory_module, "SELECTION_LOCK_LEASE", timedelta(milliseconds=90)
    )
    service.acquire_selection_lock = slow_acquire
    service._renew_apply_selection_locks = tracked_renew

    async def run():
        async with AsyncSession(engine) as session:
            job = await service.create_refresh_job(session, 7, 1)
            job.state = ModelPreheatInventoryJobStateEnum.RUNNING
            job.claim_token = "refresh-owner"
            job.scan_started_at = datetime.now(timezone.utc)
            job.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
            session.add(job)
            job_id = job.id
            await session.commit()
        await service._apply_scan(
            job_id,
            InventoryScan(records=records, generations=()),
            "refresh-owner",
        )
        async with AsyncSession(engine) as session:
            return await session.get(ModelPreheatInventoryJob, job_id)

    job = asyncio.run(run())
    assert renew_calls >= 2
    assert job.state == ModelPreheatInventoryJobStateEnum.READY


def test_duplicate_refresh_reuses_active_database_job(engine):
    store = FakeStore(InventoryScan(records=(), generations=()))
    service_a = ModelPreheatS3Inventory(engine, store_factory=lambda profile: store)
    service_b = ModelPreheatS3Inventory(engine, store_factory=lambda profile: store)

    async def create_both():
        async with AsyncSession(engine) as first:
            one = await service_a.create_refresh_job(first, 9, 2)
        async with AsyncSession(engine) as second:
            two = await service_b.create_refresh_job(second, 9, 2)
        return one.id, two.id

    first_id, second_id = asyncio.run(create_both())
    assert first_id == second_id


def test_selection_lock_allows_only_one_concurrent_publication_owner(engine):
    first = ModelPreheatS3Inventory(engine)
    second = ModelPreheatS3Inventory(engine)

    async def acquire_both():
        results = await asyncio.gather(
            first.acquire_selection_lock(7, "c" * 64, "owner-a", "publication"),
            second.acquire_selection_lock(7, "c" * 64, "owner-b", "publication"),
        )
        winner = "owner-a" if results[0] else "owner-b"
        await first.release_selection_lock(7, "c" * 64, winner)
        return results

    assert sorted(asyncio.run(acquire_both())) == [False, True]


def test_expired_running_job_is_reclaimed_after_process_restart(engine):
    store = FakeStore(InventoryScan(records=(record(),), generations=()))
    service = ModelPreheatS3Inventory(engine, store_factory=lambda profile: store)

    async def expire_and_reconcile():
        async with AsyncSession(engine) as session:
            job = await service.create_refresh_job(session, 7, 1)
            job.state = ModelPreheatInventoryJobStateEnum.RUNNING
            job.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            job_id = job.id
            session.add(job)
            await session.commit()
        await service.run_pending_jobs()
        async with AsyncSession(engine) as session:
            return await session.get(ModelPreheatInventoryJob, job_id)

    job = asyncio.run(expire_and_reconcile())
    assert job.state == ModelPreheatInventoryJobStateEnum.READY
    assert store.scan_calls == 1


def test_reclaimed_job_rejects_late_scan_from_expired_owner(engine):
    store = FakeStore(InventoryScan(records=(record(digest="b" * 64),), generations=()))
    service = ModelPreheatS3Inventory(engine, store_factory=lambda profile: store)

    async def run():
        async with AsyncSession(engine) as session:
            job = await service.create_refresh_job(session, 7, 1)
            job.state = ModelPreheatInventoryJobStateEnum.RUNNING
            job.claim_token = "old-owner"
            job.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            job_id = job.id
            session.add(job)
            await session.commit()
        await service.run_pending_jobs()
        await service._apply_scan(
            job_id,
            InventoryScan(records=(record(digest="e" * 64),), generations=()),
            "old-owner",
        )
        async with AsyncSession(engine) as session:
            item = (await session.exec(select(ModelPreheatCachedModel))).one()
            return item.manifest_digest

    assert asyncio.run(run()) == "b" * 64


def test_gc_client_failure_closes_async_job_without_leaking_error(engine):
    class BrokenFactory:
        def __call__(self, profile):
            raise OSError("plain-secret-key")

    service = ModelPreheatS3Inventory(engine, store_factory=BrokenFactory())

    async def run():
        async with AsyncSession(engine) as session:
            job = await service.create_gc_job(session, 7, 1)
            job_id = job.id
        await service.run_job(job_id)
        async with AsyncSession(engine) as session:
            return await session.get(ModelPreheatInventoryJob, job_id)

    job = asyncio.run(run())
    assert job.state == ModelPreheatInventoryJobStateEnum.ERROR
    assert job.error_code == "inventory_gc_failed"
    assert "secret" not in (job.error_message or "")


def test_gc_rechecks_ready_and_skips_when_reference_changed(engine):
    orphan = generation()
    store = FakeStore(InventoryScan(records=(), generations=(orphan,)))
    store.ready[orphan.ready_path] = ("different-generation", "b" * 64)
    store.objects[orphan.generation_path] = [f"{orphan.generation_path}/weights.bin"]
    asyncio.run(run_refresh(engine, store))

    async def age_and_gc():
        async with AsyncSession(engine) as session:
            item = (await session.exec(select(ModelPreheatInventoryGeneration))).one()
            item.first_seen_at = datetime.now(timezone.utc) - timedelta(days=3)
            item.orphaned_at = datetime.now(timezone.utc) - timedelta(days=3)
            session.add(item)
            await session.commit()
        service = ModelPreheatS3Inventory(engine, store_factory=lambda profile: store)
        return await service.run_gc(7, retention=timedelta(days=1))

    result = asyncio.run(age_and_gc())
    assert result.skipped == 1
    assert store.objects[orphan.generation_path]


def test_gc_skips_while_publication_holds_selection_lock(engine):
    orphan = replace(generation(), selection_key="c" * 64, cache_key="c" * 64)
    store = FakeStore(InventoryScan(records=(), generations=(orphan,)))
    store.ready[orphan.ready_path] = (None, orphan.ready_fingerprint)
    store.objects[orphan.generation_path] = [f"{orphan.generation_path}/weights.bin"]
    asyncio.run(run_refresh(engine, store))
    service = ModelPreheatS3Inventory(engine, store_factory=lambda profile: store)

    async def hold_and_gc():
        async with AsyncSession(engine) as session:
            item = (await session.exec(select(ModelPreheatInventoryGeneration))).one()
            item.orphaned_at = datetime.now(timezone.utc) - timedelta(days=3)
            session.add(item)
            await session.commit()
        assert await service.acquire_selection_lock(
            7, "c" * 64, "publication-owner", "publication"
        )
        try:
            return await service.run_gc(7, retention=timedelta(days=1))
        finally:
            await service.release_selection_lock(7, "c" * 64, "publication-owner")

    result = asyncio.run(hold_and_gc())
    assert result.skipped == 1
    assert store.deleted == []


@pytest.mark.parametrize(
    "ready_published",
    [False, True],
    ids=["long_upload_without_ready", "ready_published_before_delete"],
)
def test_gc_skips_old_generation_with_active_persistent_publication_marker(
    engine, ready_published
):
    generation_id = "preheat-11111111-1111-1111-1111-111111111111"
    orphan = replace(
        generation(generation_id), selection_key="c" * 64, cache_key="c" * 64
    )
    store = FakeStore(InventoryScan(records=(), generations=(orphan,)))
    store.ready[orphan.ready_path] = (
        orphan.generation_path if ready_published else None,
        orphan.ready_fingerprint,
    )
    store.objects[orphan.generation_path] = [f"{orphan.generation_path}/weights.bin"]
    asyncio.run(run_refresh(engine, store))

    async def seed_marker_and_gc():
        async with AsyncSession(engine) as session:
            task = ModelPreheatTask(
                id=91,
                source="huggingface",
                model_id="org/model",
                resolved_revision="f" * 40,
                include_patterns=[],
                exclude_patterns=[],
                selection_digest="d" * 64,
                cache_key="c" * 64,
                generation_id=generation_id,
                target_scope=ModelPreheatTargetScopeEnum.SEED_WORKER,
                target_worker_uuids=["worker"],
                target_worker_snapshot=[],
                s3_profile_id=7,
                s3_profile_config_version=1,
                s3_profile_snapshot_encrypted={},
                encryption_key_version="v1",
                s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
                execution_state="staging",
            )
            session.add(task)
            session.add(
                ModelPreheatPublicationMarker(
                    profile_id=7,
                    selection_key=task.cache_key,
                    generation_id=task.generation_id,
                    task_id=task.id,
                    parent_attempt=task.attempt,
                    profile_config_version=1,
                )
            )
            item = (await session.exec(select(ModelPreheatInventoryGeneration))).one()
            item.orphaned_at = datetime.now(timezone.utc) - timedelta(days=30)
            session.add(item)
            await session.commit()
        service = ModelPreheatS3Inventory(engine, store_factory=lambda profile: store)
        return await service.run_gc(7, retention=timedelta(days=1))

    result = asyncio.run(seed_marker_and_gc())
    assert result.skipped == 1
    assert store.deleted == []
    assert store.ready_reads == 0


def test_restarted_gc_keeps_old_marker_until_parent_and_seed_lease_are_safe(engine):
    generation_id = "preheat-22222222-2222-4222-8222-222222222222"
    orphan = replace(
        generation(generation_id), selection_key="c" * 64, cache_key="c" * 64
    )
    store = FakeStore(InventoryScan(records=(), generations=(orphan,)))
    store.ready[orphan.ready_path] = (None, orphan.ready_fingerprint)
    store.objects[orphan.generation_path] = [f"{orphan.generation_path}/weights.bin"]
    asyncio.run(run_refresh(engine, store))

    async def seed_old_marker():
        old = datetime.now(timezone.utc) - timedelta(days=30)
        async with AsyncSession(engine) as session:
            task = ModelPreheatTask(
                id=92,
                source="huggingface",
                model_id="org/model",
                resolved_revision="f" * 40,
                include_patterns=[],
                exclude_patterns=[],
                selection_digest="d" * 64,
                cache_key="c" * 64,
                generation_id=generation_id,
                target_scope=ModelPreheatTargetScopeEnum.SEED_WORKER,
                target_worker_uuids=["worker"],
                target_worker_snapshot=[],
                s3_profile_id=7,
                s3_profile_config_version=1,
                s3_profile_snapshot_encrypted={},
                encryption_key_version="v1",
                s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
                execution_state=ModelPreheatExecutionStateEnum.ERROR,
            )
            session.add(task)
            session.add(
                ModelPreheatPublicationMarker(
                    profile_id=7,
                    selection_key=task.cache_key,
                    generation_id=task.generation_id,
                    task_id=task.id,
                    parent_attempt=task.attempt,
                    profile_config_version=1,
                    created_at=old,
                    updated_at=old,
                )
            )
            session.add(
                ModelPreheatWorkerTask(
                    task_id=task.id,
                    parent_attempt=task.attempt,
                    worker_uuid="worker",
                    role=ModelPreheatWorkerTaskRoleEnum.SEED,
                    state=ModelPreheatWorkerTaskStateEnum.RUNNING,
                    lease_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                )
            )
            item = (await session.exec(select(ModelPreheatInventoryGeneration))).one()
            item.orphaned_at = old
            session.add(item)
            await session.flush()
            await session.exec(
                update(ModelPreheatPublicationMarker).values(
                    created_at=old, updated_at=old
                )
            )
            await session.commit()

    async def expire_seed_lease():
        async with AsyncSession(engine) as session:
            seed = (await session.exec(select(ModelPreheatWorkerTask))).one()
            seed.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            session.add(seed)
            await session.commit()

    asyncio.run(seed_old_marker())
    service = ModelPreheatS3Inventory(engine, store_factory=lambda profile: store)
    protected = asyncio.run(service.run_gc(7, retention=timedelta(days=1)))
    assert protected.skipped == 1
    assert store.deleted == []

    asyncio.run(expire_seed_lease())
    in_grace = asyncio.run(service.run_gc(7, retention=timedelta(days=1)))
    assert in_grace.skipped == 1
    assert store.deleted == []

    async def expire_termination_grace():
        old = datetime.now(timezone.utc) - timedelta(days=2)
        async with AsyncSession(engine) as session:
            marker = (await session.exec(select(ModelPreheatPublicationMarker))).one()
            assert marker.terminated_at is not None
            await session.exec(
                update(ModelPreheatPublicationMarker)
                .where(ModelPreheatPublicationMarker.id == marker.id)
                .values(terminated_at=old, updated_at=old)
            )
            await session.commit()

    asyncio.run(expire_termination_grace())
    recovered = asyncio.run(service.run_gc(7, retention=timedelta(days=1)))
    assert recovered.deleted == 1
    assert store.objects[orphan.generation_path] == []


def test_retry_same_generation_rebinds_marker_and_restarts_cancel_grace(engine):
    generation_id = "preheat-33333333-3333-4333-8333-333333333333"
    orphan = replace(
        generation(generation_id), selection_key="c" * 64, cache_key="c" * 64
    )
    store = FakeStore(InventoryScan(records=(), generations=(orphan,)))
    store.ready[orphan.ready_path] = (None, orphan.ready_fingerprint)
    store.objects[orphan.generation_path] = [f"{orphan.generation_path}/weights.bin"]
    asyncio.run(run_refresh(engine, store))
    marker_updates = []

    def capture_marker_update(
        connection, clauseelement, multiparams, params, execution_options
    ):
        del connection, multiparams, params, execution_options
        if (
            getattr(getattr(clauseelement, "table", None), "name", None)
            == ModelPreheatPublicationMarker.__tablename__
            and getattr(clauseelement, "whereclause", None) is not None
        ):
            marker_updates.append(str(clauseelement.whereclause))

    event.listen(engine.sync_engine, "before_execute", capture_marker_update)

    async def retry_cancel_and_gc():
        old = datetime.now(timezone.utc) - timedelta(days=2)
        async with AsyncSession(engine) as session:
            task = ModelPreheatTask(
                id=93,
                source="huggingface",
                model_id="org/model",
                resolved_revision="f" * 40,
                include_patterns=[],
                exclude_patterns=[],
                selection_digest="d" * 64,
                cache_key="c" * 64,
                generation_id=generation_id,
                target_scope=ModelPreheatTargetScopeEnum.SEED_WORKER,
                target_worker_uuids=["worker"],
                target_worker_snapshot=[],
                s3_profile_id=7,
                s3_profile_config_version=2,
                s3_profile_snapshot_encrypted={},
                encryption_key_version="v1",
                s3_backfill_policy=ModelPreheatBackfillPolicyEnum.WHEN_MISSING,
                attempt=2,
                execution_state=ModelPreheatExecutionStateEnum.PENDING,
            )
            session.add(task)
            session.add(
                ModelPreheatPublicationMarker(
                    profile_id=7,
                    selection_key=task.cache_key,
                    generation_id=task.generation_id,
                    task_id=task.id,
                    parent_attempt=1,
                    profile_config_version=1,
                    terminated_at=old,
                    created_at=old,
                    updated_at=old,
                )
            )
            item = (await session.exec(select(ModelPreheatInventoryGeneration))).one()
            item.orphaned_at = old
            session.add(item)
            await session.commit()
            await session.refresh(task)
            task_snapshot = task.model_copy(deep=True)

        service = ModelPreheatS3Inventory(engine, store_factory=lambda profile: store)
        ensured = await service.ensure_publication_marker(task_snapshot)
        assert ensured, marker_updates
        async with AsyncSession(engine) as session:
            rebound = (await session.exec(select(ModelPreheatPublicationMarker))).one()
            assert rebound.parent_attempt == 2
            assert rebound.profile_config_version == 2
            assert rebound.terminated_at is None
            task = await session.get(ModelPreheatTask, task_snapshot.id)
            task.execution_state = ModelPreheatExecutionStateEnum.CANCELED
            session.add(task)
            await session.flush()
            assert await service.terminate_publication_marker(session, task)
            await session.commit()
            terminated_at = (
                await session.exec(select(ModelPreheatPublicationMarker.terminated_at))
            ).one()

        result = await service.run_gc(7, retention=timedelta(days=1))
        return old, terminated_at, result

    try:
        old, terminated_at, result = asyncio.run(retry_cancel_and_gc())
    finally:
        event.remove(engine.sync_engine, "before_execute", capture_marker_update)

    assert any(
        "task_id" in where
        and "parent_attempt" in where
        and "profile_config_version" in where
        and "terminated_at" in where
        for where in marker_updates
    )
    assert terminated_at > old
    assert result.skipped == 1
    assert store.deleted == []
    assert store.objects[orphan.generation_path]


def test_gc_deletes_old_generation_while_unchanged_ready_references_current(engine):
    current_path = "model-cache/v1/x/generations/current"
    orphan = generation()
    orphan = ScannedGeneration(
        generation_path=orphan.generation_path,
        ready_path=orphan.ready_path,
        ready_fingerprint=orphan.ready_fingerprint,
        ready_generation_path=current_path,
        referenced=False,
    )
    store = FakeStore(InventoryScan(records=(), generations=(orphan,)))
    store.ready[orphan.ready_path] = (current_path, orphan.ready_fingerprint)
    store.objects[orphan.generation_path] = [f"{orphan.generation_path}/weights.bin"]
    asyncio.run(run_refresh(engine, store))

    async def age_and_gc():
        async with AsyncSession(engine) as session:
            item = (await session.exec(select(ModelPreheatInventoryGeneration))).one()
            item.first_seen_at = datetime.now(timezone.utc) - timedelta(days=3)
            item.orphaned_at = datetime.now(timezone.utc) - timedelta(days=3)
            session.add(item)
            await session.commit()
        service = ModelPreheatS3Inventory(engine, store_factory=lambda profile: store)
        return await service.run_gc(7, retention=timedelta(days=1))

    result = asyncio.run(age_and_gc())
    assert result.deleted == 1
    assert store.objects[orphan.generation_path] == []


def test_gc_is_per_object_idempotent_and_failure_is_closed(engine):
    orphan = generation(fingerprint="c" * 64)
    store = FakeStore(InventoryScan(records=(), generations=(orphan,)))
    store.ready[orphan.ready_path] = (None, "c" * 64)
    objects = [f"{orphan.generation_path}/a", f"{orphan.generation_path}/b"]
    store.objects[orphan.generation_path] = objects.copy()
    store.delete_failures.add(objects[1])
    asyncio.run(run_refresh(engine, store))

    async def age():
        async with AsyncSession(engine) as session:
            item = (await session.exec(select(ModelPreheatInventoryGeneration))).one()
            item.first_seen_at = datetime.now(timezone.utc) - timedelta(days=3)
            item.orphaned_at = datetime.now(timezone.utc) - timedelta(days=3)
            session.add(item)
            await session.commit()

    asyncio.run(age())
    service = ModelPreheatS3Inventory(engine, store_factory=lambda profile: store)
    failed = asyncio.run(service.run_gc(7, retention=timedelta(days=1)))
    assert failed.failed == 1
    assert store.objects[orphan.generation_path] == [objects[1]]
    store.delete_failures.clear()
    completed = asyncio.run(service.run_gc(7, retention=timedelta(days=1)))
    assert completed.deleted == 1
    assert store.objects[orphan.generation_path] == []


def test_gc_retention_starts_when_generation_becomes_orphan(engine):
    item = generation(name="current", referenced=True)
    item = ScannedGeneration(
        generation_path=item.generation_path,
        ready_path=item.ready_path,
        ready_fingerprint=item.ready_fingerprint,
        ready_generation_path=item.generation_path,
        referenced=True,
    )
    store = FakeStore(InventoryScan(records=(), generations=(item,)))
    store.ready[item.ready_path] = (item.generation_path, item.ready_fingerprint)
    store.objects[item.generation_path] = [f"{item.generation_path}/weights.bin"]
    asyncio.run(run_refresh(engine, store))

    async def age_generation():
        async with AsyncSession(engine) as session:
            stored = (await session.exec(select(ModelPreheatInventoryGeneration))).one()
            stored.first_seen_at = datetime.now(timezone.utc) - timedelta(days=3)
            session.add(stored)
            await session.commit()

    asyncio.run(age_generation())

    replacement = "model-cache/v1/x/generations/replacement"
    store.ready[item.ready_path] = (replacement, "b" * 64)
    store.scan_result = InventoryScan(
        records=(),
        generations=(
            ScannedGeneration(
                generation_path=item.generation_path,
                ready_path=item.ready_path,
                ready_fingerprint="b" * 64,
                ready_generation_path=replacement,
                referenced=False,
            ),
        ),
    )
    asyncio.run(run_refresh(engine, store))
    service = ModelPreheatS3Inventory(engine, store_factory=lambda profile: store)
    result = asyncio.run(service.run_gc(7, retention=timedelta(days=1)))
    assert result.deleted == 0
    assert store.objects[item.generation_path]


def test_gc_object_limit_fails_closed_before_any_delete(engine):
    orphan = generation(fingerprint="c" * 64)
    store = FakeStore(InventoryScan(records=(), generations=(orphan,)))
    store.ready[orphan.ready_path] = (None, "c" * 64)
    store.objects[orphan.generation_path] = [
        f"{orphan.generation_path}/{index}" for index in range(3)
    ]
    asyncio.run(run_refresh(engine, store))

    async def age_and_gc():
        async with AsyncSession(engine) as session:
            item = (await session.exec(select(ModelPreheatInventoryGeneration))).one()
            item.orphaned_at = datetime.now(timezone.utc) - timedelta(days=3)
            session.add(item)
            await session.commit()
        service = ModelPreheatS3Inventory(
            engine, store_factory=lambda profile: store, max_gc_objects=2
        )
        result = await service.run_gc(7, retention=timedelta(days=1))
        async with AsyncSession(engine) as session:
            item = (await session.exec(select(ModelPreheatInventoryGeneration))).one()
            return result, item.state

    result, state = asyncio.run(age_and_gc())
    assert result.failed == 1
    assert state == ModelPreheatInventoryGenerationStateEnum.ERROR
    assert store.deleted == []
