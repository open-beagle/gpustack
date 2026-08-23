import asyncio
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path

from sqlalchemy import (
    Column,
    Enum as SAEnum,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateIndex, CreateTable
import pytest
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3ConnectivityStateEnum,
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
)
from gpustack.schemas.model_preheats import (
    ModelPreheatConnectivityCheckStateEnum,
    ModelPreheatS3ConnectivityCheck,
    ModelPreheatWorkerTask,
    ModelPreheatWorkerTaskRoleEnum,
    ModelPreheatWorkerTaskStateEnum,
)
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.config.config import Config
from gpustack.server.model_preheat_connectivity import (
    DEFAULT_CONNECTIVITY_TTL,
    aggregate_connectivity_check,
    create_or_reuse_connectivity_check,
    current_ready_workers,
    mark_profile_stale_if_expired,
    worker_network_identity_changed,
)
from gpustack.server import model_preheat_connectivity


async def _tables(engine, action):
    async with engine.begin() as connection:
        await connection.run_sync(action)


def _profile():
    return ModelPreheatS3Profile(
        name="profile",
        endpoint="https://s3.example.com",
        bucket="models",
        access_key_encrypted={"ciphertext": "encrypted"},
        secret_key_encrypted={"ciphertext": "encrypted"},
        encryption_key_version="v1",
    )


def _worker(name, uuid, state=WorkerStateEnum.READY):
    return Worker(
        name=name,
        hostname=name,
        ip="127.0.0.1",
        port=10150,
        worker_uuid=uuid,
        state=state,
    )


def test_connectivity_check_snapshots_all_ready_workers_and_reuses_running_check(
    tmp_path,
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'connectivity.db'}", poolclass=NullPool
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                profile = _profile()
                session.add_all(
                    [
                        profile,
                        _worker("embedded", "embedded-uuid"),
                        _worker("remote", "remote-uuid"),
                    ]
                )
                await session.commit()
                check = await create_or_reuse_connectivity_check(session, profile)
                assert check is not None
                assert check.state == ModelPreheatConnectivityCheckStateEnum.RUNNING
                assert check.target_worker_uuids == ["embedded-uuid", "remote-uuid"]
                tasks = (await session.exec(select(ModelPreheatWorkerTask))).all()
                assert {(task.worker_uuid, task.role) for task in tasks} == {
                    (
                        "embedded-uuid",
                        ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK,
                    ),
                    ("remote-uuid", ModelPreheatWorkerTaskRoleEnum.CONNECTIVITY_CHECK),
                }
                assert (
                    profile.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.CHECKING
                )
                assert profile.last_connectivity_check_id == check.id
                assert (
                    await create_or_reuse_connectivity_check(session, profile) is check
                )
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_connectivity_check_reuses_same_explicit_target_snapshot_only(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'targeted-check.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                profile = _profile()
                session.add_all(
                    [
                        profile,
                        _worker("worker-a", "worker-a-uuid"),
                        _worker("worker-b", "worker-b-uuid"),
                    ]
                )
                await session.commit()

                check_b = await create_or_reuse_connectivity_check(
                    session, profile, target_worker_uuids={"worker-b-uuid"}
                )
                repeated_b = await create_or_reuse_connectivity_check(
                    session, profile, target_worker_uuids=["worker-b-uuid"]
                )
                check_b_id = check_b.id
                check_b_targets = check_b.target_worker_uuids
                assert repeated_b.id == check_b_id
                check_a = await create_or_reuse_connectivity_check(
                    session, profile, target_worker_uuids=["worker-a-uuid"]
                )

                assert check_b_targets == ["worker-b-uuid"]
                assert check_a.target_worker_uuids == ["worker-a-uuid"]
                assert check_a.id != check_b_id
                tasks = (await session.exec(select(ModelPreheatWorkerTask))).all()
                assert {
                    (task.connectivity_check_id, task.worker_uuid) for task in tasks
                } == {
                    (check_a.id, "worker-a-uuid"),
                    (check_b_id, "worker-b-uuid"),
                }
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_worker_lifecycle_check_does_not_replace_profile_presentation_pointer(
    tmp_path,
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'lifecycle-pointer.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine, expire_on_commit=False) as session:
                profile = _profile()
                session.add_all(
                    [
                        profile,
                        _worker("worker-a", "worker-a-uuid"),
                        _worker("worker-b", "worker-b-uuid"),
                    ]
                )
                await session.commit()

                full_check = await create_or_reuse_connectivity_check(session, profile)
                assert profile.last_connectivity_check_id == full_check.id

                lifecycle_check = await create_or_reuse_connectivity_check(
                    session,
                    profile,
                    target_worker_uuids=["worker-a-uuid"],
                    scope_discriminator="worker-lifecycle",
                    update_profile_pointer=False,
                )
                assert lifecycle_check.id != full_check.id
                assert profile.last_connectivity_check_id == full_check.id

                repeated_lifecycle_check = await create_or_reuse_connectivity_check(
                    session,
                    profile,
                    target_worker_uuids=["worker-a-uuid"],
                    scope_discriminator="worker-lifecycle",
                    update_profile_pointer=False,
                )
                assert repeated_lifecycle_check.id == lifecycle_check.id
                assert profile.last_connectivity_check_id == full_check.id

                profile.last_connectivity_check_id = None
                session.add(profile)
                await session.commit()
                repeated_full_check = await create_or_reuse_connectivity_check(
                    session, profile
                )
                assert repeated_full_check.id == full_check.id
                assert profile.last_connectivity_check_id == full_check.id
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_concurrent_connectivity_check_creation_reuses_database_winner(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'concurrent-connectivity.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                profile = _profile()
                session.add_all(
                    [
                        profile,
                        _worker("worker-a", "worker-a-uuid"),
                        _worker("worker-b", "worker-b-uuid"),
                    ]
                )
                await session.flush()
                profile_id = profile.id
                await session.commit()

            gate = asyncio.Event()
            gate_lock = asyncio.Lock()
            arrived = 0

            class FlushBarrierSession(AsyncSession):
                async def flush(self, objects=None):
                    nonlocal arrived
                    if any(
                        isinstance(item, ModelPreheatS3ConnectivityCheck)
                        for item in self.new
                    ):
                        async with gate_lock:
                            arrived += 1
                            if arrived == 2:
                                gate.set()
                        await asyncio.wait_for(gate.wait(), timeout=5)
                    return await super().flush(objects)

            async def create_check():
                async with FlushBarrierSession(engine) as session:
                    profile = await session.get(ModelPreheatS3Profile, profile_id)
                    check = await create_or_reuse_connectivity_check(session, profile)
                    profile_snapshot = (
                        profile.last_connectivity_check_id,
                        profile.connectivity_state,
                    )
                    return check, profile_snapshot

            first, second = await asyncio.gather(create_check(), create_check())
            first_check, first_profile = first
            second_check, second_profile = second

            async with AsyncSession(engine) as session:
                check_count = await session.scalar(
                    select(func.count()).select_from(ModelPreheatS3ConnectivityCheck)
                )
                tasks = (await session.exec(select(ModelPreheatWorkerTask))).all()
                stored_profile = await session.get(ModelPreheatS3Profile, profile_id)
                stored_profile_snapshot = (
                    stored_profile.last_connectivity_check_id,
                    stored_profile.connectivity_state,
                )
            assert first_check.id == second_check.id
            assert first_profile == (
                first_check.id,
                ModelPreheatS3ConnectivityStateEnum.CHECKING,
            )
            assert second_profile == (
                first_check.id,
                ModelPreheatS3ConnectivityStateEnum.CHECKING,
            )
            assert stored_profile_snapshot == first_profile
            assert check_count == 1
            assert {
                (task.worker_uuid, task.connectivity_check_id) for task in tasks
            } == {
                ("worker-a-uuid", first_check.id),
                ("worker-b-uuid", first_check.id),
            }
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_concurrent_loser_finds_winner_after_active_key_is_cleared(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'terminal-winner.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                profile = _profile()
                session.add_all([profile, _worker("worker", "worker-uuid")])
                await session.flush()
                profile_id = profile.id
                await session.commit()

            gate = asyncio.Event()
            gate_lock = asyncio.Lock()
            arrived = 0
            terminalized = False

            class TerminalWinnerSession(AsyncSession):
                async def flush(self, objects=None):
                    nonlocal arrived
                    if any(
                        isinstance(item, ModelPreheatS3ConnectivityCheck)
                        for item in self.new
                    ):
                        async with gate_lock:
                            arrived += 1
                            if arrived == 2:
                                gate.set()
                        await asyncio.wait_for(gate.wait(), timeout=5)
                    return await super().flush(objects)

                async def rollback(self):
                    nonlocal terminalized
                    await super().rollback()
                    if terminalized:
                        return
                    terminalized = True
                    async with AsyncSession(engine) as terminal_session:
                        winner = (
                            await terminal_session.exec(
                                select(ModelPreheatS3ConnectivityCheck)
                            )
                        ).one()
                        task = (
                            await terminal_session.exec(
                                select(ModelPreheatWorkerTask).where(
                                    ModelPreheatWorkerTask.connectivity_check_id
                                    == winner.id
                                )
                            )
                        ).one()
                        task.state = ModelPreheatWorkerTaskStateEnum.READY
                        terminal_session.add(task)
                        winner_id = winner.id
                        await terminal_session.commit()
                        await aggregate_connectivity_check(terminal_session, winner_id)

            async def create_check():
                async with TerminalWinnerSession(engine) as session:
                    profile = await session.get(ModelPreheatS3Profile, profile_id)
                    return await create_or_reuse_connectivity_check(session, profile)

            first, second = await asyncio.gather(create_check(), create_check())
            assert first.id == second.id
            assert terminalized
            async with AsyncSession(engine) as session:
                checks = (
                    await session.exec(select(ModelPreheatS3ConnectivityCheck))
                ).all()
                assert len(checks) == 1
                assert checks[0].active_key is None
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_connectivity_active_key_uses_portable_nullable_unique_constraint():
    constraints = ModelPreheatS3ConnectivityCheck.__table__.constraints
    assert any(
        isinstance(constraint, UniqueConstraint)
        and constraint.name == "uix_preheat_connectivity_active"
        and [column.name for column in constraint.columns] == ["active_key"]
        for constraint in constraints
    )
    assert ModelPreheatS3ConnectivityCheck.__table__.c.active_key.nullable is True


def test_model_preheat_core_migration_compiles_for_mysql(monkeypatch):
    migration_path = Path(
        "gpustack/migrations/versions/"
        "2026_08_10_1000-f6a7b8c9d0e1_add_model_preheat_core.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_model_preheat_core_migration", migration_path
    )
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    metadata = MetaData()
    Table("users", metadata, Column("id", Integer, primary_key=True))
    created_tables = []
    created_indexes = []

    def capture_table(name, *elements, **kwargs):
        table = Table(name, metadata, *elements, **kwargs)
        created_tables.append(table)
        return table

    def capture_index(name, table_name, columns, **kwargs):
        table = metadata.tables[table_name]
        index = Index(name, *(table.c[column] for column in columns), **kwargs)
        created_indexes.append(index)
        return index

    monkeypatch.setattr(migration.op, "create_table", capture_table)
    monkeypatch.setattr(migration.op, "create_index", capture_index)
    migration.upgrade()

    assert created_tables
    for table in created_tables:
        orm_table = SQLModel.metadata.tables[table.name]
        string_columns = [
            column for column in table.columns if type(column.type) is String
        ]
        assert all(column.type.length == 255 for column in string_columns)
        for column in string_columns:
            orm_type = orm_table.c[column.name].type
            if isinstance(orm_type, SAEnum):
                assert column.type.length >= orm_type.length
            else:
                orm_mysql_type = orm_type.dialect_impl(mysql.dialect())
                assert column.type.length == orm_mysql_type.length
        ddl = str(CreateTable(table).compile(dialect=mysql.dialect()))
        assert f"CREATE TABLE {table.name}" in ddl
    assert not any(
        index.name == "ix_preheat_worker_uuid_state" for index in created_indexes
    )
    for index in created_indexes:
        ddl = str(CreateIndex(index).compile(dialect=mysql.dialect()))
        assert f"CREATE INDEX {index.name}" in ddl

    successor = Path(
        "gpustack/migrations/versions/"
        "2026_08_11_2100-d0e1f2a3b4c5_add_preheat_worker_identity.py"
    ).read_text()
    assert '"ix_preheat_worker_uuid_state"' in successor


def test_terminal_connectivity_check_retains_stable_scope_key(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'terminal-scope.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                profile = _profile()
                session.add_all([profile, _worker("worker", "worker-uuid")])
                await session.commit()
                check = await create_or_reuse_connectivity_check(session, profile)
                check_id = check.id
                active_key = check.active_key
                task = (await session.exec(select(ModelPreheatWorkerTask))).one()
                task.state = ModelPreheatWorkerTaskStateEnum.READY
                session.add(task)
                await session.commit()

                check = await aggregate_connectivity_check(session, check_id)
                assert check.active_key is None
                assert check.scope_key == active_key
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_ready_worker_snapshot_uses_latest_registration_for_duplicate_uuid(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'duplicate-worker.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                profile = _profile()
                old_registration = _worker("embedded-old", "embedded-uuid")
                session.add_all([profile, old_registration])
                await session.flush()
                new_registration = _worker("embedded-new", "embedded-uuid")
                other_worker = _worker("remote", "remote-uuid")
                session.add_all([new_registration, other_worker])
                await session.commit()

                ready_workers = await current_ready_workers(session)
                new_registration_id = new_registration.id
                other_worker_id = other_worker.id
                assert [
                    (worker.worker_uuid, worker.id) for worker in ready_workers
                ] == [
                    ("embedded-uuid", new_registration_id),
                    ("remote-uuid", other_worker_id),
                ]

                check = await create_or_reuse_connectivity_check(session, profile)
                tasks = (await session.exec(select(ModelPreheatWorkerTask))).all()
                assert check.target_worker_uuids == ["embedded-uuid", "remote-uuid"]
                assert [(task.worker_uuid, task.worker_id) for task in tasks] == [
                    ("embedded-uuid", new_registration_id),
                    ("remote-uuid", other_worker_id),
                ]
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_latest_not_ready_registration_hides_old_ready_worker(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'latest-not-ready.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                session.add(_worker("worker-old", "worker-uuid"))
                await session.flush()
                session.add(
                    _worker(
                        "worker-new",
                        "worker-uuid",
                        WorkerStateEnum.NOT_READY,
                    )
                )
                await session.commit()

                assert await current_ready_workers(session) == []
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "old_result_state",
    [
        ModelPreheatWorkerTaskStateEnum.READY,
        ModelPreheatWorkerTaskStateEnum.ERROR,
    ],
)
def test_old_registration_terminal_result_is_invalid_after_reregistration(
    tmp_path, old_result_state
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'reregister-result.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                profile = _profile()
                old_worker = _worker("worker-old", "worker-uuid")
                session.add_all([profile, old_worker])
                await session.commit()
                check = await create_or_reuse_connectivity_check(session, profile)
                task = (
                    await session.exec(
                        select(ModelPreheatWorkerTask).where(
                            ModelPreheatWorkerTask.connectivity_check_id == check.id
                        )
                    )
                ).one()
                check_id = check.id
                old_worker_id = task.worker_id
                task.state = old_result_state
                session.add(task)
                await session.commit()

                new_worker = _worker("worker-new", "worker-uuid")
                session.add(new_worker)
                await session.flush()
                new_worker_id = new_worker.id
                await session.commit()
                assert new_worker_id != old_worker_id

                check = await aggregate_connectivity_check(session, check_id)
                assert check.success_count == 0
                assert check.failed_count == 0
                assert check.not_checked_count == 1
                assert check.state == ModelPreheatConnectivityCheckStateEnum.PARTIAL
                assert (
                    profile.connectivity_state
                    != ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                )
                assert (
                    profile.lifecycle_state
                    == ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE
                )
                assert profile.ever_used_at is None
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("first_worker_state", "expected_profile_state"),
    [
        (
            ModelPreheatWorkerTaskStateEnum.READY,
            ModelPreheatS3ConnectivityStateEnum.AVAILABLE,
        ),
        (
            ModelPreheatWorkerTaskStateEnum.ERROR,
            ModelPreheatS3ConnectivityStateEnum.PARTIAL,
        ),
    ],
)
def test_incremental_check_aggregates_latest_results_for_all_current_workers(
    tmp_path, first_worker_state, expected_profile_state
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / f'incremental-{first_worker_state}.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                profile = _profile()
                session.add_all([profile, _worker("worker-a", "worker-a-uuid")])
                await session.commit()
                check_a = await create_or_reuse_connectivity_check(
                    session, profile, target_worker_uuids=["worker-a-uuid"]
                )
                check_a_id = check_a.id
                task_a = (
                    await session.exec(
                        select(ModelPreheatWorkerTask).where(
                            ModelPreheatWorkerTask.connectivity_check_id == check_a_id
                        )
                    )
                ).one()
                task_a.state = first_worker_state
                session.add(task_a)
                await session.commit()
                check_a = await aggregate_connectivity_check(session, check_a_id)
                check_a_finished_at = check_a.finished_at

                session.add(_worker("worker-b", "worker-b-uuid"))
                await session.commit()
                check_b = await create_or_reuse_connectivity_check(
                    session, profile, target_worker_uuids=["worker-b-uuid"]
                )
                check_b_id = check_b.id
                task_b = (
                    await session.exec(
                        select(ModelPreheatWorkerTask).where(
                            ModelPreheatWorkerTask.connectivity_check_id == check_b_id
                        )
                    )
                ).one()
                task_b.state = ModelPreheatWorkerTaskStateEnum.READY
                session.add(task_b)
                await session.commit()
                await aggregate_connectivity_check(session, check_b_id)

                await session.refresh(profile)
                assert profile.connectivity_state == expected_profile_state
                assert profile.last_connectivity_check_id == check_b_id
                if first_worker_state == ModelPreheatWorkerTaskStateEnum.READY:
                    assert profile.last_connectivity_checked_at == check_a_finished_at
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_incremental_worker_pending_or_failed_never_makes_profile_available(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'incremental-incomplete.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                profile = _profile()
                session.add_all([profile, _worker("worker-a", "worker-a-uuid")])
                await session.commit()
                check_a = await create_or_reuse_connectivity_check(
                    session, profile, target_worker_uuids=["worker-a-uuid"]
                )
                check_a_id = check_a.id
                task_a = (await session.exec(select(ModelPreheatWorkerTask))).one()
                task_a.state = ModelPreheatWorkerTaskStateEnum.READY
                session.add(task_a)
                await session.commit()
                await aggregate_connectivity_check(session, check_a_id)

                session.add(_worker("worker-b", "worker-b-uuid"))
                await session.commit()
                check_b = await create_or_reuse_connectivity_check(
                    session, profile, target_worker_uuids=["worker-b-uuid"]
                )
                check_b_id = check_b.id
                await aggregate_connectivity_check(session, check_b_id)
                await session.refresh(profile)
                assert (
                    profile.connectivity_state
                    != ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                )

                task_b = (
                    await session.exec(
                        select(ModelPreheatWorkerTask).where(
                            ModelPreheatWorkerTask.connectivity_check_id == check_b_id
                        )
                    )
                ).one()
                task_b.state = ModelPreheatWorkerTaskStateEnum.ERROR
                session.add(task_b)
                await session.commit()
                await aggregate_connectivity_check(session, check_b_id)
                await session.refresh(profile)
                assert (
                    profile.connectivity_state
                    != ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                )
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_older_incremental_check_can_aggregate_without_replacing_newer_pointer(
    tmp_path,
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'incremental-out-of-order.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as setup_session:
                profile = _profile()
                setup_session.add_all(
                    [
                        profile,
                        _worker("worker-a", "worker-a-uuid"),
                        _worker("worker-b", "worker-b-uuid"),
                    ]
                )
                await setup_session.commit()
                check_a = await create_or_reuse_connectivity_check(
                    setup_session,
                    profile,
                    target_worker_uuids=["worker-a-uuid"],
                )
                profile_id = profile.id
                check_a_id = check_a.id

            async with AsyncSession(engine) as newer_session:
                profile = await newer_session.get(ModelPreheatS3Profile, profile_id)
                check_b = await create_or_reuse_connectivity_check(
                    newer_session,
                    profile,
                    target_worker_uuids=["worker-b-uuid"],
                )
                check_b_id = check_b.id
                task_b = (
                    await newer_session.exec(
                        select(ModelPreheatWorkerTask).where(
                            ModelPreheatWorkerTask.connectivity_check_id == check_b_id
                        )
                    )
                ).one()
                task_b.state = ModelPreheatWorkerTaskStateEnum.READY
                newer_session.add(task_b)
                await newer_session.commit()
                await aggregate_connectivity_check(newer_session, check_b_id)

            async with AsyncSession(engine) as older_session:
                task_a = (
                    await older_session.exec(
                        select(ModelPreheatWorkerTask).where(
                            ModelPreheatWorkerTask.connectivity_check_id == check_a_id
                        )
                    )
                ).one()
                task_a.state = ModelPreheatWorkerTaskStateEnum.READY
                older_session.add(task_a)
                await older_session.commit()
                await aggregate_connectivity_check(older_session, check_a_id)

            async with AsyncSession(engine) as verify_session:
                stored = await verify_session.get(ModelPreheatS3Profile, profile_id)
                assert stored.last_connectivity_check_id == check_b_id
                assert (
                    stored.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                )
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_connectivity_aggregation_does_not_overwrite_newer_profile_version_or_available_with_offline_worker(
    tmp_path,
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'aggregate.db'}", poolclass=NullPool
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                profile = _profile()
                worker = _worker("worker", "worker-uuid")
                session.add_all([profile, worker])
                await session.commit()
                check = await create_or_reuse_connectivity_check(session, profile)
                check_id = check.id
                assert check.active_key == check.scope_key
                assert len(check.active_key) == 64
                task = (await session.exec(select(ModelPreheatWorkerTask))).one()
                task.state = ModelPreheatWorkerTaskStateEnum.READY
                session.add(task)
                await session.commit()
                await aggregate_connectivity_check(session, check_id)
                assert check.active_key is None
                assert (
                    profile.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                )

                profile.config_version = 2
                profile.connectivity_state = (
                    ModelPreheatS3ConnectivityStateEnum.CHECKING
                )
                session.add(profile)
                await session.commit()
                await aggregate_connectivity_check(session, check_id)
                assert (
                    profile.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.CHECKING
                )

                profile.config_version = 1
                worker.state = WorkerStateEnum.NOT_READY
                session.add_all([profile, worker])
                await session.commit()
                await aggregate_connectivity_check(session, check_id)
                assert check.not_checked_count == 1
                assert check.state == ModelPreheatConnectivityCheckStateEnum.PARTIAL
                assert (
                    profile.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.NO_WORKERS
                )
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_profile_aggregation_tracks_offline_registered_workers_and_removes_deleted_workers(
    tmp_path,
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'offline-registered.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                profile = _profile()
                ready_worker = _worker("ready", "ready-uuid")
                offline_worker = _worker("offline", "offline-uuid")
                session.add_all([profile, ready_worker, offline_worker])
                await session.flush()
                ready_worker_id = ready_worker.id
                offline_worker_id = offline_worker.id
                await session.commit()
                check = await create_or_reuse_connectivity_check(session, profile)
                check_id = check.id
                tasks = (
                    await session.exec(
                        select(ModelPreheatWorkerTask).where(
                            ModelPreheatWorkerTask.connectivity_check_id == check_id
                        )
                    )
                ).all()
                for task in tasks:
                    if task.worker_uuid == "ready-uuid":
                        task.state = ModelPreheatWorkerTaskStateEnum.READY
                offline_worker = await session.get(Worker, offline_worker_id)
                offline_worker.state = WorkerStateEnum.NOT_READY
                session.add_all([*tasks, offline_worker])
                await session.commit()

                await aggregate_connectivity_check(session, check_id)
                assert (
                    profile.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                )

                await session.delete(offline_worker)
                await session.commit()
                await aggregate_connectivity_check(session, check_id)
                assert (
                    profile.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                )

                ready_worker = await session.get(Worker, ready_worker_id)
                await session.delete(ready_worker)
                await session.commit()
                await aggregate_connectivity_check(session, check_id)
                assert (
                    profile.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.NO_WORKERS
                )
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_profile_aggregation_recomputes_after_cross_session_cas_miss(
    tmp_path, monkeypatch
):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'aggregate-state-cas.db'}",
            poolclass=NullPool,
        )
        async with engine.begin() as connection:
            await connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        await _tables(engine, SQLModel.metadata.create_all)
        computed = asyncio.Event()
        resume = asyncio.Event()
        original_aggregate = (
            model_preheat_connectivity._aggregate_profile_connectivity_state
        )
        invocation_count = 0

        async def aggregate_with_barrier(*args, **kwargs):
            nonlocal invocation_count
            result = await original_aggregate(*args, **kwargs)
            invocation_count += 1
            if invocation_count == 1:
                computed.set()
                await asyncio.wait_for(resume.wait(), timeout=5)
            return result

        monkeypatch.setattr(
            model_preheat_connectivity,
            "_aggregate_profile_connectivity_state",
            aggregate_with_barrier,
        )
        try:
            async with AsyncSession(engine) as setup_session:
                profile = _profile()
                worker = _worker("worker", "worker-uuid")
                setup_session.add_all([profile, worker])
                await setup_session.commit()
                check = await create_or_reuse_connectivity_check(setup_session, profile)
                profile_id = profile.id
                check_id = check.id
                task = (
                    await setup_session.exec(
                        select(ModelPreheatWorkerTask).where(
                            ModelPreheatWorkerTask.connectivity_check_id == check_id
                        )
                    )
                ).one()
                finished_at = datetime.now(timezone.utc) - timedelta(seconds=1)
                task.state = ModelPreheatWorkerTaskStateEnum.ERROR
                check.state = ModelPreheatConnectivityCheckStateEnum.UNAVAILABLE
                check.failed_count = 1
                check.active_key = None
                check.finished_at = finished_at
                profile.connectivity_state = (
                    ModelPreheatS3ConnectivityStateEnum.UNAVAILABLE
                )
                profile.last_connectivity_checked_at = finished_at
                setup_session.add_all([profile, check, task])
                await setup_session.commit()

            async with AsyncSession(engine) as old_session:
                old_aggregate = asyncio.create_task(
                    aggregate_connectivity_check(old_session, check_id)
                )
                await asyncio.wait_for(computed.wait(), timeout=5)

                async with AsyncSession(engine) as winner_session:
                    winner_profile = await winner_session.get(
                        ModelPreheatS3Profile, profile_id
                    )
                    winner_task = (
                        await winner_session.exec(
                            select(ModelPreheatWorkerTask).where(
                                ModelPreheatWorkerTask.connectivity_check_id == check_id
                            )
                        )
                    ).one()
                    winner_task.state = ModelPreheatWorkerTaskStateEnum.READY
                    winner_profile.connectivity_state = (
                        ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                    )
                    winner_profile.last_connectivity_checked_at = datetime.now(
                        timezone.utc
                    )
                    winner_session.add_all([winner_profile, winner_task])
                    await winner_session.commit()

                resume.set()
                await old_aggregate

            async with AsyncSession(engine) as verify_session:
                stored = await verify_session.get(ModelPreheatS3Profile, profile_id)
                assert (
                    stored.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                )
                assert stored.last_connectivity_checked_at is not None
                assert invocation_count == 2
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_old_session_aggregation_cannot_overwrite_new_profile_check(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'aggregate-cas.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as setup_session:
                profile = _profile()
                worker = _worker("worker", "worker-uuid")
                setup_session.add_all([profile, worker])
                await setup_session.commit()
                check_v1 = await create_or_reuse_connectivity_check(
                    setup_session, profile
                )
                task_v1 = (
                    await setup_session.exec(
                        select(ModelPreheatWorkerTask).where(
                            ModelPreheatWorkerTask.connectivity_check_id == check_v1.id
                        )
                    )
                ).one()
                profile_id = profile.id
                check_v1_id = check_v1.id
                task_v1.state = ModelPreheatWorkerTaskStateEnum.READY
                setup_session.add(task_v1)
                await setup_session.commit()

            async with AsyncSession(engine) as old_session:
                old_profile = await old_session.get(ModelPreheatS3Profile, profile_id)
                old_check = await old_session.get(
                    ModelPreheatS3ConnectivityCheck, check_v1_id
                )
                assert old_profile.config_version == 1
                assert old_check.id == check_v1_id

                async with AsyncSession(engine) as new_session:
                    profile_v2 = await new_session.get(
                        ModelPreheatS3Profile, profile_id
                    )
                    profile_v2.config_version = 2
                    profile_v2.connectivity_state = (
                        ModelPreheatS3ConnectivityStateEnum.PENDING
                    )
                    profile_v2.last_connectivity_check_id = None
                    profile_v2.last_connectivity_checked_at = None
                    new_session.add(profile_v2)
                    await new_session.commit()
                    check_v2 = await create_or_reuse_connectivity_check(
                        new_session, profile_v2
                    )
                    check_v2_id = check_v2.id

                await aggregate_connectivity_check(old_session, check_v1_id)

            async with AsyncSession(engine) as verify_session:
                stored = await verify_session.get(ModelPreheatS3Profile, profile_id)
                assert stored.config_version == 2
                assert stored.last_connectivity_check_id == check_v2_id
                assert (
                    stored.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.CHECKING
                )
                assert stored.last_connectivity_checked_at is None
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_old_session_check_creation_cannot_replace_new_profile_check(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'create-check-cas.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as setup_session:
                profile = _profile()
                setup_session.add_all([profile, _worker("worker", "worker-uuid")])
                await setup_session.flush()
                profile_id = profile.id
                await setup_session.commit()

            async with AsyncSession(engine) as old_session:
                old_profile = await old_session.get(ModelPreheatS3Profile, profile_id)
                assert old_profile.config_version == 1

                async with AsyncSession(engine) as new_session:
                    profile_v2 = await new_session.get(
                        ModelPreheatS3Profile, profile_id
                    )
                    profile_v2.config_version = 2
                    profile_v2.connectivity_state = (
                        ModelPreheatS3ConnectivityStateEnum.PENDING
                    )
                    new_session.add(profile_v2)
                    await new_session.commit()
                    check_v2 = await create_or_reuse_connectivity_check(
                        new_session, profile_v2
                    )
                    check_v2_id = check_v2.id

                check_v1 = await create_or_reuse_connectivity_check(
                    old_session, old_profile
                )
                assert check_v1.profile_config_version == 1

            async with AsyncSession(engine) as verify_session:
                stored = await verify_session.get(ModelPreheatS3Profile, profile_id)
                assert stored.config_version == 2
                assert stored.last_connectivity_check_id == check_v2_id
                assert (
                    stored.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.CHECKING
                )
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_old_session_ttl_mark_cannot_make_new_profile_check_stale(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'stale-cas.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as setup_session:
                profile = _profile()
                profile.connectivity_state = (
                    ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                )
                profile.last_connectivity_checked_at = (
                    datetime.now(timezone.utc)
                    - DEFAULT_CONNECTIVITY_TTL
                    - timedelta(seconds=1)
                )
                setup_session.add_all([profile, _worker("worker", "worker-uuid")])
                await setup_session.flush()
                profile_id = profile.id
                await setup_session.commit()

            async with AsyncSession(engine) as old_session:
                old_profile = await old_session.get(ModelPreheatS3Profile, profile_id)
                assert (
                    old_profile.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                )

                async with AsyncSession(engine) as new_session:
                    profile_v2 = await new_session.get(
                        ModelPreheatS3Profile, profile_id
                    )
                    profile_v2.config_version = 2
                    profile_v2.connectivity_state = (
                        ModelPreheatS3ConnectivityStateEnum.PENDING
                    )
                    profile_v2.last_connectivity_checked_at = None
                    new_session.add(profile_v2)
                    await new_session.commit()
                    check_v2 = await create_or_reuse_connectivity_check(
                        new_session, profile_v2
                    )
                    check_v2_id = check_v2.id

                assert not await mark_profile_stale_if_expired(old_session, old_profile)
                await old_session.commit()

            async with AsyncSession(engine) as verify_session:
                stored = await verify_session.get(ModelPreheatS3Profile, profile_id)
                assert stored.config_version == 2
                assert stored.last_connectivity_check_id == check_v2_id
                assert (
                    stored.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.CHECKING
                )
                assert stored.last_connectivity_checked_at is None
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_zero_workers_and_successful_connectivity_does_not_expire(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'ttl.db'}", poolclass=NullPool
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                profile = _profile()
                session.add(profile)
                await session.commit()
                assert (
                    await create_or_reuse_connectivity_check(session, profile) is None
                )
                assert (
                    profile.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.NO_WORKERS
                )

                profile.connectivity_state = (
                    ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                )
                profile.last_connectivity_checked_at = (
                    datetime.now(timezone.utc)
                    - DEFAULT_CONNECTIVITY_TTL
                    - timedelta(seconds=1)
                )
                session.add(profile)
                await session.commit()
                assert not await mark_profile_stale_if_expired(session, profile)
                assert (
                    profile.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.AVAILABLE
                )

                profile.connectivity_state = ModelPreheatS3ConnectivityStateEnum.PARTIAL
                profile.last_connectivity_checked_at = (
                    datetime.now(timezone.utc)
                    - DEFAULT_CONNECTIVITY_TTL
                    - timedelta(seconds=1)
                )
                session.add(profile)
                await session.commit()
                assert await mark_profile_stale_if_expired(session, profile)
                assert (
                    profile.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.STALE
                )
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_registered_not_ready_worker_is_no_workers(tmp_path):
    async def run():
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path / 'offline-only.db'}",
            poolclass=NullPool,
        )
        await _tables(engine, SQLModel.metadata.create_all)
        try:
            async with AsyncSession(engine) as session:
                profile = _profile()
                session.add_all(
                    [
                        profile,
                        _worker("offline", "offline-uuid", WorkerStateEnum.NOT_READY),
                    ]
                )
                await session.commit()

                assert (
                    await create_or_reuse_connectivity_check(session, profile) is None
                )
                assert (
                    profile.connectivity_state
                    == ModelPreheatS3ConnectivityStateEnum.NO_WORKERS
                )
        finally:
            await _tables(engine, SQLModel.metadata.drop_all)
            await engine.dispose()

    asyncio.run(run())


def test_worker_network_identity_change_ignores_heartbeat_only_changes():
    before = _worker("worker", "uuid")
    after = _worker("worker", "uuid")
    after.heartbeat_time = datetime.now(timezone.utc)
    assert not worker_network_identity_changed(before, after)
    after.ip = "127.0.0.2"
    assert worker_network_identity_changed(before, after)


@pytest.mark.parametrize(
    "environment_name",
    [
        "GPUSTACK_MODEL_PREHEAT_CONNECTIVITY_TTL_SECONDS",
        "GPU_STACK_MODEL_PREHEAT_CONNECTIVITY_TTL_SECONDS",
    ],
)
def test_config_reads_model_preheat_connectivity_ttl_alias(
    monkeypatch, tmp_path, environment_name
):
    monkeypatch.delenv("GPUSTACK_MODEL_PREHEAT_CONNECTIVITY_TTL_SECONDS", raising=False)
    monkeypatch.delenv(
        "GPU_STACK_MODEL_PREHEAT_CONNECTIVITY_TTL_SECONDS", raising=False
    )
    monkeypatch.setenv(environment_name, "321")

    config = Config(data_dir=str(tmp_path))

    assert config.model_preheat_connectivity_ttl_seconds == 321
