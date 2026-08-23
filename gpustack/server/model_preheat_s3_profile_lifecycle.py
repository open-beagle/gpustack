"""S3 Profile 生命周期的数据库线性化辅助。"""

from sqlalchemy import update

from gpustack.schemas.model_preheat_s3_profiles import (
    DEFAULT_SLOT_GLOBAL,
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
)


class ModelPreheatS3ProfileNotActive(Exception):
    pass


async def lock_active_profile_for_new_work(
    session,
    profile_id: int,
    config_version: int,
    *,
    require_default: bool = False,
) -> ModelPreheatS3Profile:
    """以条件 UPDATE 获取写锁，并把新任务线性化在维护切换之前或之后。"""
    conditions = [
        ModelPreheatS3Profile.id == profile_id,
        ModelPreheatS3Profile.config_version == config_version,
        ModelPreheatS3Profile.lifecycle_state
        == ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE,
    ]
    if require_default:
        conditions.append(ModelPreheatS3Profile.default_slot == DEFAULT_SLOT_GLOBAL)
    result = await session.exec(
        update(ModelPreheatS3Profile)
        .where(*conditions)
        .values(active_storage_key=ModelPreheatS3Profile.active_storage_key)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        raise ModelPreheatS3ProfileNotActive
    profile = await session.get(
        ModelPreheatS3Profile, profile_id, populate_existing=True
    )
    if profile is None:
        raise ModelPreheatS3ProfileNotActive
    return profile
