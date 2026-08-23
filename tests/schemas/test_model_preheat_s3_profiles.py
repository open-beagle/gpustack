from sqlalchemy.dialects import sqlite

from gpustack.schemas.model_preheat_s3_profiles import (
    ModelPreheatS3Profile,
    ModelPreheatS3ProfileLifecycleStateEnum,
)


def test_lifecycle_state_orm_type_uses_enum_values():
    column_type = ModelPreheatS3Profile.__table__.c.lifecycle_state.type
    dialect = sqlite.dialect()

    bind_processor = column_type.bind_processor(dialect)
    result_processor = column_type.result_processor(dialect, None)

    assert bind_processor(ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE) == "active"
    assert result_processor("active") == ModelPreheatS3ProfileLifecycleStateEnum.ACTIVE
