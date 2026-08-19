from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


def validate_revision_graph(script_location: Path | str | None = None) -> str:
    location = Path(script_location or Path(__file__).resolve().parent)
    config = Config()
    config.set_main_option("script_location", str(location))
    scripts = ScriptDirectory.from_config(config)

    revisions = list(scripts.walk_revisions())
    if not revisions:
        raise RuntimeError("alembic_revision_graph_empty")
    heads = scripts.get_heads()
    if len(heads) != 1:
        raise RuntimeError(f"alembic_revision_graph_has_multiple_heads: {heads}")
    return heads[0]


if __name__ == "__main__":
    print(validate_revision_graph())
