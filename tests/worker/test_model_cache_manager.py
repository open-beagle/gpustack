import pytest

from gpustack.worker.model_cache_manager import _source_files


def test_source_files_walks_directory_with_relative_paths(tmp_path):
    model = tmp_path / "model"
    (model / "weights").mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "weights" / "model.bin").write_bytes(b"model")

    files = _source_files([str(model)])

    assert [relative for _, relative in files] == [
        "config.json",
        "weights/model.bin",
    ]


def test_source_files_rejects_symlink(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"model")
    link = tmp_path / "link"
    link.symlink_to(source)

    with pytest.raises(ValueError, match="model_cache_source_symlink"):
        _source_files([str(link)])


def test_source_files_rejects_duplicate_relative_paths(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "config.json").write_text("first")
    (second / "config.json").write_text("second")

    with pytest.raises(ValueError, match="model_cache_duplicate_source_path"):
        _source_files([str(first), str(second)])
