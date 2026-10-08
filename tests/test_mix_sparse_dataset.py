"""Tests for plant_counter.mix_sparse_dataset (sparse mixed broadleaf dataset builder)."""

import json
import sys
from pathlib import Path

import pytest

from plant_counter import mix_sparse_dataset as mix


def _write_pair(image_dir: Path, label_dir: Path, stem: str, label_text: str = "0 0.5 0.5 0.1 0.1\n"):
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{stem}.jpg").write_bytes(b"fake-image-" + stem.encode())
    (label_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")


def _make_external(root: Path, stems):
    for stem in stems:
        _write_pair(root / "images" / "train", root / "labels" / "train", stem)


def _make_target(root: Path, train=("t1",), val=("v1",), test=("s1",)):
    for split, stems in (("train", train), ("val", val), ("test", test)):
        for stem in stems:
            _write_pair(root / split / "images", root / split / "labels", stem)


def _run(monkeypatch, external, target, output, *extra):
    argv = [
        "mix_sparse_dataset",
        "--external", str(external),
        "--target", str(target),
        "--output", str(output),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    mix.main()


@pytest.fixture
def dataset(tmp_path):
    external = tmp_path / "external"
    target = tmp_path / "target"
    _make_external(external, ["e1", "e2"])
    _make_target(target)
    return external, target, tmp_path / "out"


def test_builds_train_val_test_with_target_replay(monkeypatch, dataset, capsys):
    external, target, output = dataset

    _run(monkeypatch, external, target, output, "--target-repeats", "3")

    train_images = sorted(p.name for p in (output / "images" / "train").iterdir())
    assert len(train_images) == 2 + 3
    assert [n for n in train_images if n.startswith("external__")] == ["external__e1.jpg", "external__e2.jpg"]
    assert [n for n in train_images if n.startswith("targetr")] == [
        "targetr00__t1.jpg", "targetr01__t1.jpg", "targetr02__t1.jpg",
    ]
    assert len(list((output / "labels" / "train").iterdir())) == 5
    assert (output / "images" / "val" / "target__v1.jpg").is_file()
    assert (output / "labels" / "test" / "target__s1.txt").is_file()
    assert "target_image_fraction" in capsys.readouterr().out


def test_writes_data_yaml_and_summary(monkeypatch, dataset):
    external, target, output = dataset

    _run(monkeypatch, external, target, output, "--target-repeats", "2")

    yaml_text = (output / "data.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in yaml_text
    assert "val: images/val" in yaml_text
    assert "test: images/test" in yaml_text
    assert "0: broadleaf_weed" in yaml_text
    summary = json.loads((output / "mix_summary.json").read_text(encoding="utf-8"))
    train = summary["splits"]["train"]
    assert train["external_images"] == 2
    assert train["target_images_after_replay"] == 2
    assert train["target_image_fraction"] == pytest.approx(0.5)
    assert summary["splits"]["val"]["target_images"] == 1


def test_single_repeat_uses_plain_target_prefix(monkeypatch, dataset):
    external, target, output = dataset

    _run(monkeypatch, external, target, output, "--target-repeats", "1")

    names = {p.name for p in (output / "images" / "train").iterdir()}
    assert "target__t1.jpg" in names
    assert not any(n.startswith("targetr") for n in names)


def test_missing_label_raises_file_not_found(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _write_pair(images, labels, "a")
    (labels / "a.txt").unlink()

    with pytest.raises(FileNotFoundError, match="Missing label"):
        mix._add_split(images, labels, tmp_path / "out", "train", "target", 1)


def test_refuses_non_empty_output_folder(monkeypatch, dataset):
    external, target, output = dataset
    output.mkdir()
    (output / "keep.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(SystemExit):
        _run(monkeypatch, external, target, output)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "existing"


def test_rejects_target_repeats_below_one(monkeypatch, dataset):
    external, target, output = dataset

    with pytest.raises(SystemExit):
        _run(monkeypatch, external, target, output, "--target-repeats", "0")

    assert not output.exists()


def test_external_limit_is_deterministic_and_limits_count(monkeypatch, tmp_path):
    external = tmp_path / "external"
    target = tmp_path / "target"
    _make_external(external, [f"e{i}" for i in range(6)])
    _make_target(target)

    _run(monkeypatch, external, target, tmp_path / "out1", "--external-limit", "3", "--target-repeats", "1")
    _run(monkeypatch, external, target, tmp_path / "out2", "--external-limit", "3", "--target-repeats", "1")

    def picked(out):
        return sorted(p.name for p in (out / "images" / "train").iterdir() if p.name.startswith("external__"))

    assert len(picked(tmp_path / "out1")) == 3
    assert picked(tmp_path / "out1") == picked(tmp_path / "out2")


def test_external_limit_must_be_positive(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _write_pair(images, labels, "a")

    with pytest.raises(ValueError):
        mix._add_split(images, labels, tmp_path / "out", "train", "external", 1, limit=0)


def test_falls_back_to_copy_when_hardlink_unavailable(monkeypatch, tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _write_pair(images, labels, "a")

    def no_link(src, dst):
        raise OSError("hardlinks not supported")

    monkeypatch.setattr(mix.os, "link", no_link)

    count, _, modes = mix._add_split(images, labels, tmp_path / "out", "train", "target", 1)

    assert count == 1
    assert modes == {"hardlink": 0, "copy": 1}
    assert (tmp_path / "out" / "images" / "train" / "target__a.jpg").read_bytes() == b"fake-image-a"


def test_counts_only_non_blank_label_lines(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _write_pair(images, labels, "a", "0 0.1 0.1 0.1 0.1\n\n0 0.5 0.5 0.2 0.2\n   \n")
    _write_pair(images, labels, "b", "")

    count, instances, _ = mix._add_split(images, labels, tmp_path / "out", "train", "target", 1)

    assert count == 2
    assert instances == 2


def test_refuses_to_overwrite_existing_output_files(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _write_pair(images, labels, "a")
    out = tmp_path / "out"
    mix._add_split(images, labels, out, "train", "target", 1)

    with pytest.raises(FileExistsError):
        mix._add_split(images, labels, out, "train", "target", 1)