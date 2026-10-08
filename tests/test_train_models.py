"""Tests for plant_counter.train: the wrapper must work with any YOLO model name.

Ultralytics and torch are replaced by tiny fakes, so no weights are loaded,
nothing is downloaded and no training is run.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from plant_counter import train


class FakeYOLO:
    """Records how the training wrapper uses ultralytics.YOLO."""

    created = []
    train_calls = []

    def __init__(self, model):
        self.model = model
        FakeYOLO.created.append(model)

    def train(self, **kwargs):
        FakeYOLO.train_calls.append(kwargs)


def _install_fakes(monkeypatch, cuda=False):
    FakeYOLO.created = []
    FakeYOLO.train_calls = []
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "0.0-fake"
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_name=lambda index: "Fake GPU",
    )
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.__version__ = "0.0-fake"
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)


@pytest.fixture
def run_train(monkeypatch, tmp_path):
    """Run train.main() with fake torch/ultralytics; returns (project_dir, data_yaml)."""
    _install_fakes(monkeypatch)
    data = tmp_path / "data.yaml"
    data.write_text("names:\n  0: broadleaf_weed\n", encoding="utf-8")
    project = tmp_path / "runs"

    def _run(*extra):
        argv = ["train", "--data", str(data), "--project", str(project), *extra]
        monkeypatch.setattr(sys, "argv", argv)
        train.main()
        return project, data

    return _run


@pytest.mark.parametrize(
    "model_name",
    ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo26m.pt", "models/broadleaf/custom_weights.pt", "yolov8n.yaml"],
)
def test_passes_any_model_name_to_yolo(run_train, model_name):
    run_train("--model", model_name)

    assert FakeYOLO.created == [model_name]
    assert len(FakeYOLO.train_calls) == 1


def test_default_model_is_yolo26n(run_train):
    run_train()

    assert FakeYOLO.created == ["yolo26n.pt"]


def test_forwards_core_training_arguments(run_train):
    project, data = run_train(
        "--epochs", "7", "--imgsz", "768", "--batch", "8", "--device", "cpu",
        "--workers", "2", "--patience", "5", "--seed", "42", "--name", "exp1",
    )

    kwargs = FakeYOLO.train_calls[0]
    assert kwargs["data"] == str(data.resolve())
    assert kwargs["epochs"] == 7
    assert kwargs["imgsz"] == 768
    assert kwargs["batch"] == 8
    assert kwargs["device"] == "cpu"
    assert kwargs["workers"] == 2
    assert kwargs["patience"] == 5
    assert kwargs["seed"] == 42
    assert kwargs["deterministic"] is True
    assert kwargs["project"] == str(project.resolve())
    assert kwargs["name"] == "exp1"


def test_optional_hyperparameters_are_not_sent_by_default(run_train):
    run_train()

    kwargs = FakeYOLO.train_calls[0]
    for key in ("lr0", "lrf", "weight_decay", "close_mosaic", "warmup_epochs", "optimizer"):
        assert key not in kwargs


def test_optional_hyperparameters_are_forwarded_when_given(run_train):
    run_train(
        "--lr0", "0.01", "--lrf", "0.1", "--weight-decay", "0.0005",
        "--close-mosaic", "5", "--warmup-epochs", "2.5", "--optimizer", "SGD",
    )

    kwargs = FakeYOLO.train_calls[0]
    assert kwargs["lr0"] == pytest.approx(0.01)
    assert kwargs["lrf"] == pytest.approx(0.1)
    assert kwargs["weight_decay"] == pytest.approx(0.0005)
    assert kwargs["close_mosaic"] == 5
    assert kwargs["warmup_epochs"] == pytest.approx(2.5)
    assert kwargs["optimizer"] == "SGD"


def test_missing_dataset_yaml_exits_before_loading_model(monkeypatch, tmp_path):
    _install_fakes(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(tmp_path / "missing.yaml"), "--project", str(tmp_path / "runs")])

    with pytest.raises(SystemExit, match="Dataset YAML does not exist"):
        train.main()

    assert FakeYOLO.created == []
    assert not (tmp_path / "runs").exists()


def test_missing_training_dependencies_exit_cleanly(monkeypatch, tmp_path):
    data = tmp_path / "data.yaml"
    data.write_text("names: {}\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(data), "--project", str(tmp_path / "runs")])

    with pytest.raises(SystemExit, match="Training dependencies are missing"):
        train.main()


def test_writes_environment_metadata_without_cuda(run_train):
    project, data = run_train("--model", "yolov8s.pt", "--name", "meta1", "--device", "cpu")

    metadata = json.loads((project / "meta1_environment.json").read_text(encoding="utf-8"))
    assert metadata["cuda_available"] is False
    assert metadata["gpu"] is None
    assert metadata["torch"] == "0.0-fake"
    assert metadata["ultralytics"] == "0.0-fake"
    assert metadata["arguments"]["model"] == "yolov8s.pt"
    assert metadata["arguments"]["data"] == str(data)


def test_writes_gpu_name_in_metadata_when_cuda_available(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, cuda=True)
    data = tmp_path / "data.yaml"
    data.write_text("names: {}\n", encoding="utf-8")
    project = tmp_path / "runs"
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(data), "--project", str(project), "--name", "gpu1"])

    train.main()

    metadata = json.loads((project / "gpu1_environment.json").read_text(encoding="utf-8"))
    assert metadata["cuda_available"] is True
    assert metadata["gpu"] == "Fake GPU"


def test_scripts_train_wrapper_uses_the_maintained_main():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("scripts_train_wrapper", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main is train.main