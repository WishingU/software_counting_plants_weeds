
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import count_plants as counter


@pytest.fixture
def mock_setup(tmp_path, monkeypatch):
    image = tmp_path / "image.jpg"
    weights = tmp_path / "model.pt"

    image.write_bytes(b"test image")
    weights.write_bytes(b"mock weights")

    model = MagicMock()
    model.names = {0: "wheat", 1: "wild oat"}

    result = SimpleNamespace(boxes=[])
    model.predict.return_value = [result]

    monkeypatch.setattr(counter, "YOLO", lambda path: model)

    return image, weights, model, result


def make_box(class_id):
    return SimpleNamespace(
        cls=SimpleNamespace(item=lambda: class_id)
    )


def test_single_species_count(mock_setup):
    image, weights, model, result = mock_setup
    result.boxes = [make_box(0), make_box(0)]

    counts, _ = counter.count_plants(image, weights)

    assert counts == {"wheat": 2, "wild oat": 0}


def test_multiple_species_counts(mock_setup):
    image, weights, model, result = mock_setup
    result.boxes = [
        make_box(0),
        make_box(1),
        make_box(1),
    ]

    counts, _ = counter.count_plants(image, weights)

    assert counts == {"wheat": 1, "wild oat": 2}


def test_zero_detections(mock_setup):
    image, weights, model, result = mock_setup

    counts, _ = counter.count_plants(image, weights)

    assert counts == {"wheat": 0, "wild oat": 0}


def test_missing_image(mock_setup):
    image, weights, model, result = mock_setup

    with pytest.raises(
        FileNotFoundError,
        match="Image does not exist"
    ):
        counter.count_plants(
            image.parent / "missing.jpg",
            weights
        )


def test_missing_weights(mock_setup):
    image, weights, model, result = mock_setup

    with pytest.raises(
        FileNotFoundError,
        match="Model weights do not exist"
    ):
        counter.count_plants(
            image,
            weights.parent / "missing.pt"
        )


def test_custom_parameters(mock_setup):
    image, weights, model, result = mock_setup

    counter.count_plants(
        image,
        weights,
        confidence=0.5,
        iou=0.4,
        device="cpu",
    )

    model.predict.assert_called_once_with(
        source=str(image),
        conf=0.5,
        iou=0.4,
        device="cpu",
        verbose=False,
    )


def test_default_parameters(mock_setup):
    image, weights, model, result = mock_setup

    counter.count_plants(image, weights)

    model.predict.assert_called_once_with(
        source=str(image),
        conf=0.25,
        iou=0.3,
        device=None,
        verbose=False,
    )


def test_green_enhancement(mock_setup, monkeypatch, tmp_path):
    from PIL import Image

    image, weights, model, result = mock_setup

    # Create a real image for preprocessing
    real_image = tmp_path / "real_image.png"
    Image.new("RGB", (10, 10), (0, 180, 0)).save(real_image)

    enhanced_image = object()
    enhance_mock = MagicMock(return_value=enhanced_image)
    monkeypatch.setattr(counter, "enhance_green", enhance_mock)

    counter.count_plants(
        real_image,
        weights,
        enhance_green_input=True,
    )

    enhance_mock.assert_called_once()
    assert enhance_mock.call_args.args[0].mode == "RGB"

    model.predict.assert_called_once_with(
        source=enhanced_image,
        conf=0.25,
        iou=0.3,
        device=None,
        verbose=False,
    )


def test_cli_output(mock_setup, monkeypatch, capsys):
    import sys

    image, weights, model, result = mock_setup
    result.boxes = [
        make_box(0),
        make_box(1),
        make_box(1),
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "count_plants.py",
            str(image),
            "--weights",
            str(weights),
        ],
    )

    counter.main()

    output = capsys.readouterr().out

    assert "wheat: 1" in output
    assert "wild oat: 2" in output
    assert "total: 3" in output


def test_cli_save(mock_setup, monkeypatch, tmp_path):
    import sys

    image, weights, model, result = mock_setup
    output_path = tmp_path / "output" / "annotated.jpg"

    save_mock = MagicMock()
    result.save = save_mock

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "count_plants.py",
            str(image),
            "--weights",
            str(weights),
            "--save",
            str(output_path),
        ],
    )

    counter.main()

    assert output_path.parent.is_dir()
    save_mock.assert_called_once_with(
        filename=str(output_path)
    )

