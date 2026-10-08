# Testing Log (broadleaf model)

Tester: Ami Takyu
Branch: testing

## Test 1: Existing unit tests (2026-10-08)
Command: python -m pytest -v -m "not slow and not model and not acceptance"
Environment: Mac (Apple Silicon), conda env plant-count, Python 3.11.15, pytest 9.1.1
Result: 13 passed, 0 failed, 0 skipped (30.14 s)
Note: The first run failed with "No module named pytest". requirements-dev.txt did not install pytest, so I ran `pip install pytest pytest-cov`. These tests cover supporting code and do not load any model weights.
Update: after pulling main on 2026-10-08, requirements-dev.txt lists `pytest>=8.4,<10` and `pytest-cov>=7,<8`, so the missing-pytest issue appears to be fixed in main.

## Finding: model weights are Git LFS pointers (2026-10-08)
After pulling main, `ls -lh models/*/` showed every .pt file at about 132-133 bytes, so they are LFS pointers and not real weights. README.md does not mention `git lfs pull`.
models/broadleaf/yolo26m_broadleaf.pt: expected size 44,020,313 bytes, SHA-256 edf2cab2d4185f8dd4ae27931c83d6cf57b5d1ae5c163c5c390da3c8e8192f01

## Test 2: Unit tests for src/plant_counter/mix_sparse_dataset.py (2026-10-08)
File added: tests/test_mix_sparse_dataset.py
Command: python -m pytest tests/test_mix_sparse_dataset.py -v
Environment: Mac, conda env plant-count, Python 3.11.15, pytest 9.1.1
Result: 11 passed, 0 failed, 0 skipped (0.12 s)
Covered: output layout and file names (external__ / targetNN__ prefixes), data.yaml and mix_summary.json contents, missing label raises FileNotFoundError, non-empty output folder is refused, target-repeats below 1 is refused, --external-limit is deterministic and must be positive, hardlink-to-copy fallback, blank label lines are not counted, existing files are not overwritten.
Note: the first run failed with a SyntaxError because the pasted test file was cut off at the end (line 180). I completed the last test and re-ran.

## Test 3: Unit tests for src/plant_counter/train.py with different model names (2026-10-08)
File added: tests/test_train_models.py
Command: python -m pytest tests/test_train_models.py -v
Environment: same as Test 2
Result: 15 passed, 0 failed, 0 skipped (0.07 s)
Method: torch and ultralytics.YOLO are replaced by fakes, so no weights are loaded and no training is run.
Model names passed to --model: yolov8n.pt, yolov8s.pt, yolo11n.pt, yolo26m.pt, models/broadleaf/custom_weights.pt, yolov8n.yaml. Each name reaches YOLO() unchanged.
Also covered: default model (yolo26n.pt), forwarded training arguments, optional hyperparameters (sent only when given), missing dataset YAML, missing dependencies, environment metadata with and without CUDA, and scripts/train.py using the maintained main().
The existing tests/test_train.py only checks --batch parsing, so these tests do not repeat it.

## Full fast suite (2026-10-08)
Command: python -m pytest -v -m "not slow and not model and not acceptance"
Result: 39 passed, 0 failed, 0 skipped (0.35 s)

## Not tested / limitations
- No real model was loaded. Per Yin's instruction, the weights were not pulled; the .pt files in the repo are still Git LFS pointers.
- The train tests check that arguments are forwarded. They do not check that real Ultralytics accepts each model name.
- No acceptance test or deployed-app test was run.
- Production code was not changed.
- Observation: the dependency error message in src/plant_counter/train.py has no space in "`python -m pip install -e .`first." (cosmetic; not changed).