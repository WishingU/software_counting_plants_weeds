# Testing Log (broadleaf model)

Tester: Ami 
Branch: testing


## Test 1: Existing unit tests (2026-10-08)
Command: python -m pytest -v -m "not slow and not model and not acceptance"
Environment: Mac (Apple Silicon), conda env plant-count, Python 3.11.15, pytest 9.1.1
Result: 13 passed, 0 failed, 0 skipped (30.14 s)
Note: The first run failed with "No module named pytest". requirements-dev.txt did not install pytest, so I ran `pip install pytest pytest-cov`. These tests cover supporting code and do not load any model weights.

## Finding: model weights are Git LFS pointers (2026-10-08)
After pulling main, `ls -lh models/*/` showed every .pt file at about 132-133 bytes, so they are LFS pointers and not real weights. README.md does not mention `git lfs pull`.
models/broadleaf/yolo26m_broadleaf.pt: expected size 44,020,313 bytes, SHA-256 edf2cab2d4185f8dd4ae27931c83d6cf57b5d1ae5c163c5c390da3c8e8192f01