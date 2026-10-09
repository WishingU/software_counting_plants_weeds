# Testing Agent Guide

## Purpose and scope

Use these instructions whenever you add, review, repair, or run tests in this
repository. Act as a strict but practical software tester. Your goal is to find
real defects, protect documented behaviour, and leave reproducible evidence.
Do not write tests merely to increase a coverage number.

This is a CITS3200 group project. Testing is part of the assessed engineering
process, not a final demonstration. Prefer small tests that can run on every
change, then add focused integration and acceptance evidence for paths that need
real images or model weights.

Unless the user asks for a bug fix, change tests and test fixtures only. You may
recommend a production-code change, but do not silently change the application
to make a failing test pass.

## Project contract

Read the files related to the requested behaviour before designing tests. The
main sources of truth are:

- `README.md` and `README_WINDOWS.md` for supported workflows and commands.
- `HANDOVER.md` for validated inference settings and known limitations.
- `configs/annotation_rules.yaml` and `docs/ANNOTATION_GUIDE.en.md` for the
  annotation contract.
- `src/plant_counter/` for maintained dataset, audit, metric, evaluation, and
  training code.
- `app.py` and `count_plants.py` for Streamlit and command-line inference.
- `scripts/` for conversion, preparation, diagnostics, and compatibility
  workflows. Similar script names do not guarantee identical contracts.
- `tests/` for current conventions and regression coverage.

If documentation and implementation disagree, do not guess which is correct.
Record the conflict, test the safest established behaviour if possible, and ask
for a product decision when it affects user-visible or scientific results.

## Environment and canonical commands

The supported local environment is the Conda environment `plant-count`.

In an interactive PowerShell session, use:

```powershell
conda activate plant-count
python -m pip install -r requirements-dev.txt
python -m pytest -v -m "not slow and not model and not acceptance"
```

For a non-interactive agent, use:

```powershell
conda run -n plant-count python -m pytest -v -m "not slow and not model and not acceptance"
```

Pytest is the canonical runner. Its configuration in `pyproject.toml` adds this
checkout's `src` directory to the import path, preventing Python from using a
stale editable installation. The legacy equivalent is:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

Keep the existing `unittest.TestCase` tests; pytest discovers and runs them.
Prefer pytest style for new tests: plain test functions, fixtures such as
`tmp_path` and `monkeypatch`, `pytest.raises`, and `pytest.mark.parametrize`.
Do not rewrite working unittest tests merely for stylistic consistency.

Before reporting a result, include the exact command, environment, number of
tests run, failures/errors, and skipped tests. Never say tests pass if the
command did not complete. Separate code failures from environment, dependency,
missing-data, missing-LFS-object, or GPU failures.

## Required workflow

1. Inspect `git status` and the relevant diff. Preserve unrelated work.
2. Identify the observable contract and likely failure modes before writing a
   test. For a bug, create a regression test that fails for the defect and
   passes for the intended behaviour.
3. Choose the lowest useful level: unit test first, file/CLI or UI integration
   next, and real-model acceptance only when it adds value.
4. Add a representative success case and meaningful boundary or failure cases.
   A happy-path-only test is incomplete for parsing, paths, datasets, or input.
5. Run the narrow new test while iterating, then the full canonical suite.
6. Review the diff for accidental data, generated output, absolute local paths,
   model binaries, sleeps, network calls, and assertions that cannot fail.
7. Report what was tested, what remains untested, and why. Do not hide skips or
   known failures.

When practical, prove that a regression test detects the defect by running it
against the pre-fix code or a temporary local mutation. Revert the mutation
before finishing and never weaken an assertion merely to make the suite green.

## Test levels

### Fast unit tests: required by default

Fast tests must be deterministic, CPU-only, offline, and independent of local
untracked datasets. Use `tempfile.TemporaryDirectory`, small synthetic NumPy
arrays, tiny valid images, and minimal JSON/YAML fixtures. Patch model and GPU
boundaries rather than loading YOLO weights.

For new pytest-style tests, prefer `tmp_path` over hand-managed temporary
directories and `monkeypatch` over persistent global changes. Use
`pytest.mark.parametrize` when several inputs express the same behaviour and
the individual case IDs remain readable.

Fast tests should normally complete in seconds and cover:

- return values and externally visible side effects;
- invalid types, malformed records, missing files, empty inputs, and boundaries;
- output file contents, exit codes, and contractual error messages;
- input preservation and cleanup after failures;
- Windows paths and portable `pathlib.Path` behaviour.

Do not assert private implementation details when an observable contract is
available. Do not mock the function under test. Mock only expensive or external
boundaries such as `ultralytics.YOLO`, CUDA, Streamlit uploads, and filesystem
link support.

### Integration tests: component boundaries

Integration tests may exercise temporary directory trees, real image codecs,
CLI parsing, YAML/JSON conversion, and maintained-module interactions. They
must avoid network access, downloads, training, and writes to the repository's
real `data/`, `models/`, `runs/`, or `outputs/` folders.

For a CLI, prefer its parser or `main()` with patched `sys.argv` and captured
output. Use a subprocess only when exit status, import behaviour, or an installed
entry point is the behaviour under test.

For Streamlit, use `streamlit.testing.v1.AppTest` when it expresses the user
flow. Stub YOLO for ordinary UI tests. Verify visible errors and results, not
Streamlit widget internals.

### Real-model and acceptance tests: opt in

Real inference and evaluation are slow, require Git LFS objects and local data,
and may vary with Ultralytics, Torch, device, and CUDA versions. Keep them out of
the default fast suite. Mark them with `@pytest.mark.model` and usually
`@pytest.mark.slow`. Gate unavailable artefacts with `pytest.skip` or
`@pytest.mark.skipif` and a precise reason. Running model tests must be an
explicit choice:

```powershell
$env:PLANT_RUN_MODEL_TESTS = "1"
python -m pytest -v -m model
```

Mark end-to-end client scenarios with `@pytest.mark.acceptance`. A marker alone
is not evidence that a test passed and must not replace the recorded acceptance
result.

Before loading a `.pt` file, verify it is a real LFS object rather than Git's
small text pointer. Never download weights or datasets without permission.
Never start training merely to test the training wrapper.

Acceptance tests must trace to a documented requirement or agreed client
scenario. Record the model file and hash, dataset/split, software versions,
device, confidence, IoU, image size, seed, expected result or threshold, actual
result, and pass/fail outcome. Use a fixed independent test split for reported
model quality; never present training or validation results as independent
acceptance evidence.

Avoid exact boxes or confidence values across ML runtimes unless environment and
artefacts are pinned. Prefer defensible tolerances, count ranges, thresholds, or
invariants. Existing annotations have known gaps, so distinguish raw label-based
metrics from manually reviewed counting quality. Do not change a threshold only
to make precision appear better.

## Repository-specific risk checklist

Select relevant items; do not test this entire list for every change.

### Image preprocessing and inference

- RGB/BGR conversion is explicit at the OpenCV/Ultralytics boundary.
- Green enhancement validates shape, `uint8` dtype, and strength; does not
  mutate input; leaves red/blue and neutral/non-green pixels unchanged; and
  cannot overflow.
- `count_plants.py` forwards confidence, IoU, device, and preprocessing options,
  counts each returned class, and handles missing image/weight files.
- Streamlit handles no model, invalid uploads, inference failure, zero boxes,
  species ordering, non-species models, device selection, and cache invalidation.
  Uploaded weights are untrusted executable artefacts; never load an arbitrary
  fixture as a real Torch model.

### Dataset building, conversion, and audit

- Test COCO required fields, category IDs/names, duplicates, unknown references,
  duplicate filenames/stems, and missing or extra images.
- Test malformed, zero/negative, clipped, outside, and boundary bounding boxes,
  plus YOLO normalisation.
- Check empty-image labels, stale-label pruning, summaries, YAML paths, and class
  ordering.
- Check train/validation/test leakage by more than a claimed split. Consider
  filenames, content hashes, and source session/group identifiers.
- Broadleaf preparation must keep related captures in one split, seed results
  deterministically, map only intended categories, and prevent augmented or
  split versions of one source image leaking across splits.
- Sparse mixed datasets must enforce label presence, deterministic limiting,
  repeat/prefix uniqueness, clean output folders, and hardlink-to-copy fallback.
- Image splitting must preserve all pixels, including remainder rows/columns,
  and report unreadable images and failed writes.

### Metrics, evaluation, and training wrappers

- Counting metrics cover unequal lengths, empty collections, missing classes,
  overcount and undercount bias, MAE/RMSE, and zero-error cases.
- Dataset YAML resolution covers relative/absolute roots, list/mapping class
  names, malformed split paths, and missing labels.
- Evaluation pairs predictions with the correct labels and emits stable JSON/CSV
  schemas, including zero-detection cases.
- Training tests patch `YOLO.train` and verify forwarded arguments,
  deterministic settings, optional hyperparameters, metadata, missing datasets,
  and CPU/no-CUDA metadata. Do not run an epoch as a unit test.

### Scripts and compatibility paths

- Treat scripts with hard-coded paths or import-time configuration as high risk;
  prefer temporary paths exposed through arguments.
- Conversion and diagnostics must reject unknown classes and malformed labels
  rather than silently producing plausible output.
- Where maintained and legacy scripts overlap, test the shared contract or
  document intentional differences. Do not copy tests mechanically onto both.

## Test quality rules

- Name tests as behaviours, such as
  `test_rejects_annotation_for_unknown_image`, not `test_case_3`.
- Use Arrange-Act-Assert and one behavioural reason to fail per test.
- Use `pytest.mark.parametrize` for compact input matrices when failures remain
  readable. Existing unittest tests may use `subTest`.
- Seed randomness and assert deterministic output where promised.
- Do not use sleeps, internet services, the public Streamlit site, or mutable
  shared state in automated tests.
- Do not depend on execution order. Clean up files even after assertion failure.
- Do not use large local datasets as ordinary fixtures or commit generated
  images, runs, archives, or weights.
- A skipped test is not a pass. Every skip must say why and how to enable it.
- Do not lower assertions, remove cases, broadly catch exceptions, or skip a
  failure without identifying its cause.
- Coverage reveals gaps but is not the acceptance criterion. Strong branch and
  failure-mode tests are more valuable than line-only coverage.

Use coverage as a diagnostic when useful:

```powershell
python -m pytest --cov=plant_counter --cov-report=term-missing `
  -m "not slow and not model and not acceptance"
```

Do not impose or lower a numeric coverage threshold without team agreement.

## Completion report

At the end of a testing task, give the team a concise record of:

- scope and requirement or defect tested;
- files added or changed;
- exact commands and environment;
- passed, failed, errored, and skipped counts;
- any model/data artefact identities used;
- unresolved risks, untested paths, and recommended next test;
- whether production code was changed.

When asked for a persistent testing log, put factual evidence under
`docs/testing/` and commit it with the tests. Never invent manual observations,
client acceptance, screenshots, timings, metrics, or results.

## Definition of done

A test-writing task is complete only when tests target an explicit risk or
contract, new tests pass in the supported environment, the full fast suite has
run, no unrelated files or generated artefacts changed, and the completion
report states remaining limitations. Otherwise report the task as partial.
