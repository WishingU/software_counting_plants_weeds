# Handover — Counting & Species ID work (venn branch)

Summary of my work so far, for whoever's picking this up.

## What's done

**Item 1 (counting, 45%) — complete and validated.**
- Data pipeline: `scripts/coco_to_yolo.py` converts the source COCO dataset into YOLO format, merging weed species into one `weed` class.
- Trained model: `runs/colab_50epoch/best.pt` (YOLOv8n, 640px, 50 epochs). This is what `count_plants.py` and `app.py` use by default.
- Validated accuracy against all 301 held-out test images: crop counts within 2 plants 92% of the time, weed 84%, combined 79%.
- Production settings: `conf=0.25`, `iou=0.3` — both were deliberately tuned, not defaults (see "Important gotchas" below).

**Item 2 (species ID, 35%) — in progress.**
- Data pipeline: `scripts/coco_to_yolo_species.py` — same idea but keeps all 4 species separate (wheat, wild oat, brome grass, barley grass) instead of merging.
- Trained models: `runs/colab_species_50epoch/best.pt` and the better one, `runs/colab_species_100epoch/best.pt` (precision 0.73, recall 0.68).
- Confusion matrix (`scripts/species_confusion_matrix.py`) shows wheat is reliably distinguished from weeds (~69%+ correct), but the three weed species get confused mainly with *each other*, not with wheat — makes sense, they're visually similar grasses. Missed detections (not finding a plant at all) are rare (~1-2%) across all species, so the weakness is specifically fine-grained species labeling, not detection itself.

## What's not done yet (my part)

- Was about to try a bigger model (`yolov8s` instead of `yolov8n`) for species ID to see if it improves further — stopped before starting this.
- `count_plants.py` and `app.py` still only output crop/weed — they haven't been updated to show per-species results using the species model. That's a real gap: the species model exists but isn't wired into the user-facing tools yet.

## Important gotchas — read before changing settings

1. **The original dataset has real labeling gaps.** A systematic audit (70 randomly sampled "false positive" detections, manually zoomed in and checked) found almost all were real plants the annotators missed, not model errors. So raw precision numbers understate true accuracy — see `scripts/audit_false_positives.py` / `scripts/audit_threshold_gap.py` for methodology if you want to re-verify or extend this.
2. **Don't raise the confidence threshold without re-checking this.** I initially raised it from 0.25 to 0.35 to improve a precision metric, but the audit above showed this was discarding real plant detections, not just noise — it made actual counting accuracy worse (pushed the model into undercounting). Reverted back to 0.25 for that reason.
3. Evaluation scripts to use for checking accuracy: `scripts/evaluate_counts.py` (counting error), `scripts/evaluate_counts_corrected.py` (same, adjusted for the known labeling gaps), `scripts/evaluate_plant_detection.py` (class-agnostic precision/recall with threshold sweep).

## Other team context worth knowing

- `afridi-deployment` branch already has a working Streamlit Cloud deployment with a documented test URL — worth checking that before doing any deployment work yourself.
- `feature/deployment` branch has a significant restructuring in progress (proper `src/plant_counter/` package, tests, an annotation guide at `docs/ANNOTATION_GUIDE.en.md`, multi-language READMEs). Worth checking there before duplicating structure/doc work.
