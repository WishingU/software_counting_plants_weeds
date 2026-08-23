# Annotation Guide for Four Seedling Classes

## 1. Task definition

The annotation unit is one independently countable physical plant. The only
accepted classes are `wheat`, `wild oat`, `brome grass`, and `barley grass`.
Leaf count, tiller count, and the number of visible emergence points must not
replace plant count.

Species identity should primarily come from experimental plot records, sowing
records, or seed provenance. These grasses can look very similar in top-down
seedling photographs, where ligules, auricles, and attached seeds are often not
visible. If identity cannot be established reliably, assign `needs_review` and
do not guess a training label.

## 2. Bounding-box rules shared by every class

1. Include every visible leaf belonging to the same plant. Keep the box tight
   around the outermost visible pixels with approximately 2 px of padding.
2. Annotate overlapping plants separately. Follow leaf continuity towards each
   growth centre; do not merge a cluster into one box or split one plant into
   multiple boxes.
3. Annotate an occluded plant when it is still recognisable as an independent
   instance and approximately 20% or more is visible. Cover only visible plant
   material; do not infer the extent of hidden leaves.
4. Clip edge plants at the image boundary. Ignore them when less than roughly
   20% is visible or when a separate instance cannot be established.
5. Do not annotate dead leaves, labels, tray edges, shadows, stones, or
   non-target weeds.
6. A physical plant may have only one box. Same-class boxes with IoU above 0.95
   should be reviewed as possible duplicates.

## 3. Per-class identification rules

### wheat — COCO 1 / YOLO 0

- Scope: the `Triticum` wheat material used in this experiment.
- Primary evidence: plot or sowing provenance.
- Supporting cues: leaf blades commonly twist clockwise; blunt hairy auricles
  may be visible at a sufficiently developed stage.
- Do not identify wheat from green colour, leaf width, or plant size alone.

### wild oat — COCO 2 / YOLO 1

- Scope: the `Avena` material represented in this experiment.
- Primary evidence: plot or seed provenance.
- Supporting cues: seedling leaves commonly twist anticlockwise; plants have a
  large ligule and no auricles; leaves can be slightly hairy and bluish-green.
- Main confusion: brome grass. When collar or seed traits are not visible and
  provenance is unavailable, send the instance for review.

### brome grass — COCO 3 / YOLO 2

- Scope: the `Bromus` material represented in this experiment.
- Primary evidence: plot or seed provenance.
- Supporting cues: dull hairy leaves that may show reddish-purple stripes along
  the veins; a large ligule and no auricles.
- Main confusion: wild oat. Do not label brome solely because a plant appears
  broad-leaved or sprawling.

### barley grass — COCO 4 / YOLO 3

- Scope: the `Hordeum` material represented in this experiment.
- Primary evidence: plot or seed provenance.
- Supporting cues: narrow leaves tapering to a point, sometimes with soft
  hairs; relevant barley-grass species can have prominent auricles and a
  membranous ligule; attached seed remains can assist identification.
- Main confusion: cultivated barley and other seedling grasses. Pale colour
  alone is not sufficient evidence.

## 4. Review states

- `accepted`: class and instance boundary satisfy the rules.
- `needs_review`: the plant is a valid instance but species identity is not
  reliable.
- `ignored`: less than the minimum visible fraction is present, the instance
  cannot be separated, or it is outside the four target classes.

## 5. Quality-control procedure

1. Run `python -m plant_counter.audit`; correct every `error`, then inspect the
   `warning` rows.
2. Randomly review at least 10% of images in every annotation batch, plus all
   dense, occluded, and image-edge cases.
3. Have a second annotator review every `needs_review` instance. Exclude any
   unresolved species label from training.
4. Split data by complete photograph or acquisition batch. Never distribute
   crops from the same photograph across training and validation.
5. Run the audit again and retain its JSON and CSV outputs with the dataset
   version.

The former `data/processed/images/test` directory was a validation subset used for
diagnostic previews and was removed during cleanup. Any replacement test set
must be independent of both training and validation.

## 6. Identification references

- [GRDC: Managing Wild Oats](https://grdc.com.au/__data/assets/pdf_file/0036/384966/GRDC_ManWildOats_V05.pdf)
- [GRDC: Integrated Weed Management Manual](https://grdc.com.au/__data/assets/pdf_file/0029/47873/Integrated-weed-management-manual-section-6-profiles-of-common-weeds-of-cropping.pdf)
- [GRDC: Common Weeds of Grain Cropping](https://grdc.com.au/__data/assets/pdf_file/0033/399741/UTE_Guide_Weeds20_210X148_Dec21_screen-min.pdf)

Morphological features are supporting evidence only; they do not replace the
experimental provenance records.
