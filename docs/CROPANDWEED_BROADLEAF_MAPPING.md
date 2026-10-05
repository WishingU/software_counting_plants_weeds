# CropAndWeed to `broadleaf_weed` mapping

This document records the source classes used to build the one-class
`broadleaf_weed` object-detection dataset. The mapping is derived from the
official CropAndWeed `CropsOrWeed9` mapping: classes assigned to `Weed` are
retained, then every class in the official Fine24 `Grasses` group is removed.

Crop classes are not positive labels. They remain visible in the images as
hard negatives. Likewise, grass weeds remain as hard negatives. Original
class 255 (`Vegetation`) is excluded because it represents tiny, excluded, or
ambiguous vegetation rather than a confirmed broadleaf-weed instance.

The counts below describe the downloaded official release before splitting.
The generated dataset also contains `mapping_summary.json` and
`split_manifest.csv` with the exact post-conversion and per-split counts.

## Classes mapped to `broadleaf_weed`

| Source ID | CropAndWeed name | Instances | Images |
|---:|---|---:|---:|
| 22 | Poppy | 25 | 2 |
| 29 | Hybrid goosefoot | 329 | 205 |
| 30 | Black-bindweed | 195 | 45 |
| 32 | Red-root amaranth | 3,484 | 491 |
| 33 | White goosefoot | 3,180 | 1,121 |
| 34 | Thorn apple | 469 | 167 |
| 35 | Potato weed | 2,563 | 636 |
| 36 | German chamomile | 2,113 | 300 |
| 37 | Saltbush | 0 | 0 |
| 38 | Creeping thistle | 2,693 | 410 |
| 39 | Field milk thistle | 2,623 | 513 |
| 41 | Black nightshade | 14 | 9 |
| 42 | Mercuries | 1,147 | 812 |
| 44 | Pale persicaria | 0 | 0 |
| 45 | Geraniums | 667 | 90 |
| 47 | Whitetop | 0 | 0 |
| 49 | Frosted orach | 1 | 1 |
| 50 | Black horehound | 1 | 1 |
| 51 | Shepherds purse | 8 | 5 |
| 52 | Field bindweed | 533 | 161 |
| 54 | Hedge mustard | 29 | 22 |
| 56 | Speedwell | 108 | 53 |
| 57 | Broadleaf plantain | 12 | 12 |
| 58 | White ball-mustard | 2 | 1 |
| 59 | Peppermint | 602 | 84 |
| 60 | Field pennycress | 118 | 79 |
| 61 | Corn spurry | 922 | 125 |
| 63 | Common fumitory | 227 | 73 |
| 64 | Ivy-leaved speedwell | 237 | 60 |
| 66 | Redshank | 428 | 129 |
| 67 | Common hemp-nettle | 0 | 0 |
| 70 | Small geranium | 4,428 | 406 |
| 71 | Cornflower | 1,780 | 270 |
| 72 | Common corn-cockle | 1,601 | 211 |
| 76 | Purple dead-nettle | 47 | 39 |
| 77 | Ribwort plantain | 943 | 100 |
| 78 | Pineappleweed | 608 | 69 |
| 79 | Common chickweed | 219 | 86 |
| 80 | Hedge mustard | 917 | 121 |
| 83 | Yellow rocket | 379 | 81 |
| 85 | Red poppy | 630 | 39 |
| 87 | Knotgrass | 408 | 67 |
| 88 | Prickly lettuce | 5 | 5 |
| 89 | Copse-bindweed | 1,790 | 858 |
| 91 | Common buckwheat | 211 | 83 |
| 96 | Field mustard | 402 | 49 |

Total in the official release: **37,098 mapped rows in 4,955 positive
images**. One mapped row is an exact duplicate, so the generated YOLO dataset
contains **37,097 unique instances**. Some images contain more than one source
class.

## Weed classes deliberately excluded as grasses

These official weed classes are monocot grasses and are therefore negative
examples for the broadleaf detector:

| Source ID | CropAndWeed name |
|---:|---|
| 31 | Cockspur grass |
| 48 | Meadow-grass |
| 62 | Purple crabgrass |
| 65 | Annual meadow grass |
| 68 | Rough meadow-grass |
| 69 | Green bristlegrass |
| 74 | Wall barley |
| 75 | Annual fescue |
| 81 | Soft brome |
| 84 | Common wild oat |
| 86 | Rye brome |

## Crop classes deliberately excluded

The official crop IDs for maize (1–6), sugar beet (7–12), pea (13), pumpkin
(15), potato (18), sunflower (24), common bean (26), faba bean (27), soybean
(94), and their growth-stage variants are not mapped to `broadleaf_weed`.
They are retained in images as hard negatives so the model learns that
“broadleaf” alone does not imply “broadleaf weed.”

## Reproducibility rules

- Every downloaded image is retained, including images with no positive
  broadleaf label; those images become explicit negative samples.
- Splits are assigned by acquisition session (for example, `ave-0357`), not
  by individual image, preventing near-duplicate frames from crossing splits.
- Image files are hard-linked when possible to avoid duplicating roughly
  10 GB of source imagery. Labels and manifests are newly generated.
- The mapping implemented in
  `scripts/prepare_cropandweed_broadleaf.py` is authoritative for conversion.
