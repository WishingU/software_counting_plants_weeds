\# Species Differentiation Testing Log



\## Overview



Added 10 automated tests for `count\_plants.py`, mainly focusing on species counting and command-line functionality. The YOLO model was mocked, so no real model weights were needed.



\## Testing Environment



\- Windows

\- Conda environment: `plant-count`

\- Python 3.11.15

\- Pytest 9.1.1



\## Tests



| Test | What was tested | Result |

|---|---|---|

| Single species | Counting one species | Passed |

| Multiple species | Counting different species in one image | Passed |

| Zero detections | Returning zero when no plants are detected | Passed |

| Missing image | Handling a missing image file | Passed |

| Missing weights | Handling missing model weights | Passed |

| Custom parameters | Passing custom confidence, IoU and device settings | Passed |

| Default parameters | Using the default prediction settings | Passed |

| Green enhancement | Passing the enhanced image to YOLO | Passed |

| CLI output | Displaying species counts and total count | Passed |

| CLI save | Creating the output directory and calling the save function | Passed |



\## Test Results



\*\*New tests:\*\*



`python -m pytest -v tests/test\_count\_plants.py`



Result: 10 passed in 2.27s.



\*\*Full test suite:\*\*



`python -m pytest -v -m "not slow and not model and not acceptance"`



Result: 49 passed in 2.63s.



No test failures or errors.



\## Limitations



\- Only mocked YOLO predictions were used. Real model accuracy was not tested.

\- The actual saving of annotated images was not tested, only the save function call.

\- Streamlit deployment was not tested.

\- Real-model testing can be done later when the model weights are available.



\## Files Changed



\- Added `tests/test\_count\_plants.py`

\- Added `docs/testing/species\_testing\_log.md`

\- No changes to the production code.



