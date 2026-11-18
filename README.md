# Event-Time Alignment

**Understanding Time, Aligning Attention: Temporal Reasoning for Web Forecasting**

The ACM Web Conference (WWW) 2027 · Under review

[Project page](https://joss0623.github.io/projects/event-time/) · [Research homepage](https://joss0623.github.io/)

[Paper](paper/manuscript.pdf)

## Overview

Connect public event dates to individual forecast days through event-relative representations and history-conditioned corrections of frozen numerical forecasts.

![Method overview](assets/method.png)

## Research focus

- Temporal reasoning
- Event-time alignment
- Web attention

## Implementation

The `code/` directory contains the available research implementation snapshot. Main entry points:

- [`code/collect_dated_events.py`](code/collect_dated_events.py)
- [`code/conditional_correction.py`](code/conditional_correction.py)
- [`code/neural_readouts.py`](code/neural_readouts.py)

Framework modules, dataset readers, and supporting experiments are kept with the model. Prepare the required datasets and pretrained language-model weights separately; these are not bundled with the code. Dataset locations and experiment defaults in the original scripts need to be set for your environment.

Inspect the available training arguments before configuring your dataset:

```bash
cd code
python run.py --help
```

## Snapshot and verification

`release-manifest.json` records the SHA-256 of each released source file. This release was checked for Python syntax and obvious embedded credentials; it has not been retrained or certified to reproduce every paper table on a new machine.

## Acknowledgments and license

The implementation builds on existing forecasting research libraries. Original license notices are retained inside `code/`. Upstream sources include [Time-Series-Library](https://github.com/thuml/Time-Series-Library). The paper and figures retain their respective authors' rights.
