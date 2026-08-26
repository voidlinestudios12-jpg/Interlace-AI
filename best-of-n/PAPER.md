# Technical report

**[TR-2026-02](TR-2026-02.md) — *Best-of-N: Inference-Time Compute for Small
Language Models*** is the current report, and the full text is in this
repository. It documents the measurements in [README.md](README.md), every one
of which is re-derived from the published trajectories by
`scripts/analyse.py`.

The deposited version, with the trajectory dataset attached, is on Zenodo:
[10.5281/zenodo.21936832](https://doi.org/10.5281/zenodo.21936832).

It supersedes TR-2026-01 (`10.5281/zenodo.21936833`). That earlier report
described measurements produced by version 1.0.0 of this library, which
contained defects in answer extraction and in the handling of truncated
trajectories that corrupted the inputs to the vote. Those numbers are not
reproducible with the corrected implementation and should not be cited. The
full list of corrections is in [CHANGELOG.md](CHANGELOG.md).

Also withdrawn from TR-2026-01: the claim of a trained outcome reward model
reaching ROC-AUC 0.910 and a 16.6-point gain over majority voting. That model
was never published, and in the one evidence file that was, it selected the
same answer as majority voting on every problem.

The present release ships no reward model. It provides adapters for
third-party ones.
