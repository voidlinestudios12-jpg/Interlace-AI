# Technical report

The 2026 technical report (TR-2026-01, `10.5281/zenodo.21936832`) described
measurements produced by version 1.0.0 of this library.

Several of those measurements are not reproducible with 1.1.0, because 1.0.0
contained defects in answer extraction and in the handling of truncated
trajectories that corrupted the inputs to the vote. The details are in
[CHANGELOG.md](CHANGELOG.md).

**A revised report is being prepared.** Until it is published, the numbers to
rely on are the ones in [README.md](README.md), each of which can be
recomputed from the trajectories in `results/` by running
`python scripts/analyse.py`.
