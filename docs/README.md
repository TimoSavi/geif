# GEIF Documentation & Architecture Index

**Author / Maintainer:** Timo Savinen (AI-assisted)

This directory contains the mathematical specifications, engineering design documents, and algorithmic edge-case resolutions established for **GEIF (Geometric Extended Isolation Forest)**.

---

## Specification Documents & Code Mapping

| Document | Topic | Code Implementation in `git/geif` |
|---|---|---|
| [eif-algorithmic-evolution.md](eif-algorithmic-evolution.md) | Fundamental algorithmic evolution from classic EIF to CEIF and GEIF, empirical Voronoi findings, and next-generation geometric forest architectures | Core theory & research foundation |
| [algorithm.md](algorithm.md) | Formal mathematical specification of GEIF, Voronoi bisectors, and scoring semantics | [`include/geif/geif.h`](file:///home/timo_savinen_elisa_fi/git/geif/include/geif/geif.h), [`src/lib/geometry.h`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/geometry.h) |
| [implementation.md](implementation.md) | Modern C17 architecture, API design, CLI UNIX piping, and memory layout | [`src/cli/main.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/cli/main.c), [`src/lib/forest.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/forest.c) |
| [heatmaps.md](heatmaps.md) | Visual anomaly score heatmaps across 6 benchmark datasets and CEIF comparison | [`src/lib/evaluate.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/evaluate.c), [`pics/`](pics/) |
| [compatibility.md](compatibility.md) | Compatibility analysis between GEIF and CEIF (CLI options, rcfile, model format) | [`src/cli/main.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/cli/main.c), [`src/lib/json_io.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/json_io.c) |
| [algorithmic-challenges-and-solutions.md](algorithmic-challenges-and-solutions.md) | Overview of the core mathematical subtleties, pitfalls, and solutions | [`src/lib/train.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/train.c), [`src/lib/evaluate.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/evaluate.c) |
| [dimension-scaling-and-aspect-ratios.md](dimension-scaling-and-aspect-ratios.md) | Scale invariance across extreme aspect ratios (5000:1) via midpoint bisectors | [`src/lib/geometry.h`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/geometry.h), [`test/test_voronoi.c`](file:///home/timo_savinen_elisa_fi/git/geif/test/test_voronoi.c) |
| [outer-space-geometry.md](outer-space-geometry.md) | Smooth Euclidean "Stadium" distance outside bounding boxes ($d_{\text{out}} > 0$) | [`src/lib/geometry.h`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/geometry.h), [`test/test_stadium.c`](file:///home/timo_savinen_elisa_fi/git/geif/test/test_stadium.c) |
| [zero-variance-and-constant-features.md](zero-variance-and-constant-features.md) | Dimension health masking and handling of zero-variance columns ($\text{span}_j = 0$) | [`src/lib/forest.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/forest.c), [`src/lib/train.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/train.c) |
| [duplicate-samples-and-inseparable-leaves.md](duplicate-samples-and-inseparable-leaves.md) | Pair sampling retries and analytic completion bonus for identical leaf points | [`src/lib/train.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/train.c), [`src/lib/evaluate.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/evaluate.c) |
| [topological-cavities-and-the-donut-hole.md](topological-cavities-and-the-donut-hole.md) | Resolving hollow non-convex structures with Cauchy-Lorentz residual cell damping | [`src/lib/evaluate.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/evaluate.c) |
| [continuous-metric-depth-and-density-bias.md](continuous-metric-depth-and-density-bias.md) | Replacing discrete hop counts with continuous metric depth $\Delta H = 1/\delta$ | [`src/lib/train.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/train.c), [`src/lib/evaluate.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/evaluate.c) |
| [reservoir-sampling-and-streaming-ingest.md](reservoir-sampling-and-streaming-ingest.md) | Algorithm R streaming reservoir sampling ($N_{\text{pool}} = T \times \psi$) | [`src/lib/reservoir.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/reservoir.c) |
| [tree-depth-control-and-ensemble-diversity.md](tree-depth-control-and-ensemble-diversity.md) | Depth caps, subsampling diversity, and ensemble convergence | [`include/geif/types.h`](file:///home/timo_savinen_elisa_fi/git/geif/include/geif/types.h), [`src/lib/train.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/train.c) |
| [text-embeddings-and-semantic-geometry.md](text-embeddings-and-semantic-geometry.md) | High-dimensional semantic vectors and local AI embedding caching considerations | Architecture reference for future extensions |

---

## Implementation Summary in `git/geif`

The complete implementation has been built, tested, and validated in `git/geif`:
- **Build System**: High-performance C17 Makefile producing `lib/libgeif.a`, `lib/libgeif.so`, and `bin/geif`.
- **Unit Tests**:
  - `test_voronoi`: Verifies hyperplane normal calculations, 5000:1 aspect ratio invariance, duplicate point rejection, and zero-variance feature masking.
  - `test_stadium`: Verifies Euclidean distance calculations to envelope bounding boxes, including rounded corner arcs and perpendicular excursions.
- **CLI Capabilities**:
  - Ingestion and training with `-l <file.csv> -w <model.json>`
  - Streaming analysis with `-r <model.json> -a <test.csv> -o <out.csv> -T <threshold>`
  - Model inspection and diagnostics with `-r <model.json> -q`
- **Validation**: Trained and tested on `complex2d.csv` (1,465 samples):
  - Calibrated Universal Scale $H_{\text{max}} = 36.287253$
  - Outliers identified: 15 (1.02% at $T = 0.50$)
  - Nominal inlier score range: 0.15 – 0.35
