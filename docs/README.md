# GEIF Documentation Index

**Author / Maintainer:** Timo Savinen (AI-assisted)

This directory contains the core technical documentation, algorithmic specifications, engineering manuals, and empirical analyses for **GEIF (Geometric Extended Isolation Forest)**.

---

## Technical Documentation Suite

| Document | Topic | Implementation Mapping |
| :--- | :--- | :--- |
| [algorithm.md](algorithm.md) | Algorithmic and mathematical specification of GEIF, common foundational primitives (Zero Kelvin, asymptotic outer decay, nearest bounding-box leaf calculations, scale invariance), and detailed chapters for all 4 algorithm engines (`bubble`, `voronoi`, `exemplar`, `ceif`). | [`include/geif/geif.h`](file:///home/timo_savinen_elisa_fi/git/geif/include/geif/geif.h), [`src/lib/algo*.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/) |
| [implementation.md](implementation.md) | Modern C17 software architecture, pluggable algorithm vtable (`geif_algo_ops_t`), memory layout, sparse JSON serialization, configuration parser, and CLI streaming pipelines. | [`src/lib/`](file:///home/timo_savinen_elisa_fi/git/geif/src/lib/), [`src/cli/`](file:///home/timo_savinen_elisa_fi/git/geif/src/cli/) |
| [heatmaps.md](heatmaps.md) | Empirical decision manifolds and score distributions across 6 topological benchmarks for all 4 algorithm engines, with in-depth analysis of cavity carving, aspect ratio invariance, and stadium metrics. | [`pics/geif/`](pics/geif/) |
| [manual.md](manual.md) | Comprehensive CLI reference manual covering all operational modes, options, format templates, configuration files, and UNIX pipeline patterns. | [`src/cli/main.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/cli/main.c) |
| [build.md](build.md) | Build instructions, compiler flags, prerequisites, target definitions, and automated testing suite overview. | [`Makefile`](file:///home/timo_savinen_elisa_fi/git/geif/Makefile) |

---

> [!NOTE]
> **Research & Development Monographs:** Specialized historical research monographs and edge-case investigation reports (e.g. `eif-algorithmic-evolution.md`, `outer-space-geometry.md`, `topological-cavities-and-the-donut-hole.md`) are archived in the user documentation directory (`~/docs/`).
