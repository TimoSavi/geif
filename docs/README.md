# GEIF Documentation Index

**Author / Maintainer:** Timo Savinen (AI-assisted)

This directory contains the core technical documentation, algorithmic specifications, engineering manuals, and empirical analyses for **GEIF (Geometric Extended Isolation Forest)**.

---

## Technical Documentation Suite

| Document | Topic | Implementation Mapping |
| :--- | :--- | :--- |
| [algorithm.md](algorithm.md) | Algorithmic and mathematical specification of GEIF, common foundational primitives (Zero Kelvin, asymptotic outer decay, nearest bounding-box leaf calculations, scale invariance), and detailed chapters for all 4 algorithm engines (`bubble`, `voronoi`, `exemplar`, `ceif`). | [`include/geif/geif.h`](../include/geif/geif.h), [`src/lib/`](../src/lib/) |
| [api_reference.md](api_reference.md) | Comprehensive C17 API and function reference manual extracted directly from in-code Doxygen documentation across all public headers, core engine modules, and CLI utilities. | [`include/geif/`](../include/geif/), [`src/lib/`](../src/lib/), [`src/cli/`](../src/cli/) |
| [implementation.md](implementation.md) | Modern C17 software architecture, pluggable algorithm vtable (`geif_algo_ops_t`), memory layout, sparse JSON serialization, configuration parser, and CLI streaming pipelines. | [`src/lib/`](../src/lib/), [`src/cli/`](../src/cli/) |
| [algorithm_selection.md](algorithm_selection.md) | Empirical benchmark results, visual topology evaluations, and operational selection matrices across all four engines (outlier detection, population drift, categorization, performance). | [`pics/recom/`](pics/recom/) |
| [manual.md](manual.md) | Comprehensive CLI reference manual covering all operational modes, options, format templates, configuration files, and UNIX pipeline patterns. | [`src/cli/main.c`](../src/cli/main.c) |
| [build.md](build.md) | Build instructions, compiler flags, prerequisites, target definitions, and automated testing suite overview. | [`Makefile`](../Makefile) |

---

> [!NOTE]
> **Research & Development Monographs:** Specialized historical research monographs and edge-case investigation reports (e.g. `eif-algorithmic-evolution.md`, `outer-space-geometry.md`, `topological-cavities-and-the-donut-hole.md`) are archived in the user documentation directory (`~/docs/`).
