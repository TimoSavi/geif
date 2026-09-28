# GEIF Production Readiness Implementation Plan

**Author / Maintainer:** Timo Savinen (AI-assisted)

This plan details the phased enhancements required to bring **GEIF (Geometric Extended Isolation Forest)** from proof-of-concept to production-grade parity with `ceif`, directly supporting all production cron recipes, CLI workflows, and rcfile configurations defined in [`plan.md`](file:///home/timo_savinen_elisa_fi/docs/tools/geif/prod_plan/plan.md).

---

## Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │              geif CLI                │
                      │  - Delimiter / Header (-f, -e, -H)   │
                      │  - Column Filters (-I, -U, -L)       │
                      │  - Formatter & Templating (-p, -v)   │
                      │  - RC Configuration (-g, ~/.geifrc)  │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │       Multi-Category Manager         │
                      │       (geif_category_map_t)          │
                      │  - Category Sub-Forests (-C)         │
                      │  - Regex Filters (-F, -v)            │
                      │  - Missing / New Alerts (-M, -N)     │
                      │  - Age-based Pruning (-D, -R)        │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │             libgeif Core             │
                      │  - Voronoi Midpoint Bisectors        │
                      │  - Continuous Metric Depth           │
                      │  - Cauchy-Lorentz Cavity Damping     │
                      │  - Euclidean Stadium Metric          │
                      │  - Single Dimension Impact (%e)      │
                      │  - JSON Serialization (GEIF-1.0)     │
                      └──────────────────────────────────────┘
```

---

## Detailed Implementation Tasks

### Phase 1: Core CLI Pipeline, Delimiters & Column Masks
- [x] **Feature 1: Delimiter & Header Handling (`-f`, `-e`, `-H`, stdin `-`)**
  - *Reasoning:* Production streams use varying delimiters (`;`, `,`, `\t`) and optional header rows.
  - *Options:* `-f <char>` (input field separator), `-H` (skip header line), `-e <char>` (list separator), support `-` for stdin/stdout.
  - *Test Case:* Ingest `winequality-red.csv` (semicolon delimited with header).
- [ ] **Feature 2: Column Selection & Row Labeling (`-I`, `-U`, `-L`)**
  - *Reasoning:* Telemetry streams include row IDs, timestamps, and irrelevant dimensions that must not enter geometric training.
  - *Options:* `-I <list>` (ignore column indices/ranges, e.g. `1,4-6`), `-U <list>` (explicitly use column indices), `-L <list>` (label columns stored for output `%l` but excluded from training).
  - *Test Case:* Train on wine dataset using column 1 as label, ignoring column 2.

### Phase 2: Category Management & Multi-Tenant Partitioning
- [ ] **Feature 3: Multi-Category Sub-Forest Engine (`-C <list>`)**
  - *Reasoning:* Production models partition telemetry by tenant, host, or rating (e.g. wine `quality` column 12).
  - *Options:* `-C <col_range>` (extract category key, e.g. `-C 12` or `-C 1-3`). Manage an array/hash of independent `geif_forest_t` sub-forests. Save/load complete multi-forest models in `GEIF-1.0` JSON.
  - *Test Case:* Train multi-category forest on `winequality-red.csv` grouped by `quality` (ratings 3–8). Verify queries evaluate against corresponding quality sub-forests.
- [ ] **Feature 4: Category Filtering, Tracking & Lifecycle (`-F`, `-M`, `-N`, `-R`, `-D`)**
  - *Reasoning:* Operational pipelines filter specific categories (`-F`), detect dropped/unseen categories (`-M`), alert on unseen categories (`-N`), clear categories (`-R`), and prune stale data older than $N$ days (`-D`).
  - *Options:* `-F <regex>` (supports multiple filters and `"-v ^pattern"` inversion), `-M [format]` (report missing categories), `-N [format]` (report unknown categories), `-R <cat>` (reset category), `-D <interval>` (drop categories with `last_updated` older than e.g. `30d`, `7d`).
  - *Test Case:* Evaluate filtered categories and simulate missing/new category alerts on wine batches.

### Phase 3: Thresholds, Output Templating & Feature Attribution
- [ ] **Feature 5: Outlier Score Thresholds & Aliases (`-O`, `-t`, `-s`, `-d`, `-S`)**
  - *Reasoning:* Support muscle-memory flags and flexible threshold types without breaking existing shell scripts.
  - *Options:* `-t <int>` (alias for `-i` trees), `-O <val>` (alias for `-T` threshold; supports float `0.6`, scaled `0.65s`, and percentile `80%`), `-d <int>` (decimal precision), `-S` (silent/locale).
  - *Test Case:* Execute queries with `-O 80%` and `-O 0.65s`.
- [ ] **Feature 6: Custom Output Templating (`-p`, `-v`, `-m`)**
  - *Reasoning:* UNIX pipelines pipe GEIF into `sort`, `tee`, and mail generators (`ffe`).
  - *Options:* `-p <format>` with directives: `%s` (score), `%l` (label), `%c` / `%C` (category), `%m` (dimension list), `%d` (dimension values), `%a` (category averages), `%v` (raw row), `%x` (RGB hex color). Support `-v <format>` summary and `-m <printf>` numeric formatting.
  - *Test Case:* Output format `A;%s;%C;%m;%a` matching production cron recipes.
- [ ] **Feature 7: Single Dimension Impact / Feature Attribution (`%e`, `-G`)**
  - *Reasoning:* Users need to know *which specific dimension* caused a data point to be flagged as anomalous, without the overhead of k-means clustering.
  - *Options:* Calculate marginal anomaly contribution $\Delta \text{score}_j$ by evaluating counterfactual displacement against the category median/centroid. Expose via `%e` in `-p` and `-m`, and `-G <dims>` for joint dimension scoring.
  - *Test Case:* Verify highest-impact feature on anomalous high-alcohol or high-volatile-acidity wine samples.
- [ ] **Feature 8: Population Drift & Test Grid Generation (`-T <margin>`, `-i <res>`)**
  - *Reasoning:* Humans cannot easily detect population drift from numbers alone; synthetic grid evaluation enables 2D/ND drift boundary visualization.
  - *Options:* `-T <margin>` (generate synthetic grid over data envelope), `-i <res>` (grid resolution), outputting `%d,0x%x` RGB maps for gnuplot.
  - *Test Case:* Generate 2D test grid on wine dimensions and render decision boundaries.

### Phase 4: Configuration & Model Migration
- [ ] **Feature 9: RC Configuration File (`~/.ceifrc` / `~/.geifrc` / `-g <file>`)**
  - *Reasoning:* Production environments set global defaults (`TREES`, `MAX_SAMPLES`, `OUTLIER_SCORE`, `LOW_RGB_COLOR`, `HIGH_RGB_COLOR`, `PRINT_DIMENSION`).
  - *Options:* Parse rcfile directives, override via CLI options.
  - *Test Case:* Load sample rcfile and verify defaults apply to training/inference.
- [ ] **Feature 10: CEIF to GEIF Model Migration Tool (`ceif2geif`)**
  - *Reasoning:* Seamless migration of existing `.ceif` model archives into modern `GEIF-1.0` JSON models.
  - *Options:* Converter that reads `.ceif` JSON (sample reservoirs per category) and builds calibrated `GEIF-1.0` geometric tree ensembles.
  - *Test Case:* Migrate existing `.ceif` file and verify scoring fidelity.

### Phase 5: Verification, Production Testing & Documentation
- [ ] **Feature 11: End-to-End Production Cron Validation**
  - *Reasoning:* Re-run all 15 production shell pipeline commands from `plan.md` using `winequality-red.csv` and `winequality-white.csv`.
  - *Verification:* Zero crashes, identical exit codes (0 = no outlier, 2 = outliers detected), correct pipeline output formats.
- [ ] **Feature 12: Documentation Restructuring for Public Release**
  - *Reasoning:* Organize `git/geif/docs/` for clean public distribution, archiving research discussions to `~/docs/`.

---

## Working Process (Per Feature)

For each feature in sequence:
1. **Rationale & Design:** Explain the mathematical and systems engineering rationale.
2. **Implementation:** Write clean, modular C17 code conforming to `libgeif` standards.
3. **Automated Test:** Implement a dedicated test case validating the feature.
4. **Git Diff Review:** Present the diff for human architectural review and commit.
