# GEIF — Geometric Extended Isolation Forest

**Author / Maintainer:** Timo Savinen (AI-assisted)

> [!NOTE]
> **AI-Assisted Development:** GEIF was designed and engineered by Timo Savinen with AI pair-programming assistance (Google Antigravity / Gemini) under human architectural direction. The mathematical specifications, C17 core library, CLI Unix pipelines, test suites, and architectural documentation were developed collaboratively.

**GEIF** (Geometric Extended Isolation Forest) is an industrial-strength, high-performance ISO C17 implementation of a geometric, scale-invariant anomaly detection engine. Building upon and refining the foundational concepts of Isolation Forest (iForest) and Extended Isolation Forest (EIF), GEIF introduces a modular multi-algorithm engine—featuring native default hyperspherical Bubble trees, Voronoi bisectors, continuous Gaussian cuts, and non-tree Exemplar Cauchy kernels—alongside continuous Euclidean envelope distance ("Stadium" metric), multi-category ensemble routing, single-dimension attribution, streaming age decay, and universal Zero Kelvin scale calibration.

> [!NOTE]
> **Lineage & Architectural Influence:** GEIF builds upon the architectural insights, streaming designs, and practical production experience established in [CEIF (Continuous Extended Isolation Forest)](https://github.com/TimoSavi/ceif), also created by Timo Savinen. GEIF modernizes and expands these foundations into a fully modular geometric engine with sparse JSON serialization and comprehensive multi-dimensional void detection.

---

## Table of Contents

- [Key Innovations & Mathematical Foundation](#key-innovations--mathematical-foundation)
  - [1. Multi-Algorithm Geometric Partitioning Engines](#1-multi-algorithm-geometric-partitioning-engines)
  - [2. Asymptotic Exponential Outer Space Attenuation ("Stadium Metric")](#2-asymptotic-exponential-outer-space-attenuation-stadium-metric)
  - [3. Structural "Zero Kelvin" Universal Scale Calibration](#3-structural-zero-kelvin-universal-scale-calibration)
  - [4. Streaming Reservoir Sampling & Age Decay](#4-streaming-reservoir-sampling--age-decay)
- [Architecture & Directory Structure](#architecture--directory-structure)
- [Building & Testing](#building--testing)
  - [Requirements](#requirements)
  - [Build Targets](#build-targets)
  - [Dual Test Suites](#dual-test-suites)
- [CLI Reference Manual](#cli-reference-manual)
  - [Command-Line Options](#command-line-options)
  - [Format Template Tokens](#format-template-tokens)
  - [RC Configuration File (`~/.geifrc` / `~/.ceifrc`)](#rc-configuration-file-geifrc--ceifrc)
  - [Exit Code Semantics](#exit-code-semantics)
- [CEIF to GEIF Migration (`ceif2geif`)](#ceif-to-geif-migration-ceif2geif)
- [Production Usage Patterns](#production-usage-patterns)
  - [1. Batch & Streaming Training](#1-batch--streaming-training)
  - [2. Stream Scoring with Filtering & Sorting](#2-stream-scoring-with-filtering--sorting)
  - [3. Multi-Category Classification & Routing (`-c`)](#3-multi-category-classification--routing--c)
  - [4. Rolling Model Updates with Age Decay](#4-rolling-model-updates-with-age-decay)
  - [5. Reservoir Outlier Pruning & Recalibration](#5-reservoir-outlier-pruning--recalibration)
  - [6. Visualizing Population Drift with Test Grids](#6-visualizing-population-drift-with-test-grids)
- [C17 Library API](#c17-library-api)
- [Documentation & Deep Dives](#documentation--deep-dives)
- [Development & Attribution](#development--attribution)
- [License & Dataset Attribution](#license--dataset-attribution)

---

## Key Innovations & Mathematical Foundation

### 1. Multi-Algorithm Geometric Partitioning Engines

Standard Isolation Forests make axis-aligned cuts, producing unnatural rectangular artifacts. Standard EIF draws arbitrary random normal vectors from uniform spherical distributions, which fail when dimensions possess radically disparate physical scales.

GEIF provides four modular geometric partitioning engines through a polymorphic dispatch table (`geif_algo_ops_t`), selectable at training time with `-B, --algo <name>`:

#### Engine 1: Hyperspherical Bubble Trees (`bubble` — Native Default)
The primary workhorse of GEIF. Space is partitioned using radial hyperspheres rather than hyperplanes:

$$
d^2(x, c) \le R^2 \implies \text{interior (left)}, \quad d^2(x, c) > R^2 \implies \text{exterior (right)}
$$

- **Cavity & Topological Void Detection**: Linear hyperplanes cannot isolate the center of a donut, ring, or concave cluster without slicing through dense clusters. Bubble trees isolate internal voids naturally: when a hyperspherical shell contains no training samples, it terminates into an empty void leaf ($N=0$), instantly identifying non-convex cavities as anomalies.
- **$O(N)$ Hoare Quickselect Partitioning**: At each node, the median squared radius $R^2$ is computed via Quickselect directly on squared Euclidean distances, eliminating expensive `sqrt()` operations and sorting during training.
- **In-Place Two-Pointer Partitioning**: Partitions sample indices in-place without per-node scratch memory allocations.
- **Stack-Allocated Center Geometry**: Centers in $D \le 64$ dimensions use stack arrays, eliminating heap allocation bottlenecks.

#### Engine 2: Data-Adaptive Voronoi Bisectors (`voronoi`)
Hyperplanes placed at the exact perpendicular bisector between two distinct randomly sampled points $A$ and $B$:

$$
P_{\text{mid}} = \frac{A + B}{2}, \quad \vec{n} = \frac{B - A}{\Vert B - A \Vert}
$$

- **Scale Invariant**: Intrinsically adapts to local cluster geometry and extreme aspect ratios (up to $5000:1$) without requiring manual normalization.
- **Zero-Variance Feature Masking**: Detects constant features and restricts cuts strictly to informative dimensions.

#### Engine 3: Direct SIMD Exemplar Cauchy Density (`exemplar`)
A non-tree kernel density estimation engine operating directly over reservoir samples:

$$
D(x) = \frac{1}{K}\sum_{i=1}^K \frac{1}{1 + (d_i / \sigma_i)^2}, \quad s(x) = 1.0 - D(x)
$$

- Precomputes adaptive local bandwidths $\sigma_i$ from $K$-nearest neighbor distances.
- Highly effective for dense cluster manifolds where geometric trees are not desired.

#### Engine 4: Continuous Gaussian Cuts (`ceif`)
Anchored hyperplanes with pairwise margin interpolation and isotropic Gaussian normal vectors, delivering smooth continuous metric depth surfaces.

---

### 2. Asymptotic Exponential Outer Space Attenuation ("Stadium Metric")

When an observation falls outside the training data bounding envelope ($[\min_j, \max_j]$), conventional decision trees cannot differentiate between a point slightly outside the box and a point far in outer space.

GEIF computes the normalized exterior Euclidean distance $d_{\text{norm}}$:

$$
d_{\text{norm}} = \sqrt{\sum_{j=1}^D \left(\frac{\max(0, \min_j - x_j) + \max(0, x_j - \max_j)}{\text{span}_j}\right)^2}
$$

An asymptotic exponential decay pulls the score smoothly toward 1.0 without sharp cliffs or rectangular plateaus:

$$
s = 1.0 - (1.0 - s_0) \cdot e^{-\beta \cdot d_{\text{norm}}}
$$

where $s_0 = 2^{-H / c}$ is the raw ensemble tree anomaly score and $\beta = 0.10$ (`GEIF_OUTER_DECAY_RATE`). Bending begins smoothly around $0.60 \dots 0.70$ near data boundaries and exponentially approaches 1.0 in outer space. Scores are strictly bounded below 1.0 ($s \le 1.0 - 10^{-6}$) to preserve dynamic range.

---

### 3. Structural "Zero Kelvin" Universal Scale Calibration

Anomaly scores in GEIF are calibrated using the **Zero Kelvin principle**:
- GEIF traverses all trees in memory via `geif_tree_find_max_height()`, locating the theoretical deepest path:

$$
H_{\text{leaf}} = \text{depth} + c\left(\frac{\text{sample-count}}{\text{MIN-REL-DIST}}\right)
$$

- Averaging across all trees gives:

$$
\bar{H}_{\text{zero-kelvin}} = \frac{1}{T}\sum_{t=0}^{T-1} H_{\max}(t)
$$

- The baseline minimum score is calibrated directly:

$$
s_{\min} = \frac{1}{2^{\bar{H}_{\text{zero-kelvin}} / c}}
$$

- Output scores are calibrated to $[0.0, 1.0 - 10^{-6}]$ as the standard default:

$$
s_{\text{scaled}} = \frac{s - s_{\min}}{s_{\max} - s_{\min}}
$$

---

### 4. Streaming Reservoir Sampling & Age Decay

The training pool maintains a fixed maximum capacity:

$$
N_{\text{pool}} = N_{\text{trees}} \times N_{\text{samples}} \quad (\text{e.g., } 100 \times 256 = 25{,}600)
$$

Any streaming input of arbitrary length is ingested via Algorithm R reservoir sampling, guaranteeing an unbiased uniform sample even if the input stream is ordered or clustered.

When updating existing models over time, GEIF applies exponential age decay (`-D <rate>d`):

$$
P_{\text{retain}}(\Delta t) = \exp\left(-\frac{\Delta t}{\tau}\right)
$$

Stale reservoir samples are probabilistically replaced by incoming observations, allowing the model to adapt dynamically to evolving production environments.

---

## Architecture & Directory Structure

```text
geif/
├── include/
│   └── geif/
│       ├── geif.h               # Public C17 library API
│       ├── types.h              # Core data structures, enums, & configuration
│       └── error.h              # Status codes and error reporting
├── src/
│   ├── lib/                     # Core algorithmic library
│   │   ├── algo.h               # Internal algorithm dispatch vtable (geif_algo_ops_t)
│   │   ├── algo_registry.c      # Algorithm factory & registration
│   │   ├── algo_bubble.c        # Hyperspherical Bubble tree engine (default)
│   │   ├── algo_voronoi.c       # Data-adaptive Voronoi bisector engine
│   │   ├── algo_exemplar.c      # Direct Cauchy density kernel engine
│   │   ├── algo_ceif.c          # Continuous Gaussian hyperplane engine
│   │   ├── tree_common.c/.h     # Zero Kelvin calibration, scales, leaf relative distances
│   │   ├── forest.c             # Forest allocation, lifecycle, & scale tracking
│   │   ├── ensemble.c           # Multi-category ensemble router & sub-forest collection
│   │   ├── reservoir.c          # Streaming reservoir sampling & exponential age decay
│   │   ├── evaluate.c           # Metric depth, single-dimension attribution, & scoring
│   │   ├── json_io.c            # Model serialization & transparent CEIF ingestion
│   │   ├── error.c              # Diagnostic error strings
│   │   └── geometry.h           # Vector operations, dot products, squared distances
│   └── cli/                     # CLI frontend & utilities
│       ├── main.c               # geif CLI entry point & streaming engine
│       ├── columns.c/.h         # Column routing & extraction (-U, -I, -L, -C)
│       ├── template.c/.h        # Output formatting engine (-p, -v, -M, -N, -j)
│       ├── rcfile.c/.h          # Cascading config file parser (~/.geifrc, ~/.ceifrc, -g)
│       ├── test_grid.c/.h       # Evaluation grid generator (-T, -i) & category filtering (-F)
│       ├── ceif2geif.c          # Standalone CEIF to GEIF model migration tool
│       └── xmalloc.c/.h         # Safe memory allocation wrappers
├── docs/                        # Complete technical documentation suite
│   ├── algorithm.md             # Theoretical specifications & algorithmic comparisons
│   ├── heatmaps.md              # Empirical decision heatmaps across 6 benchmark datasets
│   ├── implementation.md        # Codebase implementation architecture & internals
│   ├── manual.md                # Comprehensive CLI reference manual
│   ├── build.md                 # Compiler requirements, flags, & build targets
│   └── README.md                # Documentation index
├── test/
│   ├── unit/                    # C unit test suites
│   │   ├── test_voronoi.c       # Voronoi bisector geometry tests
│   │   ├── test_stadium.c       # Outer space stadium metric & decay tests
│   │   └── test_negative.c      # Negative test cases, garbage inputs, zero-variance
│   ├── test_cli_lifecycle.sh    # CLI lifecycle & streaming pipe tests
│   ├── test_cli_categories.sh   # Multi-category routing & filtering tests
│   ├── test_cli_recalibration.sh# Reservoir pruning (-k) & recalibration tests
│   ├── test_cli_rcfile.sh       # Cascading RC file tests
│   ├── test_cli_grid.sh         # Test grid & population drift tests
│   ├── test_cli_ceif2geif.sh    # Model migration tests
│   ├── test_cli_algorithms.sh   # Multi-algorithm engine verification tests
│   ├── test_ref_bubble.sh       # Hyperspherical Bubble reference benchmark tests
│   └── test_prod_cron_patterns.sh # End-to-end production cron automation tests
├── Makefile                     # Multi-target C17 build system
├── LICENSE                      # GNU General Public License v3 (GPL-3.0)
└── README.md                    # Repository README & overview
```

---

## Building & Testing

### Requirements
- Modern C compiler supporting **ISO C17** (`gcc` $\ge 9$ or `clang` $\ge 10$)
- Standard C runtime and math library (`-lm`)
- `json-c` development libraries (`libjson-c-dev` on Debian/Ubuntu, `json-c-devel` on RHEL/Rocky)

### Build Targets

```bash
# Build static library (lib/libgeif.a), shared library (lib/libgeif.so),
# and CLI binaries (bin/geif, bin/ceif2geif)
make all

# Run unit tests and all 8 integration feature test suites
make test

# Run end-to-end production cron test suite against real-world datasets
make test-prod

# Run complete dual test suite
make test-all

# Clean all build artifacts
make clean
```

The build compiles with `-std=c17 -O3 -march=native -Wall -Wextra -Wpedantic` and produces:
- `lib/libgeif.a` (Static Library)
- `lib/libgeif.so` (Shared Library)
- `bin/geif` (Production CLI Binary)
- `bin/ceif2geif` (Model Migration Utility)

---

## CLI Reference Manual

### Command-Line Options

| Option | Argument | Description |
|---|---|---|
| `-B`, `--algo` | `<name>` | Algorithm engine: `bubble` (default), `voronoi`, `exemplar`, `ceif` / `eif`. |
| `-l` | `<file>` | Train / learn mode: stream data from CSV file or `-` (stdin) into model. |
| `-a` | `<file>` | Analysis / score mode: score data from CSV file or `-` (stdin). |
| `-c` | `<file>` | Categorize mode: classify samples against all categories and assign best match. |
| `-w` | `<file>` | Write / save model to JSON file or `-` (stdout). |
| `-r` | `<file>` | Read / load model from JSON file or `-` (stdin). |
| `-o` | `<file>` | Output file for scoring results (default: `-` for stdout). |
| `-O` | `<spec>` | Outlier threshold: decimal (`0.65`), percentage (`80%`), or `average`. |
| `-k` | *(flag)* | Recalibrate model / prune reservoir outliers (repeatable: `-k -k`). |
| `-D` | `<rate>d` | Exponential age decay rate in days (e.g., `-D30d`, `-D7d`). |
| `-C` | `<spec>` | Category column range/list (e.g., `12`, `"2-4"`, `"1,3,5"`). |
| `-L` | `<spec>` | Label column range/list (e.g., `1`, `"1-2"`). |
| `-U` | `<spec>` | Include column range/list for feature vectors. |
| `-I` | `<spec>` | Ignore column range/list. |
| `-H` | *(flag)* | Skip header row in input CSV. |
| `-f` | `<char>` | Input field delimiter (default: `,`). |
| `-e` | `<char>` | List separator for output / field delimiter fallback (e.g., `-e ';'`). |
| `-d` | `<num>` | Output floating-point decimal precision (default: 6). |
| `-m` | `<fmt>` | Printf numeric format string for dimension output (e.g., `"%'.0f"`). |
| `-p` | `<tmpl>` | Output template for outlier rows (or all rows if `-S` is omitted). |
| `-v` | `<tmpl>` | Output template for inlier rows (dual templating). |
| `-N` | `<tmpl>` | Output template for newly observed, untrained categories. |
| `-M` | `<tmpl>` | Output template for missed categories (trained but absent in stream). |
| `-j` | `<tmpl>` | Sub-template for `%m` per-dimension expansion. |
| `-F` | `<spec>` | Category filter: regex (matching categories are filtered out; prefix `-v ` to invert). |
| `-R` | `<num>` | Minimum training row count required to retain a category sub-forest. |
| `-S` | *(flag)* | Silent outliers: suppress inliers, output only anomalies. |
| `-T` | `[margin]` | Population drift evaluation grid mode with margin (e.g., `-T0.1`). |
| `-i` | `<res>` | Grid resolution per axis (default: 256). |
| `-q` | *(flag)* | Diagnostic query summary: print model metadata and exit. |
| `-g` | `<file>` | Load configuration file (cascades with `~/.geifrc` and `~/.ceifrc`). |

### Format Template Tokens

The following template tokens can be used in `-p`, `-v`, `-N`, `-M`, and `-j`:

| Token | Description | Output Example |
|---|---|---|
| `%s` | Calibrated anomaly score $[0.0, 1.0)$ | `0.6558` |
| `%S` | Anomaly score expressed as percentage | `65.58%` |
| `%l` | Label string | `sample_42` |
| `%c` / `%C` | Category name | `Cuddalore-CO_43-alluvial` |
| `%d` | Feature dimension values (joined by delimiter) | `10.5;20.2;30.1` |
| `%a` | Category dimension averages | `10.1;19.9;29.8` |
| `%e` | Single-dimension attribution / impact scores | `0.45;0.12;0.03` |
| `%m` | Continuous metric depth $H(x)$ or `-j` dimension expansion | `15.3123` |
| `%x` | Hexadecimal RGB color interpolated from score | `FFA500` |
| `%h` | Calibration scale factor $H_{\text{max}}$ | `27.7608` |
| `%o` | Outlier indicator flag (`1` if outlier, `0` if inlier) | `1` |
| `%n` | Total training rows seen by sub-forest | `1599` |
| `%t` | Unix timestamp of evaluation | `1774866874` |
| `%v` | Raw input line tokens joined by delimiter | `token1;token2` |

### RC Configuration File (`~/.geifrc` / `~/.ceifrc`)

GEIF automatically searches for and loads configuration directives from `~/.geifrc` (falling back to legacy `~/.ceifrc`). Options can also be loaded explicitly with `-g <config_file>`.

Supported RC directives:
```ini
TREES 100
MAX_SAMPLES 512
OUTLIER_SCORE 80%
LOW_RGB_COLOR 0x20FF20
HIGH_RGB_COLOR 0xFF0000
DECIMALS 4
PRINT_DIMENSION "%d<dim>%e"
```

### Exit Code Semantics

Following standard POSIX utility conventions (such as `grep` and `diff`), `geif` signals stream classification results via exit status:
- **`0`**: Success, all evaluated rows were nominal inliers (no outliers detected).
- **`2`**: Outliers detected (at least one row scored $\ge$ threshold).
- **`1`**: Fatal error (file not found, syntax error, memory exhaustion).

---

## CEIF to GEIF Migration (`ceif2geif`)

GEIF provides complete backward and forward compatibility with legacy CEIF models:

1. **Transparent Native Loading**: `geif -r` natively detects legacy CEIF JSON files and constructs trees automatically in memory on load.
2. **Standalone Migration Tool (`ceif2geif`)**: Converts CEIF model files into serialized `GEIF-1.0` JSON models:

```bash
# Convert CEIF file to GEIF-1.0 format
bin/ceif2geif legacy_model.json geif_model.json

# Convert with custom tree and sample overrides
bin/ceif2geif -v -t 200 -s 512 legacy_model.json geif_model.json

# Stream through stdin and stdout
cat legacy_model.json | bin/ceif2geif - - > geif_model.json
```

---

## Production Usage Patterns

### 1. Batch & Streaming Training
Train a multi-category model using default hyperspherical Bubble trees from a streaming pipeline:
```bash
grep "2026" raw_stream.csv | geif -l - -L 1 -C "2-4" -m "%'.0f" -e ';' -d 4 -O average -w model.json
```

To explicitly select Voronoi bisectors or Exemplar density:
```bash
geif -l train.csv -w model_voronoi.json -B voronoi -C 1 -U "2-10"
geif -l train.csv -w model_exemplar.json -B exemplar -C 1 -U "2-10"
```

### 2. Stream Scoring with Filtering & Sorting
Analyze incoming events against a model, filter categories via regex, suppress inliers, and rank outliers:
```bash
tail -n 1000 events.csv | geif -r model.json -a - -e ';' -S -F "-v ^5" \
    -M "%t;;%C;%m" -N "NEW;%l;%c;%m" -p "%s;%l;%c;%m" | sort -nr | head -n 20
```

### 3. Multi-Category Classification & Routing (`-c`)
Classify unlabelled rows against all trained sub-forests, routing each observation to the category where it scores lowest:
```bash
cat unlabelled_events.csv | geif -r model.json -c - -O 80% -p "%c;%s;%l"
```

### 4. Rolling Model Updates with Age Decay
Apply exponential decay to stale samples and incorporate fresh daily data:
```bash
geif -r model.json -l daily_update.csv -e ';' -D30d -w model.json
```

### 5. Reservoir Outlier Pruning & Recalibration
Prune contaminants from the model's reservoir sample pool to prevent anomaly masking:
```bash
# Prune single worst outlier
geif -r model.json -k -w model.json

# Prune two worst outliers
geif -r model.json -k -k -w model.json
```

### 6. Visualizing Population Drift with Test Grids
Generate synthetic evaluation grids across feature bounds to track cluster movement:
```bash
geif -r model.json -T0.1 -i 50 -e, -d 4 -p "%d,0x%x" -F "-v ^catA$" | grep -v -- - > drift_plot.dat
```

---

## C17 Library API

```c
#include <geif/geif.h>
#include <stdio.h>

int main(void) {
    geif_ensemble_t *ensemble = NULL;
    geif_config_t config = geif_config_default();
    config.algo = GEIF_ALGO_BUBBLE; // Native hyperspherical Bubble engine

    // 1. Create a 3D ensemble
    geif_ensemble_create(&ensemble, 3, &config);

    // 2. Feed training points into a category
    double p1[3] = {10.0, 20.0, 30.0};
    double p2[3] = {10.5, 20.2, 29.8};
    geif_ensemble_feed(ensemble, "sensors", p1);
    geif_ensemble_feed(ensemble, "sensors", p2);

    // 3. Train all sub-forests
    geif_ensemble_train(ensemble);

    // 4. Score an observation
    double query[3] = {10.2, 20.1, 29.9};
    double score = 0.0;
    geif_ensemble_score(ensemble, "sensors", query, &score);

    printf("Category: sensors, Anomaly score: %.4f\n", score);

    // 5. Cleanup
    geif_ensemble_destroy(ensemble);
    return 0;
}
```

---

## Documentation & Deep Dives

Detailed technical documentation is available in [`docs/`](file:///home/timo_savinen_elisa_fi/git/geif/docs/README.md):
- [**`api_reference.md`**](file:///home/timo_savinen_elisa_fi/git/geif/docs/api_reference.md): Comprehensive C17 API and function reference generated directly from in-code Doxygen comments.
- [**`algorithm.md`**](file:///home/timo_savinen_elisa_fi/git/geif/docs/algorithm.md): Mathematical derivations, Zero Kelvin calibration, and per-algorithm mechanics.
- [**`heatmaps.md`**](file:///home/timo_savinen_elisa_fi/git/geif/docs/heatmaps.md): Empirical 2D decision boundary heatmaps and statistical threshold tables ($0, 0.35, 0.50, 0.60$).
- [**`implementation.md`**](file:///home/timo_savinen_elisa_fi/git/geif/docs/implementation.md): ISO C17 codebase architecture, SIMD vectorization, and data structures.
- [**`manual.md`**](file:///home/timo_savinen_elisa_fi/git/geif/docs/manual.md): Complete CLI reference manual, templates, and Unix piping patterns.
- [**`build.md`**](file:///home/timo_savinen_elisa_fi/git/geif/docs/build.md): Build targets, compiler optimization flags, and testing instructions.

---

## Development & Attribution

**GEIF** was designed and engineered by **Timo Savinen** with AI pair-programming assistance from **Google Antigravity (Gemini)** under human architectural direction. 

Key architectural components, mathematical models (hyperspherical Bubble trees, Voronoi bisectors, continuous stadium metrics, Zero Kelvin scale calibration), C17 implementation, CLI streaming pipeline engine, automated verification test suites, and technical documentation were developed collaboratively.

---

## License & Dataset Attribution

### Source Code License
This project is licensed under the terms of the **GNU General Public License Version 3 (GNU GPLv3)**. See the [LICENSE](file:///home/timo_savinen_elisa_fi/git/geif/LICENSE) file for the full license text.

Copyright (c) 2026 Timo Savinen.

### Test & Benchmark Datasets License
The benchmark datasets located in [`test/data/`](file:///home/timo_savinen_elisa_fi/git/geif/test/data/README.md) (`winequality-red.csv` and `paddydataset.csv`) are licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given. See [`test/data/README.md`](file:///home/timo_savinen_elisa_fi/git/geif/test/data/README.md) for full citations and licensing terms.
