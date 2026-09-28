# GEIF — Geometric Extended Isolation Forest

**Author / Maintainer:** Timo Savinen (AI-assisted)

> [!NOTE]
> **AI-Assisted Development:** GEIF was designed and engineered by Timo Savinen with AI pair-programming assistance (Google Antigravity / Gemini) under human architectural direction. The mathematical specifications, C17 core library, CLI Unix pipelines, test suites, and architectural documentation were developed collaboratively.

**GEIF** (Geometric Extended Isolation Forest) is an industrial-strength, high-performance C17 implementation of a geometric, scale-invariant anomaly detection algorithm. Building upon and refining the foundational concepts of Isolation Forest (iForest) and Extended Isolation Forest (EIF / CEIF), GEIF introduces data-adaptive Voronoi bisector hyperplanes, continuous Euclidean envelope distance ("Stadium" metric), multi-category ensemble routing, single-dimension attribution, streaming age decay, and universal scale calibration.

---

## Table of Contents

- [Key Innovations & Mathematical Foundation](#key-innovations--mathematical-foundation)
  - [1. Data-Adaptive Voronoi Bisectors](#1-data-adaptive-voronoi-bisectors)
  - [2. Smooth Euclidean Stadium Metric ("Outer Space")](#2-smooth-euclidean-stadium-metric-outer-space)
  - [3. "Zero Kelvin" Universal Scale Calibration](#3-zero-kelvin-universal-scale-calibration)
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
- [CEIF to GEIF Migration (`ceif2geif`)](#ceif-to-geif-migration-ceif2geif)
- [Production Usage Patterns](#production-usage-patterns)
  - [1. Batch & Streaming Training](#1-batch--streaming-training)
  - [2. Stream Scoring with Filtering & Sorting](#2-stream-scoring-with-filtering--sorting)
  - [3. Rolling Model Updates with Age Decay](#3-rolling-model-updates-with-age-decay)
  - [4. Reservoir Outlier Pruning & Recalibration](#4-reservoir-outlier-pruning--recalibration)
  - [5. Visualizing Population Drift with Test Grids](#5-visualizing-population-drift-with-test-grids)
- [C17 Library API](#c17-library-api)
- [Development & Attribution](#development--attribution)
- [License](#license)

---

## Key Innovations & Mathematical Foundation

### 1. Data-Adaptive Voronoi Bisectors
Standard Isolation Forests make axis-aligned cuts, creating severe rectangular artifacts. Standard EIF draws arbitrary random normal vectors from uniform spherical distributions, which fail when dimensions have radically different physical scales (e.g. coordinates with a 5000:1 aspect ratio).

GEIF solves this by choosing two distinct points $A$ and $B$ from each node's subsample and placing the splitting hyperplane at their perpendicular bisector:
$$P_{\text{mid}} = \frac{A + B}{2}, \quad \vec{n} = \frac{B - A}{\|B - A\|}$$
- **Scale Invariant**: Adapts intrinsically to the local geometry and aspect ratios of the data without requiring manual normalization or feature scaling.
- **Zero-Variance Feature Masking**: Automatically detects degenerate features with near-zero variance and isolates them safely without numerical instability.

### 2. Smooth Euclidean Stadium Metric ("Outer Space")
When an observation falls outside the bounding box observed during training:
- Conventional tree structures fail to differentiate between a point slightly outside the box and a point millions of units away in outer space.
- GEIF computes the exact Euclidean distance $d_{\text{out}}$ to the training envelope bounding box:
  $$d_{\text{out}}(x) = \sqrt{\sum_{j=1}^D \max(0, \text{min}_j - x_j)^2 + \max(0, x_j - \text{max}_j)^2}$$
- GEIF calculates the effective coordinate span $S_{\text{eff}} = \sqrt{\sum_j (\text{max}_j - \text{min}_j)^2}$.
- An exponential attenuation transforms outer space distance into a continuous metric depth penalty:
  $$\Delta h_{\text{out}} = H_{\text{max}} \cdot \left(1.0 - \exp\left(-\frac{d_{\text{out}}}{S_{\text{eff}}}\right)\right)$$
- As $d_{\text{out}} \to \infty$, the anomaly score smoothly and continuously converges to $1.000000$.

### 3. "Zero Kelvin" Universal Scale Calibration
Anomaly scores in GEIF are normalized in the range $[0.0, 1.0]$ with clear, interpretable semantics:
- **0.000000 ("Zero Kelvin")**: The theoretical absolute inlier — approachable asymptotically as sample density and depth increase, but never exceeded.
- **0.10 – 0.35**: Nominal cluster inliers.
- **0.50**: Default decision boundary threshold.
- **0.80 – 1.00**: Definite anomalies and points far outside the training distribution.

Calibration depth $H_{\text{max}}$ is determined from the deepest observed node during training, scaled by a headroom factor:
$$H_{\text{max}} = 1.25 \times \max_{t \in \text{Trees}} (\text{max-depth}_t)$$

### 4. Streaming Reservoir Sampling & Age Decay
The training pool maintains a fixed maximum capacity:
$$N_{\text{pool}} = N_{\text{trees}} \times N_{\text{samples}} \quad (\text{e.g., } 100 \times 256 = 25{,}600)$$
Any streaming input of arbitrary length is ingested via Algorithm R reservoir sampling, guaranteeing an unbiased uniform sample even if the input stream is sorted or clustered.

When updating existing models over time, GEIF applies exponential age decay (`-D <rate>d`):
$$P_{\text{retain}}(\Delta t) = \exp\left(-\frac{\Delta t}{\tau}\right)$$
Stale reservoir samples are probabilistically replaced by incoming observations, allowing the model to adapt dynamically to evolving production environments.

---

## Architecture & Directory Structure

```
geif/
├── include/
│   └── geif/
│       ├── geif.h               # Public C17 library API
│       ├── types.h              # Core data structures & constants
│       └── error.h              # Status codes and error reporting
├── src/
│   ├── lib/                     # Library implementation
│   │   ├── forest.c             # Forest allocation, lifecycle, & sub-forest ensemble
│   │   ├── reservoir.c          # Streaming reservoir sampling & age decay
│   │   ├── train.c              # Voronoi tree builder & calibration
│   │   ├── evaluate.c           # Continuous metric depth & stadium distance
│   │   ├── score.c              # Single-dimension attribution & ranking
│   │   ├── json_io.c            # Model serialization & transparent CEIF ingestion
│   │   ├── error.c              # Error strings & diagnostics
│   │   └── geometry.h           # Vector operations & bisector math
│   └── cli/                     # CLI binaries & tools
│       ├── main.c               # geif CLI entry point
│       ├── args.c / args.h      # Command-line argument parsing
│       ├── csv.c / csv.h        # High-throughput CSV streaming tokenizer
│       ├── columns.c / columns.h# Column routing (-U, -I, -L, -C)
│       ├── filter.c / filter.h  # Regex category filtering (-F)
│       ├── template.c / template.h # Output formatting engine (-p, -v, -M, -N, -j)
│       ├── rcfile.c / rcfile.h  # Cascading config file parser (~/.geifrc, ~/.ceifrc)
│       ├── test_grid.c / test_grid.h # Evaluation grid generator (-T, -i)
│       └── ceif2geif.c          # Standalone CEIF to GEIF migration tool
├── test/
│   ├── data/                    # Production test datasets
│   │   ├── winequality-red.csv  # 12 columns, semicolon-delimited, single category
│   │   └── paddydataset.csv     # 45 columns, comma-delimited, multi-column categories
│   ├── test_voronoi.c           # Voronoi bisector unit tests
│   ├── test_stadium.c           # Stadium Euclidean distance unit tests
│   ├── test_cli_*.sh            # 10 modular development feature test suites
│   └── test_prod_cron_patterns.sh # End-to-end production cron test suite
├── Makefile                     # Multi-target C17 build system
└── README.md
```

---

## Building & Testing

### Requirements
- Modern C compiler supporting **C17** (`gcc` $\ge 9$ or `clang` $\ge 10$)
- Standard C runtime and math library (`-lm`)
- `json-c` development libraries (`libjson-c-dev` on Debian/Ubuntu, `json-c-devel` on RHEL/Rocky)

### Build Targets

```bash
# Build static library, shared library, and CLI binaries (bin/geif, bin/ceif2geif)
make all

# Run unit tests and all 10 feature test suites
make test

# Run end-to-end production cron suite against real-world datasets
make test-prod

# Run complete dual test suite
make test-all

# Clean build artifacts
make clean
```

The build produces:
- `lib/libgeif.a` (Static Library)
- `lib/libgeif.so` (Shared Library)
- `bin/geif` (Production CLI Binary)
- `bin/ceif2geif` (Model Migration Utility)

---

## CLI Reference Manual

### Command-Line Options

| Option | Argument | Description |
|---|---|---|
| `-l` | `<file>` | Train / learn mode: stream data from CSV file or `-` (stdin) into model. |
| `-a` | `<file>` | Analysis / score mode: score data from CSV file or `-` (stdin). |
| `-w` | `<file>` | Write / save model to JSON file or `-` (stdout). |
| `-r` | `<file>` | Read / load model from JSON file or `-` (stdin). |
| `-o` | `<file>` | Output file for scoring results (default: `-` for stdout). |
| `-g` | `<file>` | Load configuration file (cascades with `~/.geifrc` and `~/.ceifrc`). |
| `-t` | `<num>` | Number of trees per sub-forest (default: 100). |
| `-s` | `<num>` | Samples per tree $\psi$ (default: 256). |
| `-O` | `<spec>` | Outlier threshold: decimal (`0.65`), percentage (`80%`), or `average`. |
| `-k` | *(flag)* | Recalibrate model / prune reservoir outliers (repeatable: `-k -k`). |
| `-D` | `<rate>d` | Exponential age decay rate in days (e.g., `-D30d`, `-D7d`). |
| `-C` | `<spec>` | Category column range/list (e.g., `12`, `"2-4"`, `"1,3,5"`). |
| `-L` | `<spec>` | Label column range/list (e.g., `1`, `"1-2"`). |
| `-U` | `<spec>` | Include column range/list for feature vectors. |
| `-I` | `<spec>` | Ignore column range/list. |
| `-A` | *(flag)* | Automatically infer numeric feature dimensions. |
| `-H` | *(flag)* | Skip header row in input CSV. |
| `-f` / `-e` | `<char>` | Field / delimiter character (e.g., `-e ';'`). |
| `-d` | `<num>` | Output floating-point decimal precision (e.g., `-d 4`, `-d 0`). |
| `-m` | `<fmt>` | Printf numeric format string for dimension output (e.g., `"%'.0f"`). |
| `-p` | `<tmpl>` | Output template for outlier rows (or all rows if `-S` is omitted). |
| `-v` | `<tmpl>` | Output template for inlier rows (dual templating). |
| `-N` | `<tmpl>` | Output template for newly observed, untrained categories. |
| `-M` | `<tmpl>` | Output template for missed categories (trained but absent in stream). |
| `-j` | `<tmpl>` | Sub-template for `%m` per-dimension expansion. |
| `-F` | `<spec>` | Category filter: regex (`"-v ^5"` or `"-v ^Panruti"`). |
| `-R` | `<num>` | Touch timestamp filter: only process categories updated within $N$ days. |
| `-S` | *(flag)* | Silent outliers: suppress inliers, output only anomalies. |
| `-W` | *(flag)* | Suppress warning messages. |
| `-T` | `[margin]` | Population drift evaluation grid mode with margin (e.g., `-T0.1`). |
| `-i` | `<res>` | Grid resolution per axis (default: 50). |
| `-q` | *(flag)* | Diagnostic query summary: print model metadata and exit. |
| `-V` | *(flag)* | Verbose diagnostic logging to stderr. |

### Format Template Tokens

The following template tokens can be used in `-p`, `-v`, `-N`, `-M`, and `-j`:

| Token | Description | Output Example |
|---|---|---|
| `%s` | Anomaly score | `0.6558` |
| `%S` | Anomaly score percentage | `65.58%` |
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
ANALYZE_SAMPLING 1000
LOW_RGB_COLOR 0x20FF20
HIGH_RGB_COLOR 0xFF0000
DECIMALS 4
PRINT_DIMENSION "%d<dim>%e"
```

---

## CEIF to GEIF Migration (`ceif2geif`)

GEIF provides complete backward and forward compatibility with legacy CEIF models:

1. **Transparent Native Loading**: `geif` natively detects legacy CEIF JSON files (which store raw sample pools rather than serialized trees) and trains spherical Voronoi trees automatically in memory on load.
2. **Standalone Migration Tool (`ceif2geif`)**: Converts CEIF model files into serialized `GEIF-1.0` JSON models with full bisector hyperplanes, envelopes, and universal scale calibration:

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
Train a model from a streaming Unix pipeline with label and category columns:
```bash
grep "2026" raw_stream.csv | geif -l - -L 1 -C "2-4" -m "%'.0f" -e ';' -d 0 -O average -w model.json
```

### 2. Stream Scoring with Filtering & Sorting
Analyze incoming events against a model, filter categories via regex, suppress inliers, and rank outliers by score:
```bash
tail -n 1000 events.csv | geif -r model.json -g ~/.ceifrc -a - -e ';' -S -F "-v ^5" \
    -M "%t;;%C;%m" -N "NEW;%l;%c;%m" -p "%s;%l;%c;%m" | sort -nr | head -n 20
```

### 3. Rolling Model Updates with Age Decay
Apply exponential decay to stale samples and incorporate fresh daily data:
```bash
geif -r model.json -g ~/.ceifrc -l daily_update.csv -e ';' -D30d -w model.json
```

### 4. Reservoir Outlier Pruning & Recalibration
Prune the most severe contaminants from the model's reservoir sample pool to prevent anomaly masking:
```bash
# Prune single worst outlier
geif -r model.json -k -w model.json

# Prune two worst outliers
geif -r model.json -k -k -w model.json
```

### 5. Visualizing Population Drift with Test Grids
Generate synthetic 2D evaluation grids across the feature space, color-coded by anomaly score to track cluster drift over time:
```bash
geif -r model.json -g ~/.ceifrc -T0.1 -i 50 -e, -d 4 -p "%d,0x%x" -F "-v ^5$" | grep -v -- - > drift_plot.dat
```

---

## C17 Library API

```c
#include <geif/geif.h>
#include <stdio.h>

int main(void) {
    geif_ensemble_t *ensemble = NULL;
    geif_config_t config = geif_config_default();

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

## Development & Attribution

**GEIF** was designed and engineered by **Timo Savinen** with AI pair-programming assistance from **Google Antigravity (Gemini)** under human architectural direction. 

Key architectural components, mathematical models (Voronoi bisectors, continuous stadium metrics, zero-Kelvin scale calibration), C17 implementation, CLI streaming pipeline engine, automated verification test suites, and technical documentation were developed collaboratively.

---

## License

MIT License. Copyright (c) 2026 Timo Savinen.
