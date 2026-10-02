# GEIF: Implementation Architecture & Modern C17 Engineering Guide

**Author / Maintainer:** Timo Savinen (AI-assisted)

## 1. Engineering Philosophy: Modern C17 Architecture

GEIF is built upon modern ISO C17 standards and UNIX CLI design principles. The codebase cleanly decouples a standalone core library (`libgeif`) from the command-line interface (`bin/geif`), ensuring that the algorithmic core contains zero process exits, relies on explicit type discipline, and executes zero-allocation inference loops.

```
                     +---------------------------------------+
                     |         geif CLI (Frontend)          |
                     |  - POSIX / GNU options (getopt_long)  |
                     |  - Stdin/Stdout stream piping         |
                     |  - Output templating (%s, %m, %d)     |
                     |  - Cascading RC file parser (~/.geifrc)|
                     +---------------------------------------+
                                          │
                              Calls Public C17 API
                                          ▼
                     +---------------------------------------+
                     |          libgeif (Core Engine)        |
                     |  - Zero exit() calls                  |
                     |  - Typed geif_status_t error codes    |
                     |  - Modular algo vtable (algo_ops_t)   |
                     |  - Zero Kelvin calibration engine     |
                     |  - In-place quickselect partitioning  |
                     |  - Sparse JSON serialization          |
                     +---------------------------------------+
```

| Architectural Aspect | Modern C17 Implementation in GEIF |
|---|---|
| **Language Standard** | **Strict ISO C17** (`-std=c17 -Wall -Wextra -Wpedantic -O3 -march=native`) |
| **Type Discipline** | Explicit `<stdint.h>` (`uint32_t`, `size_t`), `<stdbool.h>` (`bool`) |
| **Const Correctness** | Strict `const` qualifiers for immutable buffers (`const double *x`) |
| **Error Handling** | Typed enum return codes (`geif_status_t`); zero `exit()` in core library |
| **Pluggable Algorithms** | Clean virtual function table interface (`geif_algo_ops_t`) |
| **Memory Allocation** | Tree nodes allocated once; in-place array partitioning during training |
| **I/O & Streaming** | UNIX pipeline-first: stdin (`-`), stdout (`-`), and stderr separated |
| **Configuration** | Cascaded RC parser supporting `~/.geifrc`, `~/.ceifrc`, and `-g <file>` |
| **Model Persistence** | Standard sparse JSON via `json-c` (pool-only persistence, fast rebuild) |

---

## 2. Repository & Source Code Structure

```text
git/geif/
├── Makefile                   # C17 build targets (all, lib, bin, test, install)
├── include/
│   └── geif/
│       ├── geif.h             # Public C17 API header
│       ├── types.h            # Forest structs, enum types, error codes
│       └── error.h            # Error code definitions and status macros
├── src/
│   ├── lib/                   # libgeif: Core algorithmic engine
│   │   ├── algo.h             # Pluggable algorithm vtable interface
│   │   ├── algo_registry.c    # Algorithm factory and lookup registry
│   │   ├── algo_bubble.c      # Hyperspherical Bubble tree engine (default)
│   │   ├── algo_voronoi.c     # Voronoi perpendicular bisector tree engine
│   │   ├── algo_exemplar.c    # Exemplar kernel density estimation engine
│   │   ├── algo_ceif.c        # Continuous hyperplane isolation engine
│   │   ├── tree_common.c / .h # Shared tree traversal, Zero Kelvin, nearest nodes
│   │   ├── forest.c           # Forest lifecycle, span calculation, health masking
│   │   ├── ensemble.c         # Multi-category routing, sub-forest ensemble
│   │   ├── evaluate.c         # Score evaluation, thresholding, percentile ranking
│   │   ├── reservoir.c        # Algorithm R streaming reservoir sampling pool
│   │   ├── json_io.c          # Sparse JSON model serializer and parser
│   │   ├── geometry.h         # Vector dot products, Euclidean metrics
│   │   └── error.c            # Error string translation (geif_status_str)
│   ├── cli/                   # bin/geif: Command-line interface
│   │   ├── main.c             # CLI option parsing, stream loops, exit codes
│   │   ├── template.c / .h    # Dynamic format templating (%s, %m, %d, %rgb)
│   │   ├── columns.c / .h     # Column range selector (-I, -U, -L, -C)
│   │   ├── test_grid.c / .h   # Population drift synthetic grid generator (-T)
│   │   └── rcfile.c / .h      # Cascading configuration file parser (-g)
│   └── tools/
│       └── ceif2geif.c        # Model migration utility for legacy formats
├── test/                      # Comprehensive integration test suites
│   ├── test_cli_algorithms.sh # Multi-algorithm engine test suite
│   ├── test_cli_categories.sh # Multi-category routing and filtering test
│   ├── test_cli_grid.sh       # Population drift test grid verification
│   ├── test_cli_rcfile.sh     # RC configuration file parser test
│   └── test_ref_bubble.sh     # Bubble reference benchmark suite
└── docs/                      # Technical documentation
    ├── algorithm.md           # Algorithmic and mathematical specification
    ├── implementation.md      # Implementation architecture (this document)
    ├── heatmaps.md            # Topological heatmaps and empirical scores
    ├── manual.md              # CLI options and usage manual
    └── build.md               # Build requirements and compilation instructions
```

---

## 3. Pluggable Algorithm Interface (`geif_algo_ops_t`)

All spatial isolation algorithms in GEIF implement a common virtual method table defined in `src/lib/algo.h`:

```c
typedef struct geif_algo_ops {
    const char *name;
    const char *description;
    
    // Train an individual tree or populate model exemplars
    geif_status_t (*train_tree)(geif_forest_t *forest, size_t tree_idx,
                                const double *data, size_t n_rows,
                                size_t dim, uint32_t seed);
                                
    // Evaluate continuous metric depth or raw anomaly score for a point
    double (*evaluate_sample)(const geif_forest_t *forest, size_t tree_idx,
                              const double *x, size_t dim);
                              
    // Optional per-tree destructor
    void (*free_tree)(geif_forest_t *forest, size_t tree_idx);
} geif_algo_ops_t;
```

This interface decouples space partitioning geometry from forest management, streaming ingestion, scoring calibration, and JSON I/O.

---

## 4. Short Implementation of Common Foundational Methods

### 4.1 Zero Kelvin Scale Calibration (`tree_common.c`)
During forest initialization, `geif_tree_calc_deepest_path()` traverses each tree to find its theoretical maximum metric depth. The universal maximum depth $H_{\text{max}}$ is then calibrated:

```c
double geif_tree_calc_deepest_path(const geif_node_t *node, double current_depth) {
    if (!node) return current_depth;
    if (node->is_leaf) return current_depth + node->leaf_depth;
    
    double d_left  = geif_tree_calc_deepest_path(node->left,  current_depth + 1.0);
    double d_right = geif_tree_calc_deepest_path(node->right, current_depth + 1.0);
    return (d_left > d_right) ? d_left : d_right;
}

void geif_forest_calibrate_zero_kelvin(geif_forest_t *forest) {
    double sum_deepest = 0.0;
    for (size_t t = 0; t < forest->tree_count; t++) {
        sum_deepest += geif_tree_calc_deepest_path(forest->trees[t].root, 0.0);
    }
    double h_train_max = sum_deepest / (double)forest->tree_count;
    forest->h_max = 1.15 * h_train_max;  // Calibrated scale factor
    forest->s_min = exp(-forest->h_max / forest->c_factor);
    forest->s_max = 1.0;
}
```

### 4.2 Non-Reachable 1.0 & Stadium Metric (`tree_common.c`, `evaluate.c`)
Outer space distance is computed as the normalized Euclidean excursion outside the forest bounding envelope, smoothly attenuating the score asymptotically towards 1.0:

```c
double geif_eval_outer_stadium_decay(double base_score, double d_out) {
    if (d_out <= 0.0) return base_score;
    // Asymptotic exponential convergence towards 1.0
    return 1.0 - (1.0 - base_score) * exp(-0.10 * d_out);
}
```

### 4.3 Nearest Bounding-Box Calculation (`tree_common.c`)
When evaluating hollow cavities or leaf residual density, distance to the bounding box corners of the leaf node is computed in normalized coordinate space:

```c
double geif_calc_leaf_relative_dist(const double *x, const double *bounds_min,
                                    const double *bounds_max, const double *spans,
                                    size_t dim) {
    double sum_sq = 0.0;
    for (size_t j = 0; j < dim; j++) {
        double span = (spans && spans[j] > 1e-9) ? spans[j] : 1.0;
        double diff = 0.0;
        if (x[j] < bounds_min[j]) diff = (bounds_min[j] - x[j]) / span;
        else if (x[j] > bounds_max[j]) diff = (x[j] - bounds_max[j]) / span;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq);
}
```

### 4.4 Dimension Regularization & Health Masking (`forest.c`)
Before training, dimension spans are measured across all input coordinates. If $\text{span}_j \le 10^{-9}$, it is flagged as constant:

```c
void geif_forest_update_bounds(geif_forest_t *forest, const double *data, size_t n_rows) {
    for (size_t j = 0; j < forest->dim; j++) {
        double min_v = data[j], max_v = data[j];
        for (size_t i = 1; i < n_rows; i++) {
            double v = data[i * forest->dim + j];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
        forest->bounds_min[j] = min_v;
        forest->bounds_max[j] = max_v;
        double span = max_v - min_v;
        if (span <= 1e-9) {
            forest->spans[j] = 1.0;
            forest->dim_weights[j] = 0.0; // Health mask: zero weight
        } else {
            forest->spans[j] = span;
            forest->dim_weights[j] = 1.0;
        }
    }
}
```

---

## 5. Implementation of Algorithm Engines

### 5.1 Hyperspherical Bubble Trees (`algo_bubble.c`, Default)
The Bubble engine utilizes:
- **Stack-Allocated Center**: Statically allocated `stack_center[64]` on the call stack for $D \le 64$.
- **$O(N)$ Hoare Quickselect**: Fast median selection on a single reusable scratch buffer `dists_scratch`.
- **In-Place Two-Pointer Partitioning**: Swaps indices in-place without heap allocations.

```c
// In-place partition: left subset dist^2 <= R^2, right subset dist^2 > R^2
size_t l = 0, r = n - 1;
while (l <= r) {
    if (dists[indices[l]] <= r2) {
        l++;
    } else {
        size_t tmp = indices[l];
        indices[l] = indices[r];
        indices[r] = tmp;
        if (r == 0) break;
        r--;
    }
}
size_t left_count = l;
```

### 5.2 Voronoi Bisector Hyperplanes (`algo_voronoi.c`)
Computes the difference vector between two random points and the midpoint intercept, executing a single dot product per node:

```c
for (size_t j = 0; j < dim; j++) {
    normal[j] = sample_b[j] - sample_a[j];
    midpoint[j] = 0.5 * (sample_a[j] + sample_b[j]);
}
double pdotn = geif_dot_product(midpoint, normal, dim);
// Inference branch: geif_dot_product(x, normal, dim) < pdotn ? left : right
```

### 5.3 Exemplar Kernel Density (`algo_exemplar.c`)
Non-tree spatial density kernel evaluating Cauchy distances across the reservoir pool:

```c
double geif_exemplar_evaluate(const geif_forest_t *forest, const double *x) {
    const geif_reservoir_t *res = forest->reservoir;
    if (!res || res->count == 0) return 0.5;
    
    double sum_density = 0.0;
    for (size_t i = 0; i < res->count; i++) {
        const double *p = &res->data[i * forest->dim];
        double dist2 = geif_scaled_euclidean_dist2(x, p, forest->spans, forest->dim);
        sum_density += 1.0 / (1.0 + dist2);
    }
    double avg_density = sum_density / (double)res->count;
    return 1.0 - (1.0 / (1.0 + avg_density));
}
```

### 5.4 Continuous Hyperplane Engine (`algo_ceif.c`)
Generates data-anchored isotropic Gaussian cuts with Marsaglia polar normal vectors, accumulating continuous depth increments $\Delta H = 1/\delta$.

---

## 6. Configuration Management & RC Parser (`rcfile.c`)

GEIF supports cascading configuration discovery:
1. `~/.geifrc` (User default)
2. `~/.ceifrc` (Legacy fallback)
3. Custom file specified via `-g <file>` (can be specified multiple times; later files take precedence)

Supported RC directives:

```ini
# Core hyper-parameters
TREES 100
MAX_SAMPLES 256
OUTLIER_SCORE 0.50
ALGO bubble

# Output and display
DECIMALS 4
LOW_RGB_COLOR 4DF64D
HIGH_RGB_COLOR F25DF2
PRINT_DIMENSION 1
```

The parser uses `strcasecmp` to match keys case-insensitively, trims comments (`#`), and validates numeric bounds safely.

---

## 7. Model Migration Utility (`ceif2geif`)

Located in `src/tools/ceif2geif.c`, this utility migrates legacy CEIF models to modern GEIF-1.0 sparse JSON format:

```bash
# Ingest legacy model and output validated GEIF-1.0 JSON
bin/ceif2geif -v legacy_model.json -o geif_model.json

# Stdin / stdout pipeline migration
cat legacy_model.json | bin/ceif2geif - > migrated_model.json
```

Additionally, `bin/geif -r` automatically detects and transparently loads legacy CEIF JSON models without requiring manual pre-conversion.

---

## 8. CLI Streaming Pipeline & Execution Modes

The `geif` executable (`src/cli/main.c`) is designed for high-throughput streaming in UNIX pipelines:

```bash
# 1. Training mode: Ingest CSV and emit sparse JSON model
cat train.csv | ./bin/geif -l - -w model.json -B bubble -t 100 -s 256

# 2. Scoring pipeline: Read JSON model and score streaming test points
cat test.csv | ./bin/geif -r model.json -a - -o - -O 0.50 -p "%d;score=%s;label=%l"

# 3. Categorization mode: Classify samples against all trained sub-forests
cat stream.csv | ./bin/geif -r model.json -c - -O 0.45 -p "assigned=%c score=%s"

# 4. Population drift grid generation: Synthesize test grid for visualization
./bin/geif -l train.csv -T 0.1 -i 150 -O 0 -p "%d,0x%x" -o grid.csv
```

### Return Code Contract:
- **0**: Clean execution; no outliers detected (all scores $< \text{threshold}$).
- **2**: Outliers detected during scoring (at least one score $\ge \text{threshold}$).
- **1**: Fatal error (invalid CLI arguments, I/O failure, or memory exhaustion).
