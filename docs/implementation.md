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
| **Const Correctness** | Strict `const` qualifiers for immutable buffers (`const double *point`) |
| **Error Handling** | Typed enum return codes (`geif_status_t`); zero `exit()` in core library |
| **Pluggable Algorithms** | Clean virtual function table interface (`geif_algo_ops_t`) at the forest level |
| **Memory Allocation** | Tree nodes allocated in contiguous pools; in-place array partitioning during training |
| **Stack-Buffer Safety** | Scratch buffers bounded by named `#define` macros (`GEIF_STACK_BUFFER_DIMS`) with heap fallback |
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
│       ├── types.h            # Forest structs, enum types, error codes, constants
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
│   └── cli/                   # bin/geif: Command-line interface and tools
│       ├── main.c             # CLI option parsing, stream loops, exit codes
│       ├── template.c / .h    # Dynamic format templating (%s, %m, %d, %rgb)
│       ├── columns.c / .h     # Column range selector (-I, -U, -L, -C)
│       ├── test_grid.c / .h   # Population drift synthetic grid generator (-T)
│       ├── rcfile.c / .h      # Cascading configuration file parser (-g)
│       ├── xmalloc.c / .h     # OOM-checked memory allocation wrappers
│       └── ceif2geif.c        # Model migration utility for legacy formats
├── test/                      # Comprehensive integration & unit test suites
│   ├── test_cli_algorithms.sh # Multi-algorithm engine test suite (-B, --algo)
│   ├── test_cli_attribution.sh# Dimension attribution scoring verification
│   ├── test_cli_categories.sh # Multi-category routing and filtering test (-c, -C, -F)
│   ├── test_cli_ceif2geif.sh  # Legacy model migration verification
│   ├── test_cli_columns.sh    # Column inclusion and ignore verification (-I, -U)
│   ├── test_cli_csv.sh        # CSV dialect, delimiter, and header parser tests
│   ├── test_cli_decay.sh      # Outer space asymptotic decay verification
│   ├── test_cli_grid.sh       # Population drift test grid verification (-T)
│   ├── test_cli_lifecycle.sh  # Streaming reservoir lifecycle and prune tests
│   ├── test_cli_rcfile.sh     # RC configuration file parser test (~/.geifrc)
│   ├── test_cli_recalibration.sh # Percentile thresholds and dynamic scoring (-O)
│   ├── test_prod_cron_patterns.sh # Production periodic cron pattern verification
│   ├── test_ref_bubble.sh     # Bubble reference benchmark suite
│   ├── test_negative.c        # Unit test: deep negative coordinate spaces
│   ├── test_stadium.c         # Unit test: rounded corner stadium distance metrics
│   └── test_voronoi.c         # Unit test: extreme aspect ratio Voronoi bisectors
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
    geif_algo_type_t type;
    const char      *name;
    const char      *description;
    geif_status_t  (*train)(geif_forest_t *f);
    geif_status_t  (*score)(const geif_forest_t *f,
                            const double *point,
                            double *score_out,
                            double *metric_depth_out,
                            double *d_out_out);
    void           (*destroy)(geif_forest_t *f);
} geif_algo_ops_t;
```

### Architectural Rationale: Forest-Level Dispatch

By operating at the forest level rather than the individual tree level:
1. **Unifies Tree and Non-Tree Models:** Non-tree algorithms such as Exemplar kernel density (`algo_exemplar.c`) share the exact same interface as tree ensembles (`algo_bubble.c`, `algo_voronoi.c`, `algo_ceif.c`).
2. **Context-Aware Training:** Algorithms have access to the complete reservoir sample pool (`f->sample_pool`, `f->scaled_pool`), active envelopes (`f->envelope_span`), and precomputed normalization scales.
3. **Optimized Inference:** Each algorithm encapsulates its own coordinate scaling, stack allocation, and metric evaluation without intermediate per-tree dispatch overhead.

---

## 4. Implementation of Core Foundational Methods

### 4.1 Zero Kelvin Scale Calibration (`tree_common.c`)

During forest training and calibration, GEIF finds the theoretical maximum metric depth across all trained trees to establish an absolute baseline for the most isolated possible inlier. Tree nodes are stored in a flat contiguous array `tree->nodes[]`, where leaf nodes are indicated by `left_child == -1 && right_child == -1`:

```c
void geif_tree_find_max_height(const geif_forest_t *f,
                               const geif_tree_t *t,
                               int32_t node_idx,
                               double depth,
                               double *max_h)
{
    if (node_idx < 0 || node_idx >= (int32_t)t->node_count) return;

    const geif_node_t *node = &t->nodes[node_idx];
    if (node->left_child == -1 && node->right_child == -1) {
        double leaf_c = 0.0;
        if (f->avg_sample_dist > 0.0 && node->sample_count > 0 && t->leaf_samples) {
            double rel_dist = MIN_REL_DIST;
            double adjusted_n = (double)node->sample_count / rel_dist;
            leaf_c = geif_c(adjusted_n);
        } else if (node->sample_count > 1) {
            leaf_c = geif_c((double)node->sample_count);
        }
        double h = depth + leaf_c;
        if (h > *max_h) *max_h = h;
        return;
    }

    if (node->left_child != -1) {
        geif_tree_find_max_height(f, t, node->left_child, depth + 1.0, max_h);
    }
    if (node->right_child != -1) {
        geif_tree_find_max_height(f, t, node->right_child, depth + 1.0, max_h);
    }
}
```

In `geif_tree_calibrate()`, the ensemble average maximum depth $H_{\text{train-max}}$ is computed, establishing the scale floor:

$$s_{\min} = 2^{-H_{\text{train-max}} / c(\psi)}$$

When score scaling is active (`f->scale_score = 1`), scores are mapped to $[0, 1)$ via:

$$s_{\text{scaled}} = \frac{s - s_{\min}}{s_{\max} - s_{\min}}$$

Ensuring that core cluster centers anchor at 0.0 ("Zero Kelvin") while extreme outliers approach 1.0 asymptotically without reaching it.

---

### 4.2 Non-Reachable 1.0 & Asymptotic Outer Decay

Points outside the training envelope bounding box do not encounter an artificial boundary wall. Instead, their scaled Euclidean distance $d_{\text{out}}$ beyond the envelope attenuates the score asymptotically towards 1.0:

```c
// Outer space exponential attenuation: asymptotic convergence towards 1.0
if (d_out > 0.0) {
    double target_range = (f->scale_range_idx >= 0 && f->envelope_span) 
                          ? f->envelope_span[f->scale_range_idx] : 1.0;
    double d_norm = (target_range > 1e-12) ? (d_out / target_range) : d_out;
    score = 1.0 - (1.0 - score) * exp(-GEIF_OUTER_DECAY_RATE * d_norm);
}
if (score < 0.0) score = 0.0;
if (score >= 1.0) score = 1.0 - 1e-6; // 1.0 is strictly unreachable
```

Where `GEIF_OUTER_DECAY_RATE` is defined as `0.10` in `include/geif/types.h`.

---

### 4.3 Leaf Relative Distance (`rel_dist`) & Stack Max-Heap

In GEIF, relative leaf distance prevents false positives in high-density regions and detects sparse interior voids or donut cavities. For small sets ($d < 5$), it evaluates up to $2^d$ orthants; for higher dimensions, it caps the nearest neighbor set at `GEIF_MAX_LEAF_NEAREST_SAMPLES` (32) using a stack-allocated binary max-heap:

```c
double geif_calc_leaf_rel_dist(const geif_forest_t *f,
                              const geif_tree_t *tree,
                              const geif_node_t *node,
                              const double *scaled_point)
{
    uint32_t d = f->dimensions;
    uint32_t max_nearest = (d < GEIF_MAX_LEAF_NEAREST_DIM_CAP) 
                           ? (1U << d) : GEIF_MAX_LEAF_NEAREST_SAMPLES;
    const uint32_t *leaf_s = &tree->leaf_samples[node->leaf_sample_offset];
    int32_t n_samples = node->sample_count;

    // Fast path: leaf sample count <= max_nearest
    if ((uint32_t)n_samples <= max_nearest) {
        double sum_d = 0.0;
        uint32_t valid = 0;
        for (int32_t i = 0; i < n_samples; i++) {
            uint32_t s_idx = leaf_s[i];
            if (s_idx < f->pool_count) {
                const double *sample = (f->scaled_pool) ? &f->scaled_pool[s_idx * d]
                                                       : &f->sample_pool[s_idx * d];
                sum_d += geif_dist_sq(scaled_point, sample, d);
                valid++;
            }
        }
        if (valid == 0) return 1.0;
        double mean_d = sum_d / (double)valid;
        return (sqrt(mean_d) / f->avg_sample_dist) + MIN_REL_DIST;
    }

    // Heap path: maintain K smallest squared distances without heap allocations
    double heap[GEIF_MAX_LEAF_NEAREST_SAMPLES];
    uint32_t filled = 0;
    int32_t i = 0;
    while (i < n_samples && filled < max_nearest) {
        uint32_t s_idx = leaf_s[i++];
        if (s_idx < f->pool_count) {
            const double *sample = (f->scaled_pool) ? &f->scaled_pool[s_idx * d]
                                                   : &f->sample_pool[s_idx * d];
            heap[filled++] = geif_dist_sq(scaled_point, sample, d);
        }
    }
    // Sift down to maintain max-heap and track K nearest points...
    // (returns normalized relative distance factor >= MIN_REL_DIST)
}
```

---

### 4.4 Dimension Regularization & Isotropic Normalization

Before tree construction, `init_dimension_scales(geif_forest_t *f)` scans the reservoir pool:
1. Calculates coordinate extrema (`envelope_min`, `envelope_max`) and spans (`envelope_span`).
2. Dimensions with $\text{span}_j < 10^{-9}$ are marked inactive (`dim_active[j] = 0`), preventing division by zero.
3. Identifies the dimension with the largest span (`scale_range_idx`) to normalize all coordinates into an isotropic space (`f->scaled_pool`).
4. Precomputes the nominal cluster distance $\delta_{\text{nominal}}$ and harmonic correction factor $c(\psi)$.

---

## 5. Implementation of Algorithm Engines

### 5.1 Hyperspherical Bubble Trees (`algo_bubble.c`, Default)

The Bubble tree algorithm isolates points using hyperspherical envelopes rather than planar cuts:
- **Stack-Allocated Center**: Statically allocated `stack_center[GEIF_STACK_BUFFER_DIMS]` on the call stack for $D \le 64$.
- **$O(N)$ Hoare Quickselect**: Selects the median squared radius $R^2$ on a reusable scratch buffer.
- **In-Place Two-Pointer Partitioning**: Swaps indices in-place without heap allocations:

```c
// In-place partition: left subset dist^2 <= R^2, right subset dist^2 > R^2
size_t l = 0;
size_t r = count;
while (l < r) {
    const double *pt = &f->scaled_pool[indices[l] * d];
    if (geif_dist_sq(center, pt, d) <= radius_sq) {
        l++;
    } else {
        r--;
        uint32_t tmp = indices[l];
        indices[l] = indices[r];
        indices[r] = tmp;
    }
}
size_t left_count = l;
size_t right_count = count - l;
```

---

### 5.2 Voronoi Bisector Hyperplanes (`algo_voronoi.c`)

Partitions space using perpendicular bisector hyperplanes between two randomly selected exemplars:
1. Selects two distinct points $a$ and $b$ from the current subset.
2. Bisector normal vector: $n = b - a$.
3. Midpoint intercept: $m = 0.5(a + b)$.
4. Threshold: $p_{\text{dot}} = m \cdot n$.
5. Fast branch test during traversal: $\text{scaled\_point} \cdot n < p_{\text{dot}} \implies \text{left} : \text{right}$.

---

### 5.3 Exemplar Kernel Density (`algo_exemplar.c`)

Non-tree spatial density kernel evaluating adaptive Cauchy kernel distances across the reservoir pool:
1. **Training Phase (`geif_exemplar_train`)**: For each sample $i$ in the pool, finds its $K$-nearest neighbors (`EXEMPLAR_K = 5`) and calculates local adaptive bandwidth $\sigma_i$.
2. **Inference Phase (`geif_exemplar_score`)**: For a query point, aggregates Cauchy density:

$$D = \frac{1}{K} \sum_{i \in \text{NN}_K} \frac{1}{1 + (d_i / \sigma_i)^2}$$

Score is evaluated as $s = 1.0 - D$, attenuated by outer space decay when beyond the training envelope.

---

### 5.4 Continuous Hyperplane Engine (`algo_ceif.c`)

The classic Extended Isolation Forest engine:
1. Generates data-anchored isotropic Gaussian cuts using Box-Muller normal vectors (`geif_gaussrand`).
2. Normalizes vector $n$ to unit length: $n \leftarrow n / \|n\|$.
3. Chooses intercept $p_{\text{dot}} \sim \text{Uniform}(\min_i x_i \cdot n, \max_i x_i \cdot n)$.
4. Traversal evaluates dot product test vs $p_{\text{dot}}$ and accumulates metric depth.

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

Located in [`src/cli/ceif2geif.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/cli/ceif2geif.c) (compiled as `bin/ceif2geif`), this utility migrates legacy CEIF models to the modern GEIF-1.0 sparse JSON format:

```bash
# Ingest legacy model and output validated GEIF-1.0 JSON
bin/ceif2geif -v legacy_model.json -o geif_model.json

# Stdin / stdout pipeline migration
cat legacy_model.json | bin/ceif2geif - > migrated_model.json
```

Additionally, `bin/geif -r` automatically detects and transparently loads legacy CEIF JSON models without requiring manual pre-conversion.

---

## 8. CLI Streaming Pipeline & Execution Modes

The `geif` executable ([`src/cli/main.c`](file:///home/timo_savinen_elisa_fi/git/geif/src/cli/main.c)) is designed for high-throughput streaming in UNIX pipelines:

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
