# GEIF: Implementation Architecture & Modern C CLI Engineering Guide

**Author / Maintainer:** Timo Savinen (AI-assisted)

## 1. Engineering Philosophy: Modern C CLI Best Practices

While GEIF builds upon the proven domain concepts of `ceif` (streaming CSV, online reservoir updates, vector anomaly scoring), its codebase is designed from the ground up following **modern C standards and CLI best practices**, rather than inheriting legacy C89/C99 idioms or monolithic tool structures.

### 1.1 Core Principles of Modern C Architecture in GEIF

```
                     +---------------------------------------+
                     |         geif CLI (Frontend)          |
                     |  - POSIX / GNU options (getopt_long)  |
                     |  - Stdin/Stdout stream piping         |
                     |  - NO_COLOR / isatty() diagnostics    |
                     |  - sysexits exit codes                |
                     +---------------------------------------+
                                         │
                             Calls Public C17 API
                                         ▼
                     +---------------------------------------+
                     |          libgeif (Core Engine)        |
                     |  - Zero exit() calls                  |
                     |  - Typed geif_status_t error codes    |
                     |  - Pure Data-Oriented Memory layout   |
                     |  - SIMD / AVX2 vector primitives      |
                     |  - Zero-allocation inference loop     |
                     +---------------------------------------+
```

| Dimension | Legacy C Style (e.g. classic `ceif`) | Modern C Best Practice (GEIF) |
|---|---|---|
| **Language Standard** | Implicit C89/C99, platform-dependent | **Strict ISO C17** (`-std=c17 -Wall -Wextra -Wpedantic`) |
| **Type Discipline** | Naked `int`, `long`, `char *` | Explicit `<stdint.h>` (`int32_t`, `size_t`), `<stdbool.h>` (`bool`) |
| **Const Correctness** | Mutable pointers everywhere | Strict `const` everywhere (`const double * restrict`) |
| **Error Handling** | Calls `exit(1)` or prints inside library | **Typed status enums** (`geif_status_t`); zero `exit()` in core |
| **Separation of Concerns** | Monolithic CLI + library mingled | **Clean library (`libgeif`)** decoupled from **CLI (`geif`)** |
| **Memory Allocation** | Fragmented `malloc` per node/vector | **Data-Oriented Contiguous Buffers (Arena / SoA)** |
| **I/O & Streams** | Direct file descriptor juggling | UNIX pipeline-first: stdin (`-`), stdout, stderr separated |
| **Diagnostics & UX** | Hardcoded terminal colors | `isatty()` detection, strict `NO_COLOR` compliance |
| **Build System** | Autotools (`configure.ac`, `Makefile.am`) | **Modern CMake (>= 3.20)** + `compile_commands.json` |

---

## 2. Modern Repository & Directory Layout

```text
geif/
├── CMakeLists.txt             # Modern CMake build definition (targets, sanitizers)
├── include/
│   └── geif/
│       ├── geif.h             # Public C17 API for libgeif (clean, stable)
│       ├── types.h            # Fixed-width types, error enums, and structures
│       └── version.h          # Semantic versioning macros
├── src/
│   ├── lib/                   # libgeif: Pure, standalone algorithm library
│   │   ├── forest.c           # Forest lifecycle, memory allocation, and bounds
│   │   ├── learn.c            # Voronoi bisector generator and recursive partitioning
│   │   ├── analyze.c          # Metric continuous depth walk and score evaluation
│   │   ├── serialize.c        # Versioned binary persistence and validation
│   │   ├── simd_math.h        # Vector dot products (AVX2/NEON/compiler autovec)
│   │   └── arena.c / .h       # Contiguous node/vector arena allocator
│   └── cli/                   # geif CLI: Command-line frontend
│       ├── main.c             # Argument parsing, mode dispatch, and exit codes
│       ├── csv_stream.c / .h  # High-throughput streaming CSV parser
│       ├── term_ui.c / .h     # Terminal progress, colors, and NO_COLOR detection
│       └── json_emitter.c     # Formatted JSON output stream
├── test/
│   ├── unit/                  # Fast ctest unit tests (math, splits, memory)
│   ├── integration/           # CLI end-to-end piping tests
│   └── data/
│       ├── complex2d.csv      # Topological cavity benchmark
│       └── 2blob.csv          # Standard Gaussian benchmark
└── docs/                      # Technical documentation
    ├── algorithm.md           # Formal mathematical specification
    └── implementation.md      # Engineering architecture (this document)
```

---

## 3. Modern Type Definitions & Data Structures (`include/geif/types.h`)

### 3.1 Strict Typed Error Statuses

A core modern best practice is that **libraries must never abort or exit the calling process**:

```c
typedef enum geif_status {
    GEIF_OK                  =  0,
    GEIF_ERR_INVALID_PARAM   = -1,
    GEIF_ERR_OUT_OF_MEMORY   = -2,
    GEIF_ERR_IO              = -3,
    GEIF_ERR_FORMAT_CORRUPT  = -4,
    GEIF_ERR_EMPTY_DATASET   = -5,
    GEIF_ERR_DIM_MISMATCH    = -6
} geif_status_t;

// Utility for human-readable error diagnostics
const char *geif_status_str(geif_status_t status);
```

---

### 3.2 Data-Oriented Contiguous Memory Layout

In classic implementations, allocating `double *n` separately for every node causes thousands of tiny heap allocations, cache line misses, and pointer chasing during tree traversal.

**GEIF Modern Design**: Flatten node memory into contiguous arrays. Normal vectors are stored in a single contiguous pool per tree:

```c
// Compact 64-byte cache-aligned node structure
typedef struct geif_node {
    int32_t  left_child;        // Index in tree's node array (-1 if leaf)
    int32_t  right_child;       // Index in tree's node array (-1 if leaf)
    uint32_t normal_offset;     // Offset into tree's contiguous normals pool: &normals[offset]
    double   pdotn;             // Precomputed scalar threshold: dot((A+B)/2, n)
    double   step_weight;       // Continuous metric increment: delta_H
    double   d_AB;              // Generator separation distance: ||B - A||
    int32_t  sample_count;      // Samples in leaf (leaf only)
    double   leaf_residual;     // Residual leaf weight (leaf only)
} geif_node_t;

typedef struct geif_tree {
    geif_node_t *nodes;         // Contiguous array of nodes
    size_t       node_count;    // Total nodes in this tree
    double      *normals_pool;  // Contiguous buffer: [node_count * dimensions]
    double       max_path_depth;// Longest path in this tree
} geif_tree_t;

typedef struct geif_forest {
    uint32_t     dimensions;    // Feature count
    uint32_t     tree_count;    // Number of trees (e.g. 200)
    geif_tree_t *trees;         // Array of trees
    
    // Initial surrounding outer envelope
    double      *envelope_min;  // Array of size [dimensions]
    double      *envelope_max;  // Array of size [dimensions]
    
    // Universal geometric scale
    double       H_max;         // Forest-wide maximum depth (score = 1.0 - H / H_max)
    
    // Category string for multi-category classification (-c)
    char         category[64];
} geif_forest_t;
```

#### Cache Benefits:
- Traversal walks through `nodes[node_idx]` with maximum spatial locality.
- Normal vectors reside in a linear array `normals_pool`, streaming directly into CPU L1/L2 data cache.

---

## 4. Modern Library Core Implementation (`src/lib/`)

### 4.1 SIMD-Vectorized Bisector Decision

Modern C compilers (GCC and Clang) vectorize contiguous loops automatically when supplied with `restrict` pointers and `#pragma GCC ivdep`:

```c
static inline double geif_dot(const double * restrict a,
                              const double * restrict b,
                              size_t dim)
{
    double sum = 0.0;
    #pragma GCC ivdep
    for (size_t j = 0; j < dim; ++j) {
        sum += a[j] * b[j];
    }
    return sum;
}
```

### 4.2 Zero-Allocation Inference (`analyze.c`)

The evaluation function guarantees **zero heap allocations**:

```c
geif_status_t geif_evaluate(const geif_forest_t * restrict forest,
                            const double * restrict x,
                            double * restrict out_score)
{
    if (!forest || !x || !out_score) {
        return GEIF_ERR_INVALID_PARAM;
    }

    const size_t dim = forest->dimensions;

    // 1. Initial surrounding outer envelope check: O(D)
    for (size_t j = 0; j < dim; ++j) {
        if (x[j] < forest->envelope_min[j] || x[j] > forest->envelope_max[j]) {
            *out_score = 1.000000;
            return GEIF_OK;
        }
    }

    // 2. Ensemble traversal
    double total_metric_depth = 0.0;
    const uint32_t t_count = forest->tree_count;

    for (uint32_t t = 0; t < t_count; ++t) {
        const geif_tree_t *tree = &forest->trees[t];
        int32_t curr = 0; // Root is always at index 0
        double tree_depth = 0.0;

        while (true) {
            const geif_node_t *node = &tree->nodes[curr];
            if (node->left_child == -1 && node->right_child == -1) {
                tree_depth += node->leaf_residual;
                break;
            }

            tree_depth += node->step_weight;
            const double *n = &tree->normals_pool[node->normal_offset];

            if (geif_dot(x, n, dim) < node->pdotn) {
                curr = node->left_child;
            } else {
                curr = node->right_child;
            }
        }
        total_metric_depth += tree_depth;
    }

    // 3. Normalized score computation (Zero Kelvin Principle)
    const double H_avg = total_metric_depth / (double)t_count;
    double score = 1.0 - (H_avg / forest->H_max);

    // Natural structural invariant guarantees score > 0.0 and score <= 1.0
    if (score < 0.0) score = 0.0;
    else if (score > 1.0) score = 1.0;

    *out_score = score;
    return GEIF_OK;
}

### 4.3 Structural H_max Computation (The Zero Kelvin Principle)

In accordance with the Zero Kelvin Principle, $H_{\max}$ is derived purely and deterministically from tree topology at the end of training. No sample sweeps or test queries are needed:

```c
static double compute_tree_max_depth(const geif_tree_t *tree, int32_t node_idx, double current_depth)
{
    const geif_node_t *node = &tree->nodes[node_idx];
    if (node->left_child == -1 && node->right_child == -1) {
        return current_depth + node->leaf_residual;
    }
    double d_step = current_depth + node->step_weight;
    double left_max  = compute_tree_max_depth(tree, node->left_child, d_step);
    double right_max = compute_tree_max_depth(tree, node->right_child, d_step);
    return (left_max > right_max) ? left_max : right_max;
}

void geif_forest_finalize_hmax(geif_forest_t *forest)
{
    double total_leaf_max = 0.0;
    for (uint32_t t = 0; t < forest->tree_count; ++t) {
        total_leaf_max += compute_tree_max_depth(&forest->trees[t], 0, 0.0);
    }
    forest->H_max = total_leaf_max / (double)forest->tree_count;
}
```

---

## 5. Modern CLI Frontend Best Practices (`src/cli/`)

### 5.1 Clean Separation of Data and Diagnostics
Following the UNIX philosophy:
- **`stdout`**: Strictly reserved for processed data (CSV, JSON, scores). Can be cleanly piped to `awk`, `cut`, or downstream services.
- **`stderr`**: Progress bars, summary statistics, warnings, and error messages.

### 5.2 Terminal Color Discipline (NO_COLOR Support)
Modern CLI tools must respect the environment to prevent breaking automated log scrapers:

```c
bool cli_should_use_color(FILE *stream)
{
    // 1. If output is redirected (pipe/file), disable color
    if (!isatty(fileno(stream))) {
        return false;
    }
    // 2. Comply with https://no-color.org standard
    if (getenv("NO_COLOR") != NULL) {
        return false;
    }
    // 3. Check explicit terminal capabilities
    const char *term = getenv("TERM");
    if (!term || strcmp(term, "dumb") == 0) {
        return false;
    }
    return true;
}
```

### 5.3 First-Class Stdin / Stdout Piping (`-`)
Support standard stream piping for cloud and containerized workflows:
```bash
# Streaming inference in a pipeline:
cat telemetry_stream.csv | geif -r production.geif -a - | grep -v ',0\.0'
```

### 5.4 Standardized Exit Codes (`<sysexits.h>`)
Instead of arbitrary `exit(1)`:
- `0`: Success (`EXIT_SUCCESS`).
- `64`: Command-line usage error (`EX_USAGE`).
- `65`: Data format error (`EX_DATAERR`).
- `66`: Cannot open input file (`EX_NOINPUT`).
- `70`: Internal software error (`EX_SOFTWARE`).
- `71`: Operating system error (`EX_OSERR`).

---

## 6. Modern Build System: CMake 3.20+ with Presets

Replace legacy Autotools with clean, modern CMake:

```cmake
cmake_minimum_required(VERSION 3.20)
project(geif VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Modern compiler hardening flags
add_compile_options(
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    -Wformat=2 -Wundef -fstack-protector-strong
)

# Core library (libgeif)
add_library(geif_core STATIC
    src/lib/forest.c
    src/lib/learn.c
    src/lib/analyze.c
    src/lib/serialize.c
)
target_include_directories(geif_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# CLI executable
add_executable(geif src/cli/main.c src/cli/csv_stream.c src/cli/term_ui.c)
target_link_libraries(geif PRIVATE geif_core m)

# Sanitizer target for development & CI
option(ENABLE_SANITIZERS "Enable Address and Undefined sanitizers" OFF)
if(ENABLE_SANITIZERS)
    target_compile_options(geif_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(geif_core PUBLIC -fsanitize=address,undefined)
endif()
```

### Developer Experience:
- Run `cmake -B build -DENABLE_SANITIZERS=ON` for instant memory-error detection during tests.
- Generates `build/compile_commands.json` automatically for Neovim/VSCode `clangd` completion.

---

## 7. Migration Roadmap from `ceif` to Modern `geif`

1. **Step 1: Setup Modern CMake Skeleton & `include/geif/types.h`**
   - Create repo with strict C17 compiler flags and error enum definitions.
2. **Step 2: Implement Contiguous Memory Layout (`arena.c` / `forest.c`)**
   - Build flat node and normal buffer allocators.
3. **Step 3: Implement Pure Algorithmic Core (`learn.c` & `analyze.c`)**
   - Voronoi bisector splits, outer envelope clamping, metric depth accumulation, and $H_{\max}$ calculation.
4. **Step 4: Port High-Speed Streaming I/O (`csv_stream.c`)**
   - Adapt `ceif`'s fast CSV parser into a clean modular streaming reader.
5. **Step 5: Assemble CLI Frontend (`main.c`)**
   - GNU long options, UNIX stream piping, `NO_COLOR` terminal handling, and standard sysexits return codes.
6. **Step 6: Automated Test Suite & Benchmarking**
   - Unit tests under `ctest`, leak tests with Valgrind and AddressSanitizer, and visual heatmap comparisons against `complex2d.csv`.
