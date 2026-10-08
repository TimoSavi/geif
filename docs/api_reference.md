# GEIF: C17 API & Architecture Reference Manual

**Author / Maintainer:** Timo Savinen (AI-assisted)  
**Version:** 1.1.0  
**Repository:** [github.com/TimoSavi/geif](https://github.com/TimoSavi/geif)  

> This manual is automatically generated from the in-code Doxygen documentation across all C17 headers and implementation files. It documents all public functions, algorithm implementations, mathematical helpers, data structures, and CLI utilities.

---

## Table of Contents

- [**Data Types, Constants & Return Codes**](#data-types-constants--return-codes)
- [**Public Library C17 API**](#public-library-c17-api)
  - [`include/geif/geif.h`](#includegeifgeifh)
  - [`include/geif/types.h`](#includegeiftypesh)
  - [`include/geif/error.h`](#includegeiferrorh)
- [**Algorithm Engines & Mathematical Core**](#algorithm-engines--mathematical-core)
  - [`src/lib/algo.h`](#srclibalgoh)
  - [`src/lib/algo_registry.c`](#srclibalgo_registryc)
  - [`src/lib/algo_bubble.c`](#srclibalgo_bubblec)
  - [`src/lib/algo_voronoi.c`](#srclibalgo_voronoic)
  - [`src/lib/algo_exemplar.c`](#srclibalgo_exemplarc)
  - [`src/lib/algo_ceif.c`](#srclibalgo_ceifc)
  - [`src/lib/tree_common.c`](#srclibtree_commonc)
  - [`src/lib/tree_common.h`](#srclibtree_commonh)
  - [`src/lib/geometry.h`](#srclibgeometryh)
- [**Forest, Ensembles & Evaluation Engine**](#forest-ensembles--evaluation-engine)
  - [`src/lib/forest.c`](#srclibforestc)
  - [`src/lib/train.c`](#srclibtrainc)
  - [`src/lib/evaluate.c`](#srclibevaluatec)
  - [`src/lib/ensemble.c`](#srclibensemblec)
  - [`src/lib/reservoir.c`](#srclibreservoirc)
  - [`src/lib/json_io.c`](#srclibjson_ioc)
  - [`src/lib/error.c`](#srcliberrorc)
- [**CLI Frontend, Tools & Infrastructure**](#cli-frontend-tools--infrastructure)
  - [`src/cli/main.c`](#srcclimainc)
  - [`src/cli/columns.c`](#srcclicolumnsc)
  - [`src/cli/columns.h`](#srcclicolumnsh)
  - [`src/cli/template.c`](#srcclitemplatec)
  - [`src/cli/template.h`](#srcclitemplateh)
  - [`src/cli/rcfile.c`](#srcclircfilec)
  - [`src/cli/rcfile.h`](#srcclircfileh)
  - [`src/cli/test_grid.c`](#srcclitest_gridc)
  - [`src/cli/test_grid.h`](#srcclitest_gridh)
  - [`src/cli/xmalloc.c`](#srcclixmallocc)
  - [`src/cli/xmalloc.h`](#srcclixmalloch)
  - [`src/cli/ceif2geif.c`](#srccliceif2geifc)

---

## Data Types, Constants & Return Codes

### Tuning Constants & Safety Thresholds (`include/geif/types.h`, `src/lib/tree_common.h`)

| Macro Constant | Value | Description & Safety Semantics |
| :--- | :--- | :--- |
| `GEIF_DEFAULT_TREE_COUNT` | `100` | Default number of isolation trees per ensemble. |
| `GEIF_DEFAULT_SAMPLES_PER_TREE` | `256` | Sub-sample size $\psi$ drawn without replacement per tree. |
| `GEIF_DEFAULT_KAPPA` | `1.25` | Headroom factor for Zero Kelvin baseline depth ($H_{\max}$ anchored to $1.25 \times H_{\text{train-max}}$). |
| `GEIF_MIN_REL_DIST` | `0.033333` | Minimum relative Euclidean distance floor for leaf neighbor adjustment. |
| `GEIF_REL_DIST_MULTIPLIER` | `2.0` | Boundary half-space geometric multiplier for relative distance calibration. |
| `GEIF_MIN_LEAF_SAMPLE_DIM_CAP` | `4U` | Dimensionality threshold ($D < 4$) for $2^D$ theoretical surrounding leaf samples. |
| `GEIF_MIN_LEAF_SAMPLE_HIGH_DIM` | `8U` | Capped minimum leaf sample count for higher dimensions ($D \ge 4$). |
| `GEIF_NODE_MIN_SAMPLE(d)` | `$2^d$ or $8$` | Dynamic minimum samples on leaves before terminating splits: frames queries by surrounding orthants. |
| `GEIF_OUTER_DECAY_RATE` | `0.10` | Exponential approach rate towards asymptotic 1.0 ceiling in outer space. |
| `GEIF_STACK_BUFFER_DIMS` | `64U` | Maximum dimension count for zero-allocation stack scratch buffers. |
| `GEIF_MAX_LEAF_NEAREST_SAMPLES` | `32U` | Maximum nearest leaf neighbors tracked via stack max-heap for relative distance. |
| `GEIF_MAX_LEAF_NEAREST_DIM_CAP` | `5U` | Dimension threshold ($2^5 = 32$) switching from full orthant scan to binary max-heap. |
| `GEIF_INITIAL_TREE_NODES` | `64U` | Initial capacity for flat dynamic tree node allocations. |
| `GEIF_INITIAL_NORMALS_COUNT` | `64U` | Initial capacity multiplier for contiguous normal vector buffer. |
| `GEIF_INITIAL_LEAF_SAMPLES` | `128U` | Initial capacity for contiguous leaf sample index storage. |

### Typed Status & Return Codes (`geif_status_t`)

| Status Code | Value | Meaning / Condition |
| :--- | :--- | :--- |
| `GEIF_OK` | `0` | Success / normal execution. |
| `GEIF_ERR_INVALID_ARG` | `-1` | Null pointer or out-of-range argument provided. |
| `GEIF_ERR_OUT_OF_MEMORY` | `-2` | Heap memory allocation failed. |
| `GEIF_ERR_EMPTY_DATASET` | `-3` | Forest fed with zero samples or empty CSV. |
| `GEIF_ERR_FILE_IO` | `-4` | Unable to open, read, or write file descriptor. |
| `GEIF_ERR_JSON_PARSE` | `-5` | Malformed JSON format encountered during model loading. |
| `GEIF_ERR_NOT_TRAINED` | `-6` | Scoring attempted on un-trained forest. |
| `GEIF_ERR_DIMENSION_MISMATCH` | `-7` | Query coordinate vector length does not match model dimensionality. |

### Algorithm Engine Selector (`geif_algo_type_t`)

| Enum Constant | Value | CLI Name | Description |
| :--- | :--- | :--- | :--- |
| `GEIF_ALGO_CEIF` | `0` | `ceif` / `eif` | Data-anchored isotropic Gaussian cuts with Zero Kelvin scale floor and outer decay. |
| `GEIF_ALGO_BUBBLE` | `1` | `bubble` | Hyperspherical Bubble tree cavity carving with median quickselect (default). |
| `GEIF_ALGO_EXEMPLAR` | `2` | `exemplar` | Direct SIMD Cauchy kernel density evaluation on reservoir pool samples. |
| `GEIF_ALGO_VORONOI` | `3` | `voronoi` | Scale-invariant Voronoi perpendicular bisector hyperplane cuts. |

---

## Public Library C17 API

Public interfaces, forest/ensemble lifecycle, training dispatch, scoring, JSON persistence, and diagnostics.

### [`include/geif/geif.h`](../include/geif/geif.h)

**Module Purpose:** Public C17 API for Geometric Extended Isolation Forest (GEIF).

#### [`geif_config_default`](../include/geif/geif.h#L24)

```c
geif_config_t geif_config_default(void);
```

**Description:** Returns the default configuration for GEIF.

---

#### [`geif_algo_name`](../include/geif/geif.h#L29)

```c
const char *geif_algo_name(geif_algo_type_t algo);
```

**Description:** Converts an algorithm type enum to string identifier.

---

#### [`geif_algo_from_name`](../include/geif/geif.h#L35)

```c
geif_algo_type_t geif_algo_from_name(const char *name);
```

**Description:** Parses an algorithm name string to its enum type. Supports: "ceif", "eif", "gaussian", "bubble", "exemplar", "voronoi".

---

#### [`geif_forest_create`](../include/geif/geif.h#L45)

```c
geif_status_t geif_forest_create(geif_forest_t **forest_out, uint32_t dimensions, const geif_config_t *config);
```

**Description:** Allocates and initializes a new GEIF forest.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest_out` | `[out]` | Pointer to receive the allocated forest. |
| `dimensions` | `[in]` | Dimensionality of the input features (D > 0). |
| `config` | `[in]` | Optional configuration (NULL for defaults). |

**Returns:** GEIF_OK on success, or an error code.

---

#### [`geif_forest_destroy`](../include/geif/geif.h#L54)

```c
void geif_forest_destroy(geif_forest_t *forest);
```

**Description:** Frees all resources associated with a forest.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest` | `[in]` | Forest to free (safe to call with NULL). |

---

#### [`geif_forest_feed`](../include/geif/geif.h#L63)

```c
geif_status_t geif_forest_feed(geif_forest_t *forest, const double *point);
```

**Description:** Feeds a single observation vector into the forest's reservoir sample pool.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest` | `[in]` | Forest instance. |
| `point` | `[in]` | Feature vector of length dimensions. |

**Returns:** GEIF_OK on success, or an error code.

---

#### [`geif_forest_train`](../include/geif/geif.h#L71)

```c
geif_status_t geif_forest_train(geif_forest_t *forest);
```

**Description:** Trains all trees in the forest using the collected sample pool.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest` | `[in]` | Forest instance with collected samples. |

**Returns:** GEIF_OK on success, or an error code.

---

#### [`geif_forest_evaluate_metric_depth`](../include/geif/geif.h#L81)

```c
double geif_forest_evaluate_metric_depth(const geif_forest_t *f, const double *point, double *d_out_out);
```

**Description:** Evaluates raw unnormalized continuous metric depth H(x) for a point.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Forest instance. |
| `point` | `[in]` | Observation vector of length dimensions. |
| `d_out_out` | `[in]` | Optional pointer to receive outer stadium distance. |

**Returns:** Accumulated continuous metric depth.

---

#### [`geif_forest_score`](../include/geif/geif.h#L99)

```c
geif_status_t geif_forest_score(const geif_forest_t *forest, const double *point, double *score_out);
```

**Description:** Evaluates an observation and computes its normalized anomaly score in [0.0, 1.0].

Score semantics:
- 0.000000: Asymptotic absolute inlier ("Zero Kelvin")
- 0.10 - 0.35: Strong nominal cluster inlier
- 0.50: Boundary threshold
- 0.85 - 1.000000: Outlier / Anomaly / Outer Space

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest` | `[in]` | Trained forest instance. |
| `point` | `[in]` | Query vector of length dimensions. |
| `score_out` | `[out]` | Pointer to receive the computed anomaly score. |

**Returns:** GEIF_OK on success, or an error code.

---

#### [`geif_forest_score_detailed`](../include/geif/geif.h#L113)

```c
geif_status_t geif_forest_score_detailed(const geif_forest_t *forest, const double *point, double *score_out, double *metric_depth_out, double *d_out_out);
```

**Description:** Evaluates an observation with detailed diagnostic metric outputs.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest` | `[in]` | Trained forest instance. |
| `point` | `[in]` | Query vector of length dimensions. |
| `score_out` | `[out]` | Final anomaly score in [0.0, 1.0]. |
| `metric_depth_out` | `[out]` | Accumulated continuous metric depth H. |
| `d_out_out` | `[out]` | Relative distance outside the envelope stadium. |

**Returns:** GEIF_OK on success, or an error code.

---

#### [`geif_forest_get_averages`](../include/geif/geif.h#L126)

```c
geif_status_t geif_forest_get_averages(const geif_forest_t *forest, double *averages_out);
```

**Description:** Computes or retrieves category dimension averages.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest` | `[in]` | Trained forest instance. |
| `averages_out` | `[out]` | Destination array of size dimensions. |

**Returns:** GEIF_OK on success, or an error code.

---

#### [`geif_forest_dimension_attribution`](../include/geif/geif.h#L140)

```c
geif_status_t geif_forest_dimension_attribution(const geif_forest_t *forest, const double *point, double *attr_scores_out);
```

**Description:** Computes single-dimension attribution / impact scores (%e) for each feature.

For each dimension j in [0, dimensions-1], evaluates the anomaly score of a synthetic
vector with coordinate j taken from point and all other coordinates at category mean baseline.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest` | `[in]` | Trained forest instance. |
| `point` | `[in]` | Query vector of length dimensions. |
| `attr_scores_out` | `[out]` | Destination array of size dimensions in [0.0, 1.0]. |

**Returns:** GEIF_OK on success, or an error code.

---

#### [`geif_forest_calculate_percentile_score`](../include/geif/geif.h#L151)

```c
double geif_forest_calculate_percentile_score(const geif_forest_t *forest, double percentile);
```

**Description:** Computes the percentile anomaly score across the training reservoir sample pool.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest` | `[in]` | Forest instance containing a populated sample pool. |
| `percentile` | `[in]` | Percentile threshold in range [0.0, 100.0] (e.g. 95.0). |

**Returns:** Anomaly score corresponding to the given percentile of training samples.

---

#### [`geif_forest_save_json`](../include/geif/geif.h#L160)

```c
geif_status_t geif_forest_save_json(const geif_forest_t *forest, const char *path);
```

**Description:** Serializes a trained GEIF forest to a JSON model file.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest` | `[in]` | Forest to save. |
| `path` | `[in]` | Target filesystem path. |

**Returns:** GEIF_OK on success, or an error code.

---

#### [`geif_forest_load_json`](../include/geif/geif.h#L169)

```c
geif_status_t geif_forest_load_json(geif_forest_t **forest_out, const char *path);
```

**Description:** Deserializes a GEIF forest from a JSON model file.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest_out` | `[out]` | Pointer to receive the loaded forest. |
| `path` | `[in]` | Source filesystem path. |

**Returns:** GEIF_OK on success, or an error code.

---

#### [`geif_forest_summary`](../include/geif/geif.h#L178)

```c
void geif_forest_summary(const geif_forest_t *forest, char *buffer, size_t buffer_size);
```

**Description:** Formats a human-readable diagnostic summary of the forest.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest` | `[in]` | Forest instance. |
| `buffer` | `[in]` | Destination text buffer. |
| `buffer_size` | `[in]` | Size of destination buffer. |

---

#### [`geif_ensemble_create`](../include/geif/geif.h#L187)

```c
geif_status_t geif_ensemble_create(geif_ensemble_t **ensemble_out, uint32_t dimensions, const geif_config_t *config);
```

**Description:** Allocates and initializes a multi-category forest ensemble.

---

#### [`geif_ensemble_destroy`](../include/geif/geif.h#L194)

```c
void geif_ensemble_destroy(geif_ensemble_t *ensemble);
```

**Description:** Frees all resources associated with an ensemble and its sub-forests.

---

#### [`geif_ensemble_find`](../include/geif/geif.h#L199)

```c
geif_forest_t *geif_ensemble_find(const geif_ensemble_t *ensemble, const char *category);
```

**Description:** Finds a sub-forest by category string. Returns NULL if not found.

---

#### [`geif_ensemble_get_or_create`](../include/geif/geif.h#L204)

```c
geif_forest_t *geif_ensemble_get_or_create(geif_ensemble_t *ensemble, const char *category);
```

**Description:** Finds or dynamically creates a sub-forest for a category string.

---

#### [`geif_ensemble_feed`](../include/geif/geif.h#L209)

```c
geif_status_t geif_ensemble_feed(geif_ensemble_t *ensemble, const char *category, const double *point);
```

**Description:** Feeds an observation vector into the sub-forest for category.

---

#### [`geif_ensemble_prune_categories`](../include/geif/geif.h#L216)

```c
geif_status_t geif_ensemble_prune_categories(geif_ensemble_t *ensemble, uint64_t min_rows);
```

**Description:** Prunes any sub-forest that has accumulated fewer than min_rows samples.

---

#### [`geif_ensemble_prune_age`](../include/geif/geif.h#L226)

```c
geif_status_t geif_ensemble_prune_age(geif_ensemble_t *ensemble, time_t max_age_seconds, time_t now);
```

**Description:** Prunes any sub-forest whose last_updated timestamp is older than max_age_seconds.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ensemble` | `[in]` | Ensemble object |
| `max_age_seconds` | `[in]` | Maximum age interval in seconds |
| `now` | `[in]` | Current timestamp reference (0 for time(NULL)) |

**Returns:** GEIF_OK on success

---

#### [`geif_ensemble_train`](../include/geif/geif.h#L233)

```c
geif_status_t geif_ensemble_train(geif_ensemble_t *ensemble);
```

**Description:** Trains all category sub-forests in the ensemble.

---

#### [`geif_ensemble_score_detailed`](../include/geif/geif.h#L238)

```c
geif_status_t geif_ensemble_score_detailed(const geif_ensemble_t *ensemble, const char *category, const double *point, double *score_out, double *metric_depth_out, double *d_out_out);
```

**Description:** Scores an observation against its category's sub-forest.

---

#### [`geif_ensemble_save_json`](../include/geif/geif.h#L248)

```c
geif_status_t geif_ensemble_save_json(const geif_ensemble_t *ensemble, const char *path);
```

**Description:** Serializes a multi-category ensemble to JSON.

---

#### [`geif_ensemble_load_json`](../include/geif/geif.h#L253)

```c
geif_status_t geif_ensemble_load_json(geif_ensemble_t **ensemble_out, const char *path);
```

**Description:** Deserializes a multi-category ensemble from JSON.

---

#### [`geif_ensemble_summary`](../include/geif/geif.h#L258)

```c
void geif_ensemble_summary(const geif_ensemble_t *ensemble, char *buffer, size_t buffer_size);
```

**Description:** Formats a diagnostic summary of the ensemble and all sub-forests.

---

#### [`geif_forest_remove_outliers`](../include/geif/geif.h#L267)

```c
geif_status_t geif_forest_remove_outliers(geif_forest_t *f, uint32_t k);
```

**Description:** Prunes the N most extreme outlier samples from a forest's reservoir pool and retrains it.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to forest. |
| `k` | `[in]` | Number of outliers to remove. |

**Returns:** GEIF_OK on success.

---

#### [`geif_ensemble_remove_outliers`](../include/geif/geif.h#L276)

```c
geif_status_t geif_ensemble_remove_outliers(geif_ensemble_t *ensemble, uint32_t k);
```

**Description:** Prunes the N most extreme outlier samples across all sub-forests in the ensemble.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ensemble` | `[in]` | Pointer to ensemble. |
| `k` | `[in]` | Number of outliers to remove per sub-forest. |

**Returns:** GEIF_OK on success.

---

### [`include/geif/types.h`](../include/geif/types.h)

**Module Purpose:** Core data structures and memory layouts for GEIF.

#### [`geif_algo_type_t`](../include/geif/types.h#L46)

```c
typedef enum geif_algo_type { GEIF_ALGO_CEIF = 0,        /**< CEIF: Data-anchored isotropic Gaussian cuts with Zero Kelvin & outer decay */ GEIF_ALGO_BUBBLE,          /**< Hyperspherical Bubble Cavity Carving */ GEIF_ALGO_EXEMPLAR,        /**< Non-tree direct SIMD Cauchy density kernel */ GEIF_ALGO_VORONOI          /**< Pure Voronoi perpendicular bisector splits */ } geif_algo_type_t;
```

**Description:** Selectable algorithm engines in GEIF.

---

### [`include/geif/error.h`](../include/geif/error.h)

**Module Purpose:** Diagnostic error codes and status indicators for GEIF.

#### [`geif_status_t`](../include/geif/error.h#L16)

```c
typedef enum geif_status { GEIF_OK                  =  0,  /**< Operation completed successfully */ GEIF_ERR_INVALID_ARG     = -1,  /**< Invalid argument or null pointer */ GEIF_ERR_OUT_OF_MEMORY   = -2,  /**< Memory allocation failure */ GEIF_ERR_IO              = -3,  /**< File I/O read/write error */ GEIF_ERR_FORMAT_CORRUPT  = -4,  /**< Input format or model file corrupted */ GEIF_ERR_EMPTY_DATASET   = -5,  /**< Dataset is empty or insufficient samples */ GEIF_ERR_DIM_MISMATCH    = -6,  /**< Feature dimension count mismatch */ GEIF_ERR_DEGENERATE_DATA = -7,  /**< All data points are completely degenerate */ GEIF_ERR_NOT_SUPPORTED   = -8   /**< Requested algorithm or feature is not supported */ } geif_status_t;
```

**Description:** Status and error codes returned by GEIF functions.

---

#### [`geif_status_str`](../include/geif/error.h#L34)

```c
const char *geif_status_str(geif_status_t status);
```

**Description:** Returns a static human-readable description of a status code.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `status` | `[in]` | The status code. |

**Returns:** A constant string describing the status.

---

## Algorithm Engines & Mathematical Core

Pluggable algorithm operations vtable, geometric partition trees (Bubble, Voronoi, CEIF), Exemplar kernel density, and spatial metric depth traversal.

### [`src/lib/algo.h`](../src/lib/algo.h)

**Module Purpose:** Internal algorithm abstraction interface and registry for GEIF.

#### [`geif_algo_get_ops`](../src/lib/algo.h#L34)

```c
const geif_algo_ops_t *geif_algo_get_ops(geif_algo_type_t algo);
```

**Description:** Retrieves the operations table for the specified algorithm type.

---

### [`src/lib/algo_registry.c`](../src/lib/algo_registry.c)

**Module Purpose:** Algorithm registry and dispatching table for GEIF.

#### [`geif_algo_name`](../src/lib/algo_registry.c#L16)

```c
const char *geif_algo_name(geif_algo_type_t algo);
```

**Description:** Returns the canonical string identifier for an algorithm type.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `algo` | `[in]` | Algorithm enum identifier (e.g. GEIF_ALGO_BUBBLE). |

**Returns:** Constant string: "ceif", "bubble", "exemplar", or "voronoi".

---

#### [`geif_algo_from_name`](../src/lib/algo_registry.c#L44)

```c
geif_algo_type_t geif_algo_from_name(const char *name);
```

**Description:** Parses a string algorithm name or alias into a geif_algo_type_t enum.

Supports canonical names and aliases (case-insensitive):
- "bubble", "spherical" -> GEIF_ALGO_BUBBLE
- "voronoi"             -> GEIF_ALGO_VORONOI
- "exemplar", "knn", "density" -> GEIF_ALGO_EXEMPLAR
- "ceif", "eif", "gaussian"    -> GEIF_ALGO_CEIF

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `name` | `[in]` | Algorithm string identifier. |

**Returns:** Parsed geif_algo_type_t enum (defaults to GEIF_ALGO_DEFAULT if unknown).

---

#### [`geif_algo_get_ops`](../src/lib/algo_registry.c#L80)

```c
const geif_algo_ops_t *geif_algo_get_ops(geif_algo_type_t algo);
```

**Description:** Retrieves the polymorphic operations dispatch table for an algorithm.

Returns pointer to the static geif_algo_ops_t table containing function pointers
for train, score, serialize, deserialize, and destroy.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `algo` | `[in]` | Algorithm type enum. |

**Returns:** Pointer to algorithm operations table.

---

### [`src/lib/algo_bubble.c`](../src/lib/algo_bubble.c)

**Module Purpose:** Bubble Algorithm: Hyperspherical cavity carving trees with empty void leaves.

#### [`swap_doubles`](../src/lib/algo_bubble.c#L16)

```c
static inline void swap_doubles(double *a, double *b);
```

**Description:** Swaps two double-precision floating point values in-place.

---

#### [`quickselect_median`](../src/lib/algo_bubble.c#L34)

```c
static double quickselect_median(double *arr, size_t n, size_t k);
```

**Description:** Finds the k-th smallest element using Hoare's Quickselect in O(N) average time.

Used to compute the median squared radius directly on squared Euclidean distances,
avoiding O(N log N) sorting and eliminating sqrt() operations entirely during training.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `arr` | `[in]` | Array of double values (partially reordered in-place). |
| `n` | `[in]` | Total number of elements in arr. |
| `k` | `[in]` | 0-based index of the desired order statistic (e.g. n / 2 for median). |

**Returns:** The value of the k-th smallest element.

---

#### [`build_bubble_node`](../src/lib/algo_bubble.c#L73)

```c
static int32_t build_bubble_node(geif_forest_t *f, geif_tree_t *tree, uint32_t *indices, size_t count, uint32_t depth, uint32_t max_depth, double *dists_scratch);
```

**Description:** Recursively constructs a hyperspherical Bubble tree node.

Selects an exemplar point as center, computes squared distances for all samples,
finds the median squared radius via O(N) quickselect, and partitions samples
in-place into interior (<= R^2) and exterior (> R^2) child branches.
Non-convex internal cavities terminate into empty void leaves.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `tree` | `[in]` | Pointer to the tree being built. |
| `indices` | `[in]` | Subarray of sample pool indices assigned to this node. |
| `count` | `[in]` | Number of samples in indices. |
| `depth` | `[in]` | Current tree depth from root. |
| `max_depth` | `[in]` | Maximum allowable tree depth. |
| `dists_scratch` | `[in]` | Reusable scratchpad for squared distances. |

**Returns:** Allocated node index, or -1 on allocation failure.

---

#### [`evaluate_bubble_tree`](../src/lib/algo_bubble.c#L194)

```c
static double evaluate_bubble_tree(const geif_forest_t *f, const geif_tree_t *tree, const double *scaled_point);
```

**Description:** Traverses a Bubble tree to evaluate continuous metric depth for a point.

Traverses inside the hypersphere if dist_sq(x, center) <= R^2, otherwise
branches outside. Applies leaf cavity relative distance damping at leaves.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `tree` | `[in]` | Pointer to the isolation tree. |
| `scaled_point` | `[in]` | Query point in normalized coordinate space. |

**Returns:** Accumulated metric depth H.

---

#### [`geif_bubble_train`](../src/lib/algo_bubble.c#L245)

```c
static geif_status_t geif_bubble_train(geif_forest_t *f);
```

**Description:** Trains an ensemble of hyperspherical Bubble isolation trees.

Normalizes dimension scales, draws subsamples per tree, and builds cavity-carving
trees with in-place median partitioning and Zero Kelvin universal calibration.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest instance. |

**Returns:** GEIF_OK on success, or error status.

---

#### [`geif_bubble_score`](../src/lib/algo_bubble.c#L314)

```c
static geif_status_t geif_bubble_score(const geif_forest_t *f, const double *point, double *score_out, double *metric_depth_out, double *d_out_out);
```

**Description:** Scores a sample observation against the trained Bubble forest.

Traverses all trees in the ensemble, accumulates average metric depth H_avg,
computes normalized outer space distance d_out to the bounding box, and applies
asymptotic exponential stadium attenuation: score = 1 - (1 - score)*exp(-0.10*d_out).

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `point` | `[in]` | Raw unscaled observation vector. |
| `score_out` | `[out]` | Pointer to receive calibrated anomaly score. |
| `metric_depth_out` | `[out]` | Optional pointer to receive average depth H_avg. |
| `d_out_out` | `[out]` | Optional pointer to receive outer distance d_out. |

**Returns:** GEIF_OK on success, or error status.

---

### [`src/lib/algo_voronoi.c`](../src/lib/algo_voronoi.c)

**Module Purpose:** Voronoi Algorithm: Pure perpendicular bisector splits between sample pairs.

#### [`build_voronoi_node`](../src/lib/algo_voronoi.c#L28)

```c
static int32_t build_voronoi_node(geif_forest_t *f, geif_tree_t *tree, uint32_t *indices, size_t count, uint32_t depth, uint32_t max_depth);
```

**Description:** Recursively builds a Voronoi perpendicular bisector tree node.

Randomly samples pairs of distinct observations from the node subset,
computes their perpendicular bisector hyperplane (n = (B - A) / ||B - A||,
p = (A + B) / 2), and partitions points based on the dot product sign.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `tree` | `[in]` | Pointer to the tree being built. |
| `indices` | `[in]` | Subarray of sample pool indices for this node. |
| `count` | `[in]` | Number of samples in indices. |
| `depth` | `[in]` | Current tree depth from root. |
| `max_depth` | `[in]` | Maximum allowable tree depth. |

**Returns:** Allocated node index, or -1 on error.

---

#### [`geif_voronoi_train`](../src/lib/algo_voronoi.c#L181)

```c
static geif_status_t geif_voronoi_train(geif_forest_t *f);
```

**Description:** Trains an ensemble of Voronoi perpendicular bisector trees.

Subsamples observations per tree and constructs binary trees using pure
midpoint perpendicular bisector cuts.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest instance. |

**Returns:** GEIF_OK on success, or error status.

---

#### [`geif_voronoi_score`](../src/lib/algo_voronoi.c#L240)

```c
static geif_status_t geif_voronoi_score(const geif_forest_t *f, const double *point, double *score_out, double *metric_depth_out, double *d_out_out);
```

**Description:** Evaluates calibrated anomaly score using the Voronoi bisector ensemble.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `point` | `[in]` | Raw unscaled observation vector. |
| `score_out` | `[out]` | Pointer to receive calibrated anomaly score. |
| `metric_depth_out` | `[out]` | Optional pointer to receive average depth H_avg. |
| `d_out_out` | `[out]` | Optional pointer to receive outer distance d_out. |

**Returns:** GEIF_OK on success, or error status.

---

### [`src/lib/algo_exemplar.c`](../src/lib/algo_exemplar.c)

**Module Purpose:** Exemplar Algorithm: Non-tree direct SIMD Cauchy density kernel on reservoir samples.

#### [`geif_exemplar_destroy`](../src/lib/algo_exemplar.c#L26)

```c
static void geif_exemplar_destroy(geif_forest_t *f);
```

**Description:** Releases heap-allocated adaptive bandwidth state for the exemplar model.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest instance. |

---

#### [`geif_exemplar_train`](../src/lib/algo_exemplar.c#L45)

```c
static geif_status_t geif_exemplar_train(geif_forest_t *f);
```

**Description:** Trains the Exemplar model by computing adaptive local bandwidths.

Bypasses binary tree construction entirely. For each sample in the reservoir pool,
finds its K-nearest neighbors and calculates local bandwidth sigma_i.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest instance. |

**Returns:** GEIF_OK on success, or error status.

---

#### [`geif_exemplar_score`](../src/lib/algo_exemplar.c#L156)

```c
static geif_status_t geif_exemplar_score(const geif_forest_t *f, const double *point, double *score_out, double *metric_depth_out, double *d_out_out);
```

**Description:** Scores a query point via direct Cauchy kernel density estimation.

Finds the K nearest exemplars in the reservoir pool and aggregates their
kernel density: D = (1/K) * sum(1 / (1 + (dist_i / sigma_i)^2)).
Score is evaluated as 1.0 - D with outer space stadium attenuation.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `point` | `[in]` | Raw unscaled observation vector. |
| `score_out` | `[out]` | Pointer to receive calibrated anomaly score. |
| `metric_depth_out` | `[out]` | Optional pointer to receive density-based depth. |
| `d_out_out` | `[out]` | Optional pointer to receive outer distance d_out. |

**Returns:** GEIF_OK on success, or error status.

---

### [`src/lib/algo_ceif.c`](../src/lib/algo_ceif.c)

**Module Purpose:** CEIF Algorithm: Data-anchored isotropic Gaussian hyperplanes with Zero Kelvin & outer decay.

#### [`build_ceif_node`](../src/lib/algo_ceif.c#L28)

```c
static int32_t build_ceif_node(geif_forest_t *f, geif_tree_t *tree, uint32_t *indices, size_t count, uint32_t depth, uint32_t max_depth);
```

**Description:** Recursively constructs a continuous Gaussian hyperplane node.

Anchors the intercept hyperplane between randomly selected sample pairs with
depth-dependent pairwise margin interpolation, and samples an isotropic
Gaussian normal vector.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `tree` | `[in]` | Pointer to the tree being built. |
| `indices` | `[in]` | Subarray of sample pool indices for this node. |
| `count` | `[in]` | Number of samples in indices. |
| `depth` | `[in]` | Current tree depth from root. |
| `max_depth` | `[in]` | Maximum allowable tree depth. |

**Returns:** Allocated node index, or -1 on error.

---

#### [`geif_ceif_train`](../src/lib/algo_ceif.c#L188)

```c
static geif_status_t geif_ceif_train(geif_forest_t *f);
```

**Description:** Trains an ensemble of continuous Gaussian hyperplane trees.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest instance. |

**Returns:** GEIF_OK on success, or error status.

---

#### [`geif_ceif_score`](../src/lib/algo_ceif.c#L247)

```c
static geif_status_t geif_ceif_score(const geif_forest_t *f, const double *point, double *score_out, double *metric_depth_out, double *d_out_out);
```

**Description:** Evaluates calibrated anomaly score using the continuous hyperplane ensemble.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `point` | `[in]` | Raw unscaled observation vector. |
| `score_out` | `[out]` | Pointer to receive calibrated anomaly score. |
| `metric_depth_out` | `[out]` | Optional pointer to receive average depth H_avg. |
| `d_out_out` | `[out]` | Optional pointer to receive outer distance d_out. |

**Returns:** GEIF_OK on success, or error status.

---

### [`src/lib/tree_common.c`](../src/lib/tree_common.c)

**Module Purpose:** Implementation of common tree memory allocation, traversal, and calibration.

#### [`init_dimension_scales`](../src/lib/tree_common.c#L22)

```c
void init_dimension_scales(geif_forest_t *f);
```

**Description:** Initializes dimension scales, bounding envelopes, and nominal cluster distances.

Scans the reservoir sample pool across all dimensions to compute coordinate extrema
([min, max]) and active spans. Identifies the dimension with the largest span to establish
a normalized isotropic coordinate space for distance-invariant calculations. Also pre-computes
average sample spacing (delta_nominal) and harmonic correction factor c(psi).

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in,out]` | Pointer to the forest instance. |

---

#### [`allocate_node`](../src/lib/tree_common.c#L97)

```c
int32_t allocate_node(geif_tree_t *tree);
```

**Description:** Dynamically allocates a new node slot in the tree's contiguous node pool.

Doubles capacity as needed starting from GEIF_INITIAL_TREE_NODES. Initializes children to -1.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `tree` | `[in,out]` | Pointer to the tree. |

**Returns:** Non-negative node index on success, or -1 on allocation failure.

---

#### [`append_normal`](../src/lib/tree_common.c#L123)

```c
uint32_t append_normal(geif_tree_t *tree, const double *normal, uint32_t d);
```

**Description:** Appends a normal vector or hypersphere center into the tree's contiguous float pool.

Expands the tree->normals_pool buffer dynamically as needed.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `tree` | `[in,out]` | Pointer to the tree. |
| `normal` | `[in]` | Normal or center coordinate vector of length d. |
| `d` | `[in]` | Dimensionality of the vector. |

**Returns:** Offset in tree->normals_pool where the normal vector is stored.

---

#### [`append_leaf_samples`](../src/lib/tree_common.c#L150)

```c
uint32_t append_leaf_samples(geif_tree_t *tree, const uint32_t *samples, size_t count);
```

**Description:** Appends sample pool indices into the tree's leaf sample repository.

Used for leaf-level nearest neighbor relative distance calculations. Expands
tree->leaf_samples dynamically starting from GEIF_INITIAL_LEAF_SAMPLES.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `tree` | `[in,out]` | Pointer to the tree. |
| `samples` | `[in]` | Array of sample indices. |
| `count` | `[in]` | Number of sample indices. |

**Returns:** Starting offset in tree->leaf_samples where indices are stored.

---

#### [`max_heap_sift_down`](../src/lib/tree_common.c#L178)

```c
static inline void max_heap_sift_down(double *heap, uint32_t i, uint32_t n);
```

**Description:** Restores the max-heap property by sifting down the element at index i.

Used to maintain the K smallest squared distances in the nearest-neighbor heap
without heap allocations.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `heap` | `[in,out]` | Array representing the binary max-heap. |
| `i` | `[in]` | Index of the element to sift down. |
| `n` | `[in]` | Total count of elements currently in the heap. |

---

#### [`geif_calc_leaf_rel_dist`](../src/lib/tree_common.c#L209)

```c
double geif_calc_leaf_rel_dist(const geif_forest_t *f, const geif_tree_t *tree, const geif_node_t *node, const double *scaled_point);
```

**Description:** Computes relative Euclidean distance from query point to nearest leaf samples.

In GEIF, relative leaf distance (rel_dist) scales effective sample density in leaf nodes
to prevent false positives in high-density regions and detect sparse interior voids/cavities.
For low dimensions (d < 5), it searches up to 2^d samples (with floor GEIF_MIN_LEAF_SAMPLE_FLOOR);
for d >= 5, it caps the nearest neighbor set at GEIF_MAX_LEAF_NEAREST_SAMPLES (32) to bound
computational complexity. Uses a stack-allocated binary max-heap to maintain the K smallest
squared distances without heap allocations.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `tree` | `[in]` | Pointer to the isolation tree. |
| `node` | `[in]` | Leaf node containing sample indices. |
| `scaled_point` | `[in]` | Query point in normalized coordinate space. |

**Returns:** Normalized relative distance factor (>= GEIF_REL_DIST_MULTIPLIER * GEIF_MIN_REL_DIST).

---

#### [`evaluate_tree`](../src/lib/tree_common.c#L308)

```c
double evaluate_tree(const geif_forest_t *f, const geif_tree_t *tree, const double *scaled_point);
```

**Description:** Traverses a single isolation tree to evaluate continuous metric path depth.

Recursively (or iteratively) navigates split hyperplanes (dot product test vs pdotn)
until reaching a terminal leaf. At the leaf, applies continuous metric depth estimation
adjusted by relative leaf distance (rel_dist) and the harmonic function c(n).

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `tree` | `[in]` | Pointer to the isolation tree. |
| `scaled_point` | `[in]` | Query point in normalized coordinate space. |

**Returns:** Accumulated continuous depth including leaf residual completion.

---

#### [`geif_tree_evaluate_metric_depth`](../src/lib/tree_common.c#L367)

```c
double geif_tree_evaluate_metric_depth(const geif_forest_t *f, const double *point, double *d_out_out);
```

**Description:** Evaluates average metric depth H across all trees in the forest.

Normalizes query coordinates across active bounding envelope dimensions and evaluates
the average path depth across all isolation trees in the ensemble. If the query point
lies outside the training envelope bounding box, computes the Euclidean outer space distance
d_out in scaled units for subsequent outer decay scoring.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `point` | `[in]` | Raw unscaled query point. |
| `d_out_out` | `[out]` | Optional pointer to receive outer space Euclidean distance. |

**Returns:** Average continuous metric depth across ensemble.

---

#### [`geif_tree_score_point`](../src/lib/tree_common.c#L439)

```c
geif_status_t geif_tree_score_point(const geif_forest_t *f, const double *point, double *score_out, double *metric_depth_out, double *d_out_out);
```

**Description:** Computes calibrated anomaly score, metric depth, and outer space distance for a query point.

Applies the canonical Isolation Forest exponential mapping s = 2^(-H / c(psi)), augmented by:
1. Outer space smooth asymptotic decay: s' = 1 - (1 - s) * exp(-GEIF_OUTER_DECAY_RATE * d_norm).
2. Zero Kelvin scale floor calibration: maps s_min (deepest observed training path) to 0.0.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `point` | `[in]` | Raw unscaled query coordinate vector of length d. |
| `score_out` | `[out]` | Pointer to receive anomaly score in [0..1). |
| `metric_depth_out` | `[out]` | Optional pointer to receive evaluated continuous metric depth H. |
| `d_out_out` | `[out]` | Optional pointer to receive scaled Euclidean distance beyond envelope. |

**Returns:** GEIF_OK on success, or GEIF_ERR_INVALID_ARG if invalid pointers are supplied.

---

#### [`geif_tree_find_max_height`](../src/lib/tree_common.c#L499)

```c
void geif_tree_find_max_height(const geif_forest_t *f, const geif_tree_t *t, int32_t node_idx, double depth, double *max_h);
```

**Description:** Recursively traverses an isolation tree to determine the maximum path depth.

Explores all branches to the deepest terminal leaf node to find the theoretical
maximum height H_max achievable by any point falling into this tree.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `t` | `[in]` | Pointer to the tree. |
| `node_idx` | `[in]` | Current node index. |
| `depth` | `[in]` | Accumulated split depth from root. |
| `max_h` | `[in,out]` | Pointer tracking the maximum depth discovered so far. |

---

#### [`geif_tree_calibrate`](../src/lib/tree_common.c#L545)

```c
void geif_tree_calibrate(geif_forest_t *f);
```

**Description:** Calibrates "Zero Kelvin" baseline scale floor and ensemble average inlier score.

Computes average maximum height H_train_max across all trained trees to establish
min_score = 2^(-H_train_max / c(psi)). Evaluates all training points in the sample pool
to compute the average baseline score and empirical dimension means.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in,out]` | Pointer to the forest instance. |

---

### [`src/lib/tree_common.h`](../src/lib/tree_common.h)

**Module Purpose:** Common tree manipulation, memory allocation, and evaluation utilities.

#### [`init_dimension_scales`](../src/lib/tree_common.h#L31)

```c
void init_dimension_scales(geif_forest_t *f);
```

**Description:** Initialize dimension scales, bounding envelopes, and nominal distances.

Scans the sample pool to compute coordinate minimums, maximums, and active
dimensions. Normalizes features across disparate aspect ratios.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest instance. |

---

#### [`allocate_node`](../src/lib/tree_common.h#L41)

```c
int32_t allocate_node(geif_tree_t *tree);
```

**Description:** Dynamically allocates a new node slot in the tree's node pool.

Grows tree->nodes buffer exponentially as needed.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `tree` | `[in]` | Pointer to the isolation tree. |

**Returns:** 0-based node index on success, or -1 on allocation failure.

---

#### [`append_normal`](../src/lib/tree_common.h#L51)

```c
uint32_t append_normal(geif_tree_t *tree, const double *normal, uint32_t d);
```

**Description:** Appends a normal vector or hypersphere center to the tree's float pool.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `tree` | `[in]` | Pointer to the isolation tree. |
| `normal` | `[in]` | Coordinate vector of length d. |
| `d` | `[in]` | Dimensionality of the feature vector. |

**Returns:** Offset in tree->normals_pool where vector was stored.

---

#### [`append_leaf_samples`](../src/lib/tree_common.h#L61)

```c
uint32_t append_leaf_samples(geif_tree_t *tree, const uint32_t *samples, size_t count);
```

**Description:** Stores indices of samples terminating in a leaf node.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `tree` | `[in]` | Pointer to the isolation tree. |
| `samples` | `[in]` | Array of sample indices in the forest pool. |
| `count` | `[in]` | Number of sample indices to append. |

**Returns:** Offset in tree->leaf_samples where indices start.

---

#### [`geif_calc_leaf_rel_dist`](../src/lib/tree_common.h#L75)

```c
double geif_calc_leaf_rel_dist(const geif_forest_t *f, const geif_tree_t *tree, const geif_node_t *node, const double *scaled_point);
```

**Description:** Computes relative distance from query point to nearest leaf samples.

Evaluates Euclidean distance to the 2^D nearest leaf samples (or heap-bounded
subset) relative to nominal cluster density (avg_sample_dist) for cavity damping.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `tree` | `[in]` | Pointer to the isolation tree. |
| `node` | `[in]` | Leaf node containing sample indices. |
| `scaled_point` | `[in]` | Query point in normalized coordinate space. |

**Returns:** Relative distance factor (>= GEIF_REL_DIST_MULTIPLIER * GEIF_MIN_REL_DIST).

---

#### [`evaluate_tree`](../src/lib/tree_common.h#L88)

```c
double evaluate_tree(const geif_forest_t *f, const geif_tree_t *tree, const double *scaled_point);
```

**Description:** Traverses a single isolation tree to evaluate continuous metric depth.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `tree` | `[in]` | Pointer to the tree. |
| `scaled_point` | `[in]` | Query point in normalized coordinate space. |

**Returns:** Accumulated continuous depth including leaf residual completion.

---

#### [`geif_tree_evaluate_metric_depth`](../src/lib/tree_common.h#L100)

```c
double geif_tree_evaluate_metric_depth(const geif_forest_t *f, const double *point, double *d_out_out);
```

**Description:** Evaluates average metric depth H across all trees in the forest.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `point` | `[in]` | Raw unscaled query point. |
| `d_out_out` | `[out]` | Optional pointer to receive outer space Euclidean distance. |

**Returns:** Average continuous metric depth across ensemble.

---

#### [`geif_tree_score_point`](../src/lib/tree_common.h#L114)

```c
geif_status_t geif_tree_score_point(const geif_forest_t *f, const double *point, double *score_out, double *metric_depth_out, double *d_out_out);
```

**Description:** Evaluates calibrated anomaly score, metric depth, and outer distance.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `point` | `[in]` | Raw unscaled query point. |
| `score_out` | `[out]` | Pointer to receive calibrated anomaly score [0..1). |
| `metric_depth_out` | `[out]` | Optional pointer to receive average depth H. |
| `d_out_out` | `[out]` | Optional pointer to receive outer space distance. |

**Returns:** GEIF_OK on success, or error status.

---

#### [`geif_tree_find_max_height`](../src/lib/tree_common.h#L131)

```c
void geif_tree_find_max_height(const geif_forest_t *f, const geif_tree_t *t, int32_t node_idx, double depth, double *max_h);
```

**Description:** Recursively traverses tree to find the maximum possible leaf height.

Used during Zero Kelvin calibration to determine the theoretical deepest path.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest. |
| `t` | `[in]` | Pointer to the tree. |
| `node_idx` | `[in]` | Current node index. |
| `depth` | `[in]` | Accumulated depth from root. |
| `max_h` | `[in,out]` | Pointer tracking maximum encountered path height. |

---

#### [`geif_tree_calibrate`](../src/lib/tree_common.h#L145)

```c
void geif_tree_calibrate(geif_forest_t *f);
```

**Description:** Calibrates Zero Kelvin universal scale floor and average score.

Traverses deepest leaf paths across all trees to calibrate H_max, s_min,
and ensemble average inlier baseline.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Pointer to the forest instance. |

---

### [`src/lib/geometry.h`](../src/lib/geometry.h)

**Module Purpose:** High-performance SIMD geometric routines for Voronoi bisectors and outer space stadium.

#### [`geif_scale_value`](../src/lib/geometry.h#L34)

```c
static inline double geif_scale_value(double value, double range, double scale_min, double min, double max);
```

**Description:** Linearly scales a value from [min, max] into a target range starting at scale_min.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `value` | `[in]` | Input value. |
| `range` | `[in]` | Span of target range. |
| `scale_min` | `[in]` | Minimum of target range. |
| `min` | `[in]` | Source domain lower bound. |
| `max` | `[in]` | Source domain upper bound. |

**Returns:** Linearly scaled value.

---

#### [`geif_dist_sq`](../src/lib/geometry.h#L50)

```c
static inline double geif_dist_sq(const double * restrict a, const double * restrict b, uint32_t d);
```

**Description:** Computes squared Euclidean distance between two d-dimensional points.

Provides specialized unrolled branches for 2D and 3D with compiler vectorization hints.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `a` | `[in]` | First point coordinate array. |
| `b` | `[in]` | Second point coordinate array. |
| `d` | `[in]` | Number of dimensions. |

**Returns:** Squared Euclidean distance.

---

#### [`geif_gaussrand`](../src/lib/geometry.h#L81)

```c
static inline double geif_gaussrand(void);
```

**Description:** Generates standard normal Gaussian random numbers using Marsaglia-Bray Box-Muller transform.

**Returns:** Pseudo-random sample from standard normal distribution N(0, 1).

---

#### [`geif_dot`](../src/lib/geometry.h#L102)

```c
static inline double geif_dot(const double * restrict a, const double * restrict b, uint32_t d);
```

**Description:** Evaluates vector dot product sum(a[j] * b[j]) for j = 0..d-1. Uses restrict pointer annotations for compiler autovectorization.

---

#### [`geif_stadium_distance`](../src/lib/geometry.h#L122)

```c
static inline double geif_stadium_distance(const double * restrict x, const double * restrict env_min, const double * restrict env_max, const double * restrict effective_span, uint32_t d);
```

**Description:** Computes normalized Euclidean outer space distance outside the bounding stadium. Points inside [min, max] have distance 0.0.

---

#### [`geif_compute_bisector`](../src/lib/geometry.h#L156)

```c
static inline bool geif_compute_bisector(const double * restrict A, const double * restrict B, const double * restrict effective_span, const uint8_t * restrict dim_active, uint32_t d, double * restrict normal_out, double * restrict pdotn_out, double * restrict delta_out);
```

**Description:** Calculates pre-baked scale-invariant Voronoi bisector hyperplane parameters.

Normal vector: n_j = (B_j - A_j) / (span_j^2)
Scalar threshold: p_scalar = sum_j n_j * (A_j + B_j) / 2
Generator distance: delta = sqrt(sum_j ((B_j - A_j) / span_j)^2)

Returns false if points A and B are identical (delta < 1e-12).

---

#### [`geif_residual_distance`](../src/lib/geometry.h#L197)

```c
static inline double geif_residual_distance(const double * restrict x, const double * restrict P_leaf, const double * restrict effective_span, const uint8_t * restrict dim_active, uint32_t d);
```

**Description:** Computes normalized residual Euclidean distance from query point x to leaf point P.

---

## Forest, Ensembles & Evaluation Engine

Streaming reservoir ingestion, multi-category ensemble routing, calibration, inlier baselines, dimension attribution, and sparse JSON I/O.

### [`src/lib/forest.c`](../src/lib/forest.c)

**Module Purpose:** Lifecycle management, initialization, and deallocation for GEIF forests.

#### [`geif_config_default`](../src/lib/forest.c#L26)

```c
geif_config_t geif_config_default(void);
```

**Description:** Returns the default configuration for GEIF forest creation.

Defaults:
- tree_count:       GEIF_DEFAULT_TREE_COUNT (100)
- samples_per_tree: GEIF_DEFAULT_SAMPLES_PER_TREE (256)
- max_depth:        GEIF_DEFAULT_MAX_DEPTH (adaptive or 32)
- kappa:            GEIF_DEFAULT_KAPPA (1.15 headroom)
- alpha:            GEIF_DEFAULT_ALPHA (1.0 density sensitivity)
- algo:             GEIF_ALGO_DEFAULT (Hyperspherical Bubble trees)

**Returns:** Populated geif_config_t structure with factory defaults.

---

#### [`geif_forest_create`](../src/lib/forest.c#L52)

```c
geif_status_t geif_forest_create(geif_forest_t **forest_out, uint32_t dimensions, const geif_config_t *config);
```

**Description:** Allocates and initializes a new GEIF forest instance.

Sets up dynamic arrays for coordinate envelopes (min, max, span, effective span),
active dimension tracking, dimension averages, the contiguous reservoir sample pool,
and individual tree structures. Seeds PRNG if seed is 0.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest_out` | `[out]` | Pointer to receive allocated forest handle. |
| `dimensions` | `[in]` | Number of feature dimensions (must be > 0). |
| `config` | `[in]` | Pointer to user configuration, or NULL for default configuration. |

**Returns:** GEIF_OK on success, or GEIF_ERR_INVALID_ARG / GEIF_ERR_OUT_OF_MEMORY on failure.

---

#### [`geif_forest_destroy`](../src/lib/forest.c#L132)

```c
void geif_forest_destroy(geif_forest_t *f);
```

**Description:** Deallocates all resources associated with a GEIF forest.

Traverses individual trees to free per-tree node arrays, normal vectors,
and leaf samples. Calls algorithm-specific cleanup hook if registered,
frees all coordinate envelopes, sample pools, dimension averages,
and finally frees the forest container. Safe to invoke with NULL.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in,out]` | Forest instance to destroy. |

---

#### [`geif_forest_summary`](../src/lib/forest.c#L179)

```c
void geif_forest_summary(const geif_forest_t *f, char *buf, size_t size);
```

**Description:** Generates a multi-line human-readable summary of forest diagnostic metrics.

Formats dimensionality, active dimensions, tree counts, sample pool utilization,
rows seen, nominal spacing, and calibrated maximum height/scale factor into
the destination string buffer.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Forest instance. |
| `buf` | `[out]` | Destination string buffer. |
| `size` | `[in]` | Capacity of the destination buffer in bytes. |

---

### [`src/lib/train.c`](../src/lib/train.c)

**Module Purpose:** Unified training dispatcher for GEIF algorithms.

#### [`geif_forest_train`](../src/lib/train.c#L19)

```c
geif_status_t geif_forest_train(geif_forest_t *f);
```

**Description:** Trains all trees in the forest using the algorithm configured in forest config.

Validates forest presence and ensures the reservoir pool is non-empty. Dispatches
training via the polymorphic algorithm operations table (geif_algo_ops_t) matching
f->config.algo (e.g., bubble, voronoi, exemplar, ceif).

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in,out]` | Forest instance containing populated reservoir samples. |

**Returns:** GEIF_OK on success, GEIF_ERR_EMPTY_DATASET if no samples, or an error status code.

---

### [`src/lib/evaluate.c`](../src/lib/evaluate.c)

**Module Purpose:** Inference engine, continuous metric depth traversal, leaf void damping, and outer stadium decay.

#### [`geif_init_c_cache`](../src/lib/evaluate.c#L21)

```c
void geif_init_c_cache(void);
```

**Description:** Precomputes average path length normalization factors c(n) for small sample sizes.

Populates s_fast_c_cache for indices 0 to GEIF_FAST_C_SAMPLES using iterative
harmonic number accumulation to avoid expensive logarithmic approximations.

---

#### [`geif_c`](../src/lib/evaluate.c#L44)

```c
double geif_c(double n);
```

**Description:** Computes the standard Isolation Forest average path length expectation c(n).

Uses cached harmonic values for n < GEIF_FAST_C_SAMPLES, and the Euler-Mascheroni
logarithmic approximation with 2nd-order expansion for larger n:
c(n) = 2 * (ln(n-1) + 0.5772156649 + 1/(2*(n-1)) - 1/(12*(n-1)^2)) - 2*(n-1)/n.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `n` | `[in]` | Subsample count. |

**Returns:** Expected path length c(n), or 0.0 if n <= 1.

---

#### [`geif_forest_evaluate_metric_depth`](../src/lib/evaluate.c#L71)

```c
double geif_forest_evaluate_metric_depth(const geif_forest_t *f, const double *point, double *d_out_out);
```

**Description:** Evaluates the raw uncalibrated continuous metric depth across all trees.

Traverses trees dispatching to the configured algorithm operations table,
computing continuous tree traversal depth and exterior bounding box Euclidean distance.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Forest instance. |
| `point` | `[in]` | Feature vector. |
| `d_out_out` | `[out]` | Optional pointer to receive exterior Euclidean distance. |

**Returns:** Average continuous metric depth across the ensemble.

---

#### [`geif_forest_score_detailed`](../src/lib/evaluate.c#L104)

```c
geif_status_t geif_forest_score_detailed(const geif_forest_t *f, const double *point, double *score_out, double *metric_depth_out, double *d_out_out);
```

**Description:** Evaluates an observation with full diagnostic metrics.

Dispatches to the active algorithm engine to compute:
1. Normalized anomaly score in [0.0, 1.0).
2. Unscaled continuous metric depth H(x).
3. Exterior normalized stadium distance d_out.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Forest instance. |
| `point` | `[in]` | Feature vector to score. |
| `score_out` | `[out]` | Receives calibrated anomaly score. |
| `metric_depth_out` | `[out]` | Optional pointer to receive unscaled depth. |
| `d_out_out` | `[out]` | Optional pointer to receive exterior distance. |

**Returns:** GEIF_OK on success, or an error status code.

---

#### [`geif_forest_score`](../src/lib/evaluate.c#L130)

```c
geif_status_t geif_forest_score(const geif_forest_t *f, const double *point, double *score_out);
```

**Description:** Evaluates the calibrated anomaly score for an observation.

Convenience wrapper around geif_forest_score_detailed().

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Forest instance. |
| `point` | `[in]` | Feature vector to score. |
| `score_out` | `[out]` | Receives calibrated anomaly score in [0.0, 1.0). |

**Returns:** GEIF_OK on success, or error code.

---

#### [`geif_forest_get_averages`](../src/lib/evaluate.c#L147)

```c
geif_status_t geif_forest_get_averages(const geif_forest_t *forest, double *averages_out);
```

**Description:** Computes or retrieves per-dimension mean coordinate averages for the forest.

Looks up forest->averages first, falls back to computing the mean across all
reservoir pool samples, or takes the midpoint of the coordinate bounding envelope.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest` | `[in]` | Forest instance. |
| `averages_out` | `[out]` | Array of length forest->dimensions receiving baseline averages. |

**Returns:** GEIF_OK on success, or GEIF_ERR_INVALID_ARG on NULL arguments.

---

#### [`geif_forest_dimension_attribution`](../src/lib/evaluate.c#L194)

```c
geif_status_t geif_forest_dimension_attribution(const geif_forest_t *forest, const double *point, double *attr_scores_out);
```

**Description:** Computes single-dimension attribution / impact scores (%e) for an observation.

Isolates each feature's marginal contribution to anomaly score by replacing that
dimension in the category baseline average vector with the observation's value,
evaluating the resulting score.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest` | `[in]` | Forest instance. |
| `point` | `[in]` | Observation feature vector. |
| `attr_scores_out` | `[out]` | Array of length forest->dimensions receiving attribution scores. |

**Returns:** GEIF_OK on success, or error status code.

---

#### [`pscore_cmp`](../src/lib/evaluate.c#L234)

```c
static int pscore_cmp(const void *a, const void *b);
```

**Description:** Comparator for sorting double precision anomaly scores ascendingly.

---

#### [`geif_forest_calculate_percentile_score`](../src/lib/evaluate.c#L253)

```c
double geif_forest_calculate_percentile_score(const geif_forest_t *forest, double percentile);
```

**Description:** Computes the empirical anomaly score threshold corresponding to a given percentile.

Scores all observations currently in the forest's reservoir sample pool, sorts
the score array ascendingly via qsort, and returns the score at rank (N - 1) * (percentile / 100).

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest` | `[in]` | Forest instance. |
| `percentile` | `[in]` | Target percentile in [0.0, 100.0] (e.g. 80.0, 94.0). |

**Returns:** Empirical score cutoff at that percentile rank.

---

### [`src/lib/ensemble.c`](../src/lib/ensemble.c)

**Module Purpose:** Multi-category sub-forest ensemble management, dynamic routing, and lifecycle.

#### [`ensemble_hash`](../src/lib/ensemble.c#L21)

```c
static uint32_t ensemble_hash(const char *str);
```

**Description:** Computes 32-bit FNV-1a hash for string category keys.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `str` | `[in]` | Null-terminated category name string. |

**Returns:** 32-bit hash value.

---

#### [`ensemble_rehash`](../src/lib/ensemble.c#L37)

```c
static void ensemble_rehash(geif_ensemble_t *ens);
```

**Description:** Doubles hash table capacity and rehashes all category hash nodes.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in,out]` | Ensemble instance whose hash table is expanding. |

---

#### [`geif_ensemble_create`](../src/lib/ensemble.c#L71)

```c
geif_status_t geif_ensemble_create(geif_ensemble_t **ensemble_out, uint32_t dimensions, const geif_config_t *config);
```

**Description:** Allocates and initializes a multi-category GEIF ensemble router.

Allocates dynamic category entries array and hash bucket array for fast O(1)
sub-forest dispatching based on category labels.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ensemble_out` | `[out]` | Pointer receiving allocated ensemble handle. |
| `dimensions` | `[in]` | Feature vector dimensionality. |
| `config` | `[in]` | Configuration options copied to each newly created sub-forest. |

**Returns:** GEIF_OK on success, or GEIF_ERR_INVALID_ARG / GEIF_ERR_OUT_OF_MEMORY.

---

#### [`geif_ensemble_destroy`](../src/lib/ensemble.c#L114)

```c
void geif_ensemble_destroy(geif_ensemble_t *ens);
```

**Description:** Destroys an ensemble and all associated category sub-forests.

Traverses entries to destroy all sub-forests, frees hash buckets and chain nodes,
and frees the ensemble container. Safe to invoke with NULL.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in,out]` | Ensemble instance to destroy. |

---

#### [`geif_ensemble_find`](../src/lib/ensemble.c#L152)

```c
geif_forest_t *geif_ensemble_find(const geif_ensemble_t *ens, const char *category);
```

**Description:** Searches for an existing category's sub-forest in the ensemble hash table.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in]` | Ensemble instance. |
| `category` | `[in]` | Category string name (empty string or NULL refers to default forest). |

**Returns:** Pointer to matching geif_forest_t, or NULL if category has not been created.

---

#### [`geif_ensemble_get_or_create`](../src/lib/ensemble.c#L183)

```c
geif_forest_t *geif_ensemble_get_or_create(geif_ensemble_t *ens, const char *category);
```

**Description:** Looks up a category's sub-forest, or allocates and registers it if not found.

Automatically expands the category entries dynamic array and rehashes the lookup
table when load factor exceeds 75%. Propagates configuration and column slicing
specifications to the newly instantiated sub-forest.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in,out]` | Ensemble instance. |
| `category` | `[in]` | Category name string. |

**Returns:** Pointer to existing or newly created geif_forest_t, or NULL on error.

---

#### [`geif_ensemble_feed`](../src/lib/ensemble.c#L254)

```c
geif_status_t geif_ensemble_feed(geif_ensemble_t *ens, const char *category, const double *point);
```

**Description:** Feeds an observation vector into a category's sub-forest.

Looks up or instantiates the sub-forest for category, updates the sub-forest's
row count and timestamp, and passes the vector to geif_forest_feed() for
streaming reservoir sampling.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in,out]` | Ensemble instance. |
| `category` | `[in]` | Category name (or empty string/NULL for default). |
| `point` | `[in]` | Observation feature vector. |

**Returns:** GEIF_OK on success, or an error status code.

---

#### [`geif_ensemble_prune_categories`](../src/lib/ensemble.c#L290)

```c
geif_status_t geif_ensemble_prune_categories(geif_ensemble_t *ens, uint64_t min_rows);
```

**Description:** Prunes categories having fewer than min_rows training samples (-R).

Destroys underpopulated sub-forests, compacts the active category entries list,
and rebuilds the category hash table.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in,out]` | Ensemble instance. |
| `min_rows` | `[in]` | Minimum sample count required to keep a category sub-forest. |

**Returns:** GEIF_OK on success, or GEIF_ERR_INVALID_ARG on NULL.

---

#### [`geif_ensemble_prune_age`](../src/lib/ensemble.c#L344)

```c
geif_status_t geif_ensemble_prune_age(geif_ensemble_t *ens, time_t max_age_seconds, time_t now);
```

**Description:** Prunes categories whose last ingestion timestamp is older than max_age_seconds.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in,out]` | Ensemble instance. |
| `max_age_seconds` | `[in]` | Retention window in seconds (e.g. from -D days). |
| `now` | `[in]` | Current Unix epoch timestamp (0 uses time(NULL)). |

**Returns:** GEIF_OK on success, or GEIF_ERR_INVALID_ARG on NULL.

---

#### [`geif_ensemble_train`](../src/lib/ensemble.c#L402)

```c
geif_status_t geif_ensemble_train(geif_ensemble_t *ens);
```

**Description:** Trains all sub-forests in the ensemble that contain reservoir samples.

Iterates through each category entry, calling geif_forest_train() on any
sub-forest with pool_count > 0.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in,out]` | Ensemble instance. |

**Returns:** GEIF_OK on success, GEIF_ERR_EMPTY_DATASET if no sub-forests exist, or an error code.

---

#### [`geif_ensemble_score_detailed`](../src/lib/ensemble.c#L434)

```c
geif_status_t geif_ensemble_score_detailed(const geif_ensemble_t *ens, const char *category, const double *point, double *score_out, double *metric_depth_out, double *d_out_out);
```

**Description:** Scores an observation against its category's sub-forest.

Performs O(1) hash table lookup for category. If found, evaluates using that
sub-forest and marks seen_in_analysis = true. If category was never seen during
training, returns maximum outlier score (1.0 - 1e-6) and GEIF_ERR_INVALID_ARG.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in]` | Ensemble instance. |
| `category` | `[in]` | Category name of observation. |
| `point` | `[in]` | Feature vector. |
| `score_out` | `[out]` | Receives calibrated anomaly score. |
| `metric_depth_out` | `[out]` | Optional pointer to receive metric depth. |
| `d_out_out` | `[out]` | Optional pointer to receive exterior distance. |

**Returns:** GEIF_OK on success, or GEIF_ERR_INVALID_ARG if category is unknown.

---

#### [`geif_ensemble_summary`](../src/lib/ensemble.c#L480)

```c
void geif_ensemble_summary(const geif_ensemble_t *ens, char *buffer, size_t buffer_size);
```

**Description:** Formats a diagnostic summary of the ensemble and all sub-forests.

Prints total dimensions, sub-forest count, column specs, and detailed statistics
for each sub-forest (rows seen, pool size, tree count, calibrated H_max, timestamp).

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in]` | Ensemble instance. |
| `buffer` | `[out]` | Destination text buffer. |
| `buffer_size` | `[in]` | Buffer capacity in bytes. |

---

#### [`geif_forest_remove_outliers`](../src/lib/ensemble.c#L536)

```c
geif_status_t geif_forest_remove_outliers(geif_forest_t *f, uint32_t k);
```

**Description:** Prunes the k worst anomaly outliers from a forest's reservoir sample pool (-k).

Iteratively scores all current reservoir samples against the forest, locates
the sample with the highest anomaly score, removes it from the sample pool via
memmove, and retrains the trees to recalibrate spatial boundaries without outlier bias.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in,out]` | Forest instance. |
| `k` | `[in]` | Number of worst outlier samples to prune. |

**Returns:** GEIF_OK on success, or an error status code.

---

#### [`geif_ensemble_remove_outliers`](../src/lib/ensemble.c#L583)

```c
geif_status_t geif_ensemble_remove_outliers(geif_ensemble_t *ens, uint32_t k);
```

**Description:** Prunes the k worst outlier samples across all sub-forests in an ensemble.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in,out]` | Ensemble instance. |
| `k` | `[in]` | Number of outlier samples to prune per category. |

**Returns:** GEIF_OK on success, or an error code.

---

### [`src/lib/reservoir.c`](../src/lib/reservoir.c)

**Module Purpose:** Streaming reservoir sampling with adaptive ceiling factor.

#### [`geif_forest_feed`](../src/lib/reservoir.c#L25)

```c
geif_status_t geif_forest_feed(geif_forest_t *f, const double *point);
```

**Description:** Ingests an observation vector into the forest's streaming reservoir sample pool.

Implements Algorithm R reservoir sampling with an optional ceiling factor cap:
1. Increment total_rows_seen and expands the coordinate bounding box envelope.
2. If pool_count < pool_capacity (Phase 1), point is stored directly in the pool.
3. Once capacity is reached (Phase 2), uses a 64-bit pseudo-random replacement
index modulo effective_seen (bounded by ceiling_factor * capacity to prevent
sample freezing over very long streams). If slot falls within capacity, the
existing sample in that slot is replaced.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in,out]` | Forest instance. |
| `point` | `[in]` | Array of double coordinates of length f->dimensions. |

**Returns:** GEIF_OK on success, or GEIF_ERR_INVALID_ARG if arguments are NULL.

---

### [`src/lib/json_io.c`](../src/lib/json_io.c)

**Module Purpose:** JSON serialization and deserialization for GEIF models and ensembles.

#### [`geif_clean_double_json`](../src/lib/json_io.c#L26)

```c
static struct json_object *geif_clean_double_json(double val, int decimals);
```

**Description:** Formats a floating-point value to JSON with trimmed trailing zeros.

Avoids precision bloat in saved JSON files by formatting with specified decimals
and removing trailing decimal zeros and negative zeros ("-0").

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `val` | `[in]` | Floating point value. |
| `decimals` | `[in]` | Maximum decimal precision. |

**Returns:** Allocated json_object representing the double value.

---

#### [`geif_forest_to_json_object`](../src/lib/json_io.c#L53)

```c
struct json_object *geif_forest_to_json_object(const geif_forest_t *f);
```

**Description:** Serializes a geif_forest_t into a json-c json_object.

Encodes forest hyperparameters, coordinate envelopes, dimension metadata,
calibration parameters, and the raw reservoir sample pool.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Forest instance to serialize. |

**Returns:** Pointer to newly allocated json_object root, or NULL on error.

---

#### [`geif_forest_from_json_object`](../src/lib/json_io.c#L122)

```c
geif_status_t geif_forest_from_json_object(geif_forest_t **forest_out, struct json_object *root);
```

**Description:** Deserializes a json-c json_object into a populated geif_forest_t.

Instantiates the forest, populates envelopes and sample pool, and automatically
builds and calibrates the tree structures in RAM.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest_out` | `[out]` | Pointer receiving the newly allocated forest. |
| `root` | `[in]` | json-c root object representing the forest. |

**Returns:** GEIF_OK on success, or GEIF_ERR_FORMAT_CORRUPT / GEIF_ERR_OUT_OF_MEMORY.

---

#### [`geif_forest_save_json`](../src/lib/json_io.c#L340)

```c
geif_status_t geif_forest_save_json(const geif_forest_t *f, const char *path);
```

**Description:** Serializes a single forest and writes it to a file or stdout in formatted JSON.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `f` | `[in]` | Forest instance to save. |
| `path` | `[in]` | File destination path (or "-" for stdout). |

**Returns:** GEIF_OK on success, or GEIF_ERR_IO / GEIF_ERR_INVALID_ARG on failure.

---

#### [`geif_ensemble_save_json`](../src/lib/json_io.c#L369)

```c
geif_status_t geif_ensemble_save_json(const geif_ensemble_t *ens, const char *path);
```

**Description:** Serializes an ensemble containing multiple category sub-forests to JSON.

Emits global ensemble configuration, category routing dimensions, and an array
of sub-forest objects with their respective sample pools and metadata.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in]` | Ensemble instance to save. |
| `path` | `[in]` | File destination path (or "-" for stdout). |

**Returns:** GEIF_OK on success, or GEIF_ERR_IO / GEIF_ERR_INVALID_ARG on failure.

---

#### [`geif_ensemble_load_json`](../src/lib/json_io.c#L437)

```c
geif_status_t geif_ensemble_load_json(geif_ensemble_t **ensemble_out, const char *path);
```

**Description:** Reads and parses a JSON model file or stdin into a geif_ensemble_t.

Supports native GEIF models, multi-category ensemble JSON files, and legacy CEIF
model schemas, dynamically reconstructing memory structures and building trees.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ensemble_out` | `[out]` | Pointer receiving the loaded ensemble. |
| `path` | `[in]` | File path or "-" to read from stdin. |

**Returns:** GEIF_OK on success, or an error status code.

---

#### [`geif_forest_load_json`](../src/lib/json_io.c#L667)

```c
geif_status_t geif_forest_load_json(geif_forest_t **forest_out, const char *path);
```

**Description:** Reads and parses a single forest model from a JSON file.

Convenience function that loads the model via geif_ensemble_load_json()
and extracts its primary/first sub-forest.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `forest_out` | `[out]` | Pointer receiving the loaded forest. |
| `path` | `[in]` | File path or "-" to read from stdin. |

**Returns:** GEIF_OK on success, or an error status code.

---

### [`src/lib/error.c`](../src/lib/error.c)

**Module Purpose:** Error code string formatting.

#### [`geif_status_str`](../src/lib/error.c#L14)

```c
const char *geif_status_str(geif_status_t status);
```

**Description:** Translates a geif_status_t numeric error code into a human-readable English description.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `status` | `[in]` | Status code to convert. |

**Returns:** Static string pointer containing diagnostic message.

---

## CLI Frontend, Tools & Infrastructure

Command-line pipeline driver, streaming loops, OOM-safe memory allocation, column extraction, color companding, RC configuration parsing, and legacy model migration.

### [`src/cli/main.c`](../src/cli/main.c)

**Module Purpose:** GEIF CLI frontend for training, inference, and model management.

#### [`print_usage`](../src/cli/main.c#L28)

```c
static void print_usage(const char *prog);
```

**Description:** Displays command-line syntax and comprehensive option reference for GEIF CLI.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `prog` | `[in]` | Executable program name (argv[0]). |

---

#### [`tokenize_line`](../src/cli/main.c#L87)

```c
static uint32_t tokenize_line(char *line, char delim, char **tokens, uint32_t max_tokens);
```

**Description:** Tokenizes a delimited text line with quote awareness and whitespace trimming.

Strips carriage returns and newlines, parses quoted strings without splitting on interior
delimiters, and trims surrounding whitespace. Modifies the input line buffer in-place.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `line` | `[in,out]` | Input character buffer to tokenize. |
| `delim` | `[in]` | Delimiter character (e.g. ',', '\t', ';'). |
| `tokens` | `[out]` | Array of string pointers to receive token starts. |
| `max_tokens` | `[in]` | Maximum token capacity of tokens array. |

**Returns:** Number of tokens parsed.

---

#### [`parse_delete_interval`](../src/cli/main.c#L141)

```c
static time_t parse_delete_interval(const char *s);
```

**Description:** Parses human-readable duration strings (e.g. "30d", "24h", "60m") into seconds.

Supports units: 'y' (years), 'm'/'M' (months or minutes depending on context),
'w' (weeks), 'd' (days), 'h' (hours), 's' (seconds). Defaults to days if unit is omitted.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `s` | `[in]` | Duration string (e.g. "7d"). |

**Returns:** Duration in seconds, or 0 on error.

---

#### [`process_scoring_row`](../src/cli/main.c#L221)

```c
static void process_scoring_row(geif_ensemble_t *ensemble, const geif_column_config_t *col_cfg, const cat_filter_t *cat_filter, char **tokens, uint32_t n_tok, const char *orig_line, uint64_t line_num, char list_sep, char cat_sep, double threshold, bool threshold_is_average, bool threshold_is_percentage, bool silent_outliers, const char *point_tmpl, const char *average_tmpl, const char *new_cat_tmpl, int decimals, const char *printf_format, const char *print_dimension, uint32_t low_rgb, uint32_t high_rgb, double *vec, uint32_t dims, FILE *out_fp, uint64_t *analyzed, uint64_t *total_outliers);
```

**Description:** Evaluates an incoming CSV record against an ensemble during streaming analysis (-a).

Extracts category and label, verifies category filters, routes to the corresponding
sub-forest (or flags as an unseen category), computes anomaly score and metric depth,
determines outlier status against threshold, and formats output via templates.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ensemble` | `[in,out]` | Category ensemble model. |
| `col_cfg` | `[in]` | Column mapping configuration. |
| `cat_filter` | `[in]` | Active category filter rules (-F). |
| `tokens` | `[in]` | Parsed text fields of current row. |
| `n_tok` | `[in]` | Token count. |
| `orig_line` | `[in]` | Original unparsed text line. |
| `line_num` | `[in]` | 1-based row counter. |
| `list_sep` | `[in]` | Output list separator character (-e). |
| `cat_sep` | `[in]` | Category delimiter character (-C). |
| `threshold` | `[in]` | Outlier decision boundary [0, 1]. |
| `threshold_is_average` | `[in]` | True if threshold is calibrated to category mean score. |
| `threshold_is_percentage` | `[in]` | True if threshold is percentile-based. |
| `silent_outliers` | `[in]` | True to suppress nominal inlier rows (-S). |
| `point_tmpl` | `[in]` | Output format template for scored points (-p). |
| `average_tmpl` | `[in]` | Output format template for nominal inliers (-v). |
| `new_cat_tmpl` | `[in]` | Output format template for unseen categories (-N). |
| `decimals` | `[in]` | Floating point precision (-d). |
| `printf_format` | `[in]` | Dimension format override string (-m). |
| `print_dimension` | `[in]` | Per-dimension template expansion (-j). |
| `low_rgb` | `[in]` | Low-score RGB color hex. |
| `high_rgb` | `[in]` | High-score RGB color hex. |
| `vec` | `[in,out]` | Scratch buffer for feature vector. |
| `dims` | `[in]` | Number of feature dimensions. |
| `out_fp` | `[in,out]` | Output file stream. |
| `analyzed` | `[in,out]` | Cumulative analyzed rows counter. |
| `total_outliers` | `[in,out]` | Cumulative outliers detected counter. |

---

#### [`process_categorize_row`](../src/cli/main.c#L414)

```c
static void process_categorize_row(geif_ensemble_t *ensemble, const geif_column_config_t *col_cfg, bool direct_feature_mapping, const cat_filter_t *cat_filter, char **tokens, uint32_t n_tok, const char *orig_line, uint64_t line_num, char list_sep, char cat_sep, double threshold, bool threshold_is_average, bool threshold_is_percentage, bool score_limit_given, const char *point_tmpl, int decimals, const char *printf_format, const char *print_dimension, uint32_t low_rgb, uint32_t high_rgb, double *vec, uint32_t dims, FILE *out_fp, uint64_t *analyzed);
```

**Description:** Classifies an unassigned input sample against all ensemble categories (-c).

Scores the sample across all trained category sub-forests permitted by category filters,
selects the category yielding the minimum anomaly score (closest geometric fit),
checks outlier threshold limits, and outputs the assigned classification.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ensemble` | `[in,out]` | Category ensemble model. |
| `col_cfg` | `[in]` | Column mapping configuration. |
| `direct_feature_mapping` | `[in]` | True if input has no label/category columns (features only). |
| `cat_filter` | `[in]` | Active category filter rules (-F). |
| `tokens` | `[in]` | Parsed text fields of current row. |
| `n_tok` | `[in]` | Token count. |
| `orig_line` | `[in]` | Original unparsed text line. |
| `line_num` | `[in]` | 1-based row counter. |
| `list_sep` | `[in]` | Output list separator character (-e). |
| `cat_sep` | `[in]` | Category delimiter character (-C). |
| `threshold` | `[in]` | Outlier threshold limit [0, 1]. |
| `threshold_is_average` | `[in]` | True if threshold is calibrated to category mean score. |
| `threshold_is_percentage` | `[in]` | True if threshold is percentile-based. |
| `score_limit_given` | `[in]` | True if -O outlier threshold was specified on CLI. |
| `point_tmpl` | `[in]` | Output format template (-p). |
| `decimals` | `[in]` | Floating point precision (-d). |
| `printf_format` | `[in]` | Dimension format override string (-m). |
| `print_dimension` | `[in]` | Per-dimension template expansion (-j). |
| `low_rgb` | `[in]` | Low-score RGB color hex. |
| `high_rgb` | `[in]` | High-score RGB color hex. |
| `vec` | `[in,out]` | Scratch buffer for feature vector. |
| `dims` | `[in]` | Number of feature dimensions. |
| `out_fp` | `[in,out]` | Output file stream. |
| `analyzed` | `[in,out]` | Cumulative analyzed rows counter. |

---

#### [`update_ensemble_percentage_scores`](../src/cli/main.c#L562)

```c
static void update_ensemble_percentage_scores(geif_ensemble_t *ens, double pct, bool verbose);
```

**Description:** Recalculates empirical percentile threshold scores for all sub-forests in an ensemble.

Evaluates reservoir samples to locate the empirical score boundary corresponding
to the given cumulative percentile (e.g. 80% or 95%).

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in,out]` | Category ensemble instance. |
| `pct` | `[in]` | Percentile value (0.0 to 100.0). |
| `verbose` | `[in]` | True to print per-category percentile diagnostics. |

---

#### [`main`](../src/cli/main.c#L589)

```c
int main(int argc, char *argv[]);
```

**Description:** Main entry point for the GEIF CLI application.

Handles CLI command-line arguments, RC configuration files, model loading/saving,
training on CSV datasets, reservoir streaming, outlier pruning (-k),
synthetic test grid generation (-T), and real-time anomaly inference (-a/-c).

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `argc` | `[in]` | Argument count. |
| `argv` | `[in]` | Argument vector. |

**Returns:** 0 on success, non-zero exit code on failure.

---

### [`src/cli/columns.c`](../src/cli/columns.c)

**Module Purpose:** Column range specification parsing and feature masking for GEIF CLI.

#### [`trim_whitespace`](../src/cli/columns.c#L20)

```c
static char *trim_whitespace(char *str);
```

**Description:** Trims leading and trailing ASCII whitespace in-place.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `str` | `[in,out]` | Input string. |

**Returns:** Pointer to first non-whitespace character in str.

---

#### [`geif_parse_dim_spec`](../src/cli/columns.c#L41)

```c
int geif_parse_dim_spec(const char *spec, uint32_t *indices, uint32_t max_indices);
```

**Description:** Parses a comma-separated 1-based column range string into 0-based indices.

Parses tokens such as "1,3,5-8" into an array of distinct 0-based indices
[0, 2, 4, 5, 6, 7] while discarding duplicates.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `spec` | `[in]` | Column specification string (e.g. "1-4,7"). |
| `indices` | `[out]` | Destination array receiving 0-based indices. |
| `max_indices` | `[in]` | Maximum capacity of indices array. |

**Returns:** Number of resolved column indices, or -1 on syntax/range error.

---

#### [`geif_has_col_index`](../src/cli/columns.c#L113)

```c
bool geif_has_col_index(uint32_t col_idx, const uint32_t *list, uint32_t count);
```

**Description:** Tests whether a given column index exists within an index array.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `col_idx` | `[in]` | Column index to query. |
| `list` | `[in]` | Array of column indices. |
| `count` | `[in]` | Number of elements in list. |

**Returns:** True if col_idx is present, false otherwise.

---

#### [`geif_column_config_init`](../src/cli/columns.c#L134)

```c
void geif_column_config_init(geif_column_config_t *cfg, const char *ignore_spec, const char *include_spec, const char *label_spec, const char *category_spec);
```

**Description:** Initializes a geif_column_config_t structure from CLI argument strings.

Copies and parses the raw column specification strings for ignore (-I),
include (-U), label (-L), and category (-C).

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `cfg` | `[out]` | Column configuration object. |
| `ignore_spec` | `[in]` | Specification string for ignored columns (or NULL). |
| `include_spec` | `[in]` | Specification string for included columns (or NULL). |
| `label_spec` | `[in]` | Specification string for label columns (or NULL). |
| `category_spec` | `[in]` | Specification string for category columns (or NULL). |

---

#### [`geif_column_config_free`](../src/cli/columns.c#L170)

```c
void geif_column_config_free(geif_column_config_t *cfg);
```

**Description:** Frees dynamically allocated specification strings in column configuration.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `cfg` | `[in,out]` | Column configuration instance to release. |

---

#### [`geif_column_config_resolve`](../src/cli/columns.c#L192)

```c
bool geif_column_config_resolve(geif_column_config_t *cfg, uint32_t total_cols);
```

**Description:** Resolves active feature column indices based on total input columns.

Determines the set of numerical feature columns by applying precedence:
1. Label columns (-L) are excluded.
2. Category columns (-C) are excluded.
3. Included columns (-U), if specified, must contain the column.
4. Ignored columns (-I) are excluded.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `cfg` | `[in,out]` | Column configuration instance. |
| `total_cols` | `[in]` | Total number of fields detected in CSV header/row. |

**Returns:** True if at least one feature column was resolved, false otherwise.

---

#### [`geif_extract_features`](../src/cli/columns.c#L239)

```c
bool geif_extract_features(const geif_column_config_t *cfg, char **tokens, uint32_t total_cols, double *vec);
```

**Description:** Extracts continuous numerical feature values from parsed text tokens into a double vector.

Iterates through active feature column indices, parsing each field with strtod().
Replaces non-numeric tokens, NaNs, and infinities with 0.0.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `cfg` | `[in]` | Resolved column configuration. |
| `tokens` | `[in]` | Array of string pointers representing fields in current row. |
| `total_cols` | `[in]` | Number of tokens available in row. |
| `vec` | `[out]` | Destination double array (size >= cfg->feature_dim_count). |

**Returns:** True on success, false if input arguments are invalid.

---

#### [`geif_extract_label`](../src/cli/columns.c#L277)

```c
void geif_extract_label(const geif_column_config_t *cfg, char **tokens, uint32_t total_cols, char sep, char *out_buf, size_t max_len);
```

**Description:** Concatenates values of configured label columns (-L) into an output string.

Joins label fields using the specified separator character.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `cfg` | `[in]` | Column configuration instance. |
| `tokens` | `[in]` | Parsed string tokens from row. |
| `total_cols` | `[in]` | Total tokens available in row. |
| `sep` | `[in]` | Delimiter character (e.g. '/' or '_'). |
| `out_buf` | `[out]` | Destination buffer for composite label string. |
| `max_len` | `[in]` | Capacity of destination buffer. |

---

#### [`geif_extract_category`](../src/cli/columns.c#L327)

```c
void geif_extract_category(const geif_column_config_t *cfg, char **tokens, uint32_t total_cols, char sep, char *out_buf, size_t max_len);
```

**Description:** Concatenates values of configured category columns (-C) into a routing key.

Joins category fields using the specified separator character to form
the lookup key for ensemble sub-forest dispatching.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `cfg` | `[in]` | Column configuration instance. |
| `tokens` | `[in]` | Parsed string tokens from row. |
| `total_cols` | `[in]` | Total tokens available in row. |
| `sep` | `[in]` | Delimiter character. |
| `out_buf` | `[out]` | Destination buffer for composite category string. |
| `max_len` | `[in]` | Capacity of destination buffer. |

---

### [`src/cli/columns.h`](../src/cli/columns.h)

**Module Purpose:** Column range specification parsing and feature masking for GEIF CLI.

#### [`geif_parse_dim_spec`](../src/cli/columns.h#L44)

```c
int geif_parse_dim_spec(const char *spec, uint32_t *indices, uint32_t max_indices);
```

**Description:** Parse a 1-based comma/hyphen range string into 0-based indices. Example: "1,3,5-7" -> [0, 2, 4, 5, 6], count = 5.

**Returns:** Number of unique parsed indices, or -1 on error.

---

#### [`geif_column_config_init`](../src/cli/columns.h#L49)

```c
void geif_column_config_init(geif_column_config_t *cfg, const char *ignore_spec, const char *include_spec, const char *label_spec, const char *category_spec);
```

**Description:** Initialize column configuration from CLI option strings.

---

#### [`geif_column_config_free`](../src/cli/columns.h#L58)

```c
void geif_column_config_free(geif_column_config_t *cfg);
```

**Description:** Free any heap-allocated strings in column configuration.

---

#### [`geif_column_config_resolve`](../src/cli/columns.h#L65)

```c
bool geif_column_config_resolve(geif_column_config_t *cfg, uint32_t total_cols);
```

**Description:** Compute the feature column mapping for a row with total_cols columns. Excludes label, category, ignored, and non-included columns. Sets cfg->feature_dim_count and cfg->feature_cols[].

---

#### [`geif_has_col_index`](../src/cli/columns.h#L70)

```c
bool geif_has_col_index(uint32_t col_idx, const uint32_t *list, uint32_t count);
```

**Description:** Check if a 0-based column index is present in an index list.

---

#### [`geif_extract_features`](../src/cli/columns.h#L76)

```c
bool geif_extract_features(const geif_column_config_t *cfg, char **tokens, uint32_t total_cols, double *vec);
```

**Description:** Extract feature vector from row tokens according to column config. Handles NaNs, infinities, and missing fields by defaulting to 0.0.

---

#### [`geif_extract_label`](../src/cli/columns.h#L85)

```c
void geif_extract_label(const geif_column_config_t *cfg, char **tokens, uint32_t total_cols, char sep, char *out_buf, size_t max_len);
```

**Description:** Build label string from row tokens according to cfg->label_indices. Concatenates label tokens with sep. Buffer out_buf is populated.

---

#### [`geif_extract_category`](../src/cli/columns.h#L96)

```c
void geif_extract_category(const geif_column_config_t *cfg, char **tokens, uint32_t total_cols, char sep, char *out_buf, size_t max_len);
```

**Description:** Build category string from row tokens according to cfg->category_indices. Concatenates category tokens with sep. If no category columns, out_buf is set to "".

---

### [`src/cli/template.c`](../src/cli/template.c)

**Module Purpose:** Full-featured output templating and attribution engine for GEIF CLI.

#### [`srgb_companding`](../src/cli/template.c#L19)

```c
static inline void srgb_companding(double *color);
```

**Description:** Applies sRGB non-linear companding (gamma transfer function) to RGB components.

Implements standard IEC 61966-2-1 transfer function to map linear light intensity
to perception-corrected display RGB values.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `color` | `[in,out]` | Array of 3 doubles (R, G, B in range [0, 255]). |

---

#### [`score_to_rgb`](../src/cli/template.c#L40)

```c
static uint32_t score_to_rgb(double score, uint32_t low_rgb, uint32_t high_rgb);
```

**Description:** Interpolates between two hex colors according to anomaly score with sRGB companding.

Clamps score to [0.0, 1.0], blends R/G/B channels linearly, and applies companding
to produce visually uniform color gradients for UI/CLI output (%x).

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `score` | `[in]` | Anomaly score in [0.0, 1.0]. |
| `low_rgb` | `[in]` | 24-bit hex color for nominal / low score (e.g. 0xFFFF00 yellow). |
| `high_rgb` | `[in]` | 24-bit hex color for anomaly / high score (e.g. 0xFF0000 red). |

**Returns:** 24-bit RGB packed integer.

---

#### [`format_double`](../src/cli/template.c#L78)

```c
static int format_double(char *buf, size_t buf_sz, double val, int decimals, const char *fmt);
```

**Description:** Helper to format floating point values according to decimals or custom format.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `buf` | `[out]` | Destination text buffer. |
| `buf_sz` | `[in]` | Capacity of buffer. |
| `val` | `[in]` | Value to format. |
| `decimals` | `[in]` | Number of decimal places (or negative for %g). |
| `fmt` | `[in]` | Optional printf format override string (or NULL). |

**Returns:** Number of characters written.

---

#### [`geif_format_template`](../src/cli/template.c#L102)

```c
size_t geif_format_template(char *out, size_t out_size, const char *tmpl, const geif_template_context_t *ctx);
```

**Description:** Expands an output template string using evaluation context variables.

Evaluates tokens such as %s (score), %l (label), %c (category), %m (metric depth),
%d (feature dimensions), %e (attribution scores), %x (RGB color), and %v (raw row).

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `out` | `[out]` | Destination buffer receiving expanded string. |
| `out_size` | `[in]` | Capacity of out buffer in bytes. |
| `tmpl` | `[in]` | Format template string (e.g. "%l: score=%s (%x)"). |
| `ctx` | `[in]` | Template evaluation context populated during scoring. |

**Returns:** Number of bytes written excluding trailing null terminator.

---

### [`src/cli/template.h`](../src/cli/template.h)

**Module Purpose:** Output templating engine for GEIF CLI (-p, -v, -N, -M).

#### [`geif_format_template`](../src/cli/template.h#L80)

```c
size_t geif_format_template(char *out, size_t out_size, const char *tmpl, const geif_template_context_t *ctx);
```

**Description:** Formats a template string by replacing % specifiers with context values.

Supported specifiers:
%s - Anomaly score (formatted with precision -d)
%S - Anomaly score percentage (%.2f%%)
%l - Label string
%c - Category string
%C - Category string
%m - Dimension expansion with -j template, or continuous metric depth H
%d - Feature dimension values list (joined by -e), or outer stadium distance
%e - Single-dimension impact / attribution scores list (joined by -e)
%a - Category dimension averages list (joined by -e), or raw input line
%v - Raw token column values (joined by -e), or raw input line
%x - 6-hex RGB color interpolated by score between low_rgb and high_rgb
%h - Forest H_max universal scale
%o - Outlier flag (0 or 1)
%n - Category total training rows
%t - Epoch timestamp (%ld)
%% - Literal percent sign

Inside dimension template (-j, expanded by %m):
%d - Feature value for current dimension (formatted with -m printf_format or -d)
%a - Category average value for current dimension
%e - Single-dimension impact / attribution score for current dimension
%i - 1-based feature dimension index
%% - Literal percent sign

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `out` | `[in]` | Output buffer |
| `out_size` | `[in]` | Capacity of output buffer |
| `tmpl` | `[in]` | Template string |
| `ctx` | `[in]` | Context values |

**Returns:** Number of characters written (excluding null terminator)

---

### [`src/cli/rcfile.c`](../src/cli/rcfile.c)

**Module Purpose:** Run-command (RC) configuration file parser for GEIF.

#### [`geif_rc_config_init`](../src/cli/rcfile.c#L17)

```c
void geif_rc_config_init(geif_rc_config_t *rc);
```

**Description:** Initializes run-command configuration to factory defaults.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `rc` | `[out]` | RC configuration structure to initialize. |

---

#### [`expand_path`](../src/cli/rcfile.c#L31)

```c
static void expand_path(const char *in, char *out, size_t out_size);
```

**Description:** Expands leading tilde ('~') in a file path to the user's HOME directory.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `in` | `[in]` | Input path string. |
| `out` | `[out]` | Output buffer for expanded absolute path. |
| `out_size` | `[in]` | Capacity of output buffer. |

---

#### [`extract_value`](../src/cli/rcfile.c#L54)

```c
static char *extract_value(char *line, const char *key);
```

**Description:** Extracts a configuration value matching a given key name from a line.

Handles case-insensitive key comparison, optional whitespace, equal signs,
and double quotes around string values.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `line` | `[in,out]` | Line buffer from configuration file. |
| `key` | `[in]` | Directive key name (e.g. "TREES"). |

**Returns:** Pointer to trimmed value string inside line buffer, or NULL if no match.

---

#### [`geif_rc_parse_file`](../src/cli/rcfile.c#L93)

```c
bool geif_rc_parse_file(geif_rc_config_t *rc, const char *filepath);
```

**Description:** Parses an RC configuration file and populates an rc_config structure.

Reads directives such as TREES, SAMPLES, DECIMALS, OUTLIER_SCORE, PRINT_DIMENSION,
and hex color codes. Ignores comments ('#') and blank lines.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `rc` | `[in,out]` | RC configuration instance. |
| `filepath` | `[in]` | Path to configuration file. |

**Returns:** True if file was read and parsed successfully, false on I/O error.

---

#### [`geif_rc_load_default`](../src/cli/rcfile.c#L166)

```c
bool geif_rc_load_default(geif_rc_config_t *rc);
```

**Description:** Discovers and loads the default user RC configuration file.

Checks in priority order:
1. ~/.geifrc
2. ~/.ceifrc (legacy fallback)

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `rc` | `[in,out]` | RC configuration instance. |

**Returns:** True if a file was loaded or if no configuration file was present.

---

### [`src/cli/rcfile.h`](../src/cli/rcfile.h)

**Module Purpose:** Run-command (RC) configuration file parser for GEIF.

#### [`geif_rc_config_init`](../src/cli/rcfile.h#L48)

```c
void geif_rc_config_init(geif_rc_config_t *rc);
```

**Description:** Initialize an rc config struct with default zeroed values.

---

#### [`geif_rc_parse_file`](../src/cli/rcfile.h#L54)

```c
bool geif_rc_parse_file(geif_rc_config_t *rc, const char *filepath);
```

**Description:** Parse a single RC file into the config struct. Expands '~' if leading path. Returns true on success, false if file cannot be opened.

---

#### [`geif_rc_load_default`](../src/cli/rcfile.h#L60)

```c
bool geif_rc_load_default(geif_rc_config_t *rc);
```

**Description:** Load default global RC file (~/.geifrc, or fallback to ~/.ceifrc if present). If neither exists, returns true (not an error).

---

### [`src/cli/test_grid.c`](../src/cli/test_grid.c)

**Module Purpose:** Population drift visualization and test grid generator for GEIF.

#### [`geif_cat_filter_add`](../src/cli/test_grid.c#L25)

```c
bool geif_cat_filter_add(cat_filter_t *cf, const char *arg);
```

**Description:** Adds a regex pattern to the category filter list.

Supports `-v <regex>` prefix for inverting matches (keep only matches).
Matching categories are filtered out by default.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `cf` | `[in]` | Pointer to the category filter structure. |
| `arg` | `[in]` | Command-line argument string containing regex pattern. |

**Returns:** true on successful regex compilation, false on error.

---

#### [`geif_cat_filter_allows`](../src/cli/test_grid.c#L67)

```c
bool geif_cat_filter_allows(const cat_filter_t *cf, const char *category);
```

**Description:** Checks if a category string is permitted by active regex filters.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `cf` | `[in]` | Pointer to filter structure (NULL or inactive allows all). |
| `category` | `[in]` | Category name string to test. |

**Returns:** true if category should be processed, false if filtered out.

---

#### [`geif_cat_filter_free`](../src/cli/test_grid.c#L92)

```c
void geif_cat_filter_free(cat_filter_t *cf);
```

**Description:** Releases compiled regex resources in a category filter.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `cf` | `[in]` | Pointer to category filter structure. |

---

#### [`geif_generate_test_grid`](../src/cli/test_grid.c#L127)

```c
void geif_generate_test_grid(const geif_ensemble_t *ens, double test_extension_factor, int test_sample_interval, const cat_filter_t *filter, double threshold, bool threshold_is_average, bool threshold_is_percentage, const char *point_tmpl, int decimals, char list_sep, uint32_t low_rgb, uint32_t high_rgb, const char *printf_format, const char *print_dimension, FILE *out_fp);
```

**Description:** Synthesizes an N-dimensional uniform test grid across sample bounds.

Evaluates anomaly scores over an odometer-stepped coordinate lattice to
visualize decision manifolds, population drift, and cluster contours.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in]` | Trained ensemble containing sub-forests. |
| `test_extension_factor` | `[in]` | Margin expansion factor outside bounding box (e.g. 0.1). |
| `test_sample_interval` | `[in]` | Number of grid steps along each dimension. |
| `filter` | `[in]` | Optional category regex filter. |
| `threshold` | `[in]` | Anomaly score cutoff. |
| `threshold_is_average` | `[in]` | Whether threshold is locked to ensemble mean. |
| `threshold_is_percentage` | `[in]` | Whether threshold is locked to sample percentile. |
| `point_tmpl` | `[in]` | Output template string for each grid point. |
| `decimals` | `[in]` | Floating point output decimal precision. |
| `list_sep` | `[in]` | Field separator character. |
| `low_rgb` | `[in]` | Hex RGB color for inliers (score 0). |
| `high_rgb` | `[in]` | Hex RGB color for anomalies (score 1). |
| `printf_format` | `[in]` | Format string for coordinates. |
| `print_dimension` | `[in]` | Dimension filter pattern. |
| `out_fp` | `[in]` | Destination file stream. |

---

### [`src/cli/test_grid.h`](../src/cli/test_grid.h)

**Module Purpose:** Population drift visualization and test grid generator for GEIF.

#### [`geif_generate_test_grid`](../src/cli/test_grid.h#L57)

```c
void geif_generate_test_grid(const geif_ensemble_t *ens, double test_extension_factor, int test_sample_interval, const cat_filter_t *filter, double threshold, bool threshold_is_average, bool threshold_is_percentage, const char *point_tmpl, int decimals, char list_sep, uint32_t low_rgb, uint32_t high_rgb, const char *printf_format, const char *print_dimension, FILE *out_fp);
```

**Description:** Generate a synthetic evaluation grid and training sample scatter points for population drift visualization.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ens` | `[in]` | Ensemble model containing categories and sub-forests |
| `test_extension_factor` | `[in]` | Factor to extend grid beyond bounding box (e.g. 0.1) |
| `test_sample_interval` | `[in]` | Number of grid subdivisions per dimension (default 256) |
| `filter` | `[in]` | Optional category filter regex |
| `threshold_is_average` | `[in]` | If true, filter grid points using forest average score |
| `threshold_is_percentage` | `[in]` | If true, filter grid points using forest percentage score |
| `point_tmpl` | `[in]` | Template for formatting points (e.g. "%d,0x%x") |
| `decimals` | `[in]` | Decimal precision |
| `list_sep` | `[in]` | Separator for list elements (e.g. ',') |
| `low_rgb` | `[in]` | Low score RGB hex color |
| `high_rgb` | `[in]` | High score RGB hex color |
| `printf_format` | `[in]` | Custom printf format |
| `print_dimension` | `[in]` | Dimension template (%j) |
| `out_fp` | `[in]` | Output file stream |

---

### [`src/cli/xmalloc.c`](../src/cli/xmalloc.c)

**Module Purpose:** Memory allocation and stream wrappers with out-of-memory checking.

#### [`panic_oom`](../src/cli/xmalloc.c#L33)

```c
static void panic_oom(size_t n);
```

**Description:** Prints an out-of-memory error message to stderr and terminates process.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `n` | `[in]` | Number of bytes that failed allocation. |

---

#### [`xmalloc`](../src/cli/xmalloc.c#L45)

```c
void *xmalloc(size_t n);
```

**Description:** Allocates heap memory with out-of-memory abort and heap usage tracking.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `n` | `[in]` | Number of bytes to allocate (if 0, allocates 1 byte). |

**Returns:** Pointer to allocated memory buffer.

---

#### [`xcalloc`](../src/cli/xmalloc.c#L67)

```c
void *xcalloc(size_t count, size_t size);
```

**Description:** Allocates zero-initialized heap memory with out-of-memory abort.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `count` | `[in]` | Number of elements. |
| `size` | `[in]` | Size of each element. |

**Returns:** Pointer to zero-initialized memory.

---

#### [`xrealloc`](../src/cli/xmalloc.c#L92)

```c
void *xrealloc(void *ptr, size_t n);
```

**Description:** Reallocates heap memory buffer with out-of-memory checking.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ptr` | `[in]` | Existing memory pointer (or NULL to malloc). |
| `n` | `[in]` | New size in bytes. |

**Returns:** Pointer to reallocated memory.

---

#### [`xstrdup`](../src/cli/xmalloc.c#L119)

```c
char *xstrdup(const char *s);
```

**Description:** Duplicates a string using xmalloc with null-termination and memory tracking.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `s` | `[in]` | Source string to duplicate. |

**Returns:** Newly allocated copy of string (or NULL if s is NULL).

---

#### [`xfree`](../src/cli/xmalloc.c#L133)

```c
void xfree(void *ptr);
```

**Description:** Frees memory buffer and decrements heap usage tracking counter.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `ptr` | `[in]` | Pointer to allocated memory to free. |

---

#### [`xfopen`](../src/cli/xmalloc.c#L153)

```c
FILE *xfopen(const char *path, const char *mode);
```

**Description:** Opens a file stream with transparent stdin/stdout support for "-".

Prevents multiple simultaneous open attempts on stdin/stdout, and emits
descriptive errno messages if fopen fails.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `path` | `[in]` | File path or "-" for standard streams. |
| `mode` | `[in]` | Open mode ("r", "w", "a", etc.). |

**Returns:** Opened FILE pointer, or NULL on error.

---

#### [`xfclose`](../src/cli/xmalloc.c#L190)

```c
int xfclose(FILE *fp);
```

**Description:** Closes a file stream, safely flushing but not closing stdin/stdout/stderr.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `fp` | `[in]` | FILE stream pointer to close. |

**Returns:** 0 on success, or EOF on error.

---

#### [`xget_total_allocated`](../src/cli/xmalloc.c#L214)

```c
size_t xget_total_allocated(void);
```

**Description:** Returns total active bytes currently tracked across all allocations.

**Returns:** Allocation total in bytes.

---

#### [`xreset_total_allocated`](../src/cli/xmalloc.c#L222)

```c
void xreset_total_allocated(void);
```

**Description:** Resets the active allocation counter back to 0.

---

### [`src/cli/xmalloc.h`](../src/cli/xmalloc.h)

**Module Purpose:** Memory allocation and file stream wrappers with OOM checking for GEIF.

#### [`xmalloc`](../src/cli/xmalloc.h#L15)

```c
void *xmalloc(size_t n);
```

**Description:** Allocates heap memory with out-of-memory abort and heap usage tracking.

---

#### [`xcalloc`](../src/cli/xmalloc.h#L20)

```c
void *xcalloc(size_t count, size_t size);
```

**Description:** Allocates zero-initialized heap memory with out-of-memory abort.

---

#### [`xrealloc`](../src/cli/xmalloc.h#L25)

```c
void *xrealloc(void *ptr, size_t n);
```

**Description:** Reallocates heap memory buffer with out-of-memory checking.

---

#### [`xstrdup`](../src/cli/xmalloc.h#L30)

```c
char *xstrdup(const char *s);
```

**Description:** Duplicates a string using xmalloc with null-termination and memory tracking.

---

#### [`xfree`](../src/cli/xmalloc.h#L35)

```c
void  xfree(void *ptr);
```

**Description:** Frees memory buffer and decrements heap usage tracking counter.

---

#### [`xfopen`](../src/cli/xmalloc.h#L40)

```c
FILE *xfopen(const char *path, const char *mode);
```

**Description:** Opens a file stream with transparent stdin/stdout support for "-".

---

#### [`xfclose`](../src/cli/xmalloc.h#L45)

```c
int   xfclose(FILE *fp);
```

**Description:** Closes a file stream, safely flushing but not closing stdin/stdout/stderr.

---

#### [`xget_total_allocated`](../src/cli/xmalloc.h#L50)

```c
size_t xget_total_allocated(void);
```

**Description:** Returns total active bytes currently tracked across all allocations.

---

#### [`xreset_total_allocated`](../src/cli/xmalloc.h#L55)

```c
void   xreset_total_allocated(void);
```

**Description:** Resets the active allocation counter back to 0.

---

### [`src/cli/ceif2geif.c`](../src/cli/ceif2geif.c)

**Module Purpose:** CEIF JSON model to GEIF-1.0 JSON model migration tool.

#### [`print_usage`](../src/cli/ceif2geif.c#L22)

```c
static void print_usage(const char *prog);
```

**Description:** Prints command-line help and usage instructions for ceif2geif.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `prog` | `[in]` | Executable program name (argv[0]). |

---

#### [`main`](../src/cli/ceif2geif.c#L49)

```c
int main(int argc, char *argv[]);
```

**Description:** Main entry point for ceif2geif model migration utility.

Ingests legacy CEIF JSON models, reconstructs sample reservoirs and trees,
optionally applies retrain overrides, and outputs clean GEIF-1.0 JSON models.

**Parameters:**

| Parameter | Direction | Description |
| :--- | :--- | :--- |
| `argc` | `[in]` | Argument count. |
| `argv` | `[in]` | Argument vector. |

**Returns:** 0 on success, non-zero exit code on error.

---

