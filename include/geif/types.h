/**
 * @file types.h
 * @brief Core data structures and memory layouts for GEIF.
 */

#ifndef GEIF_TYPES_H
#define GEIF_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEIF_DEFAULT_TREE_COUNT        100
#define GEIF_DEFAULT_SAMPLES_PER_TREE  256
#define GEIF_DEFAULT_MAX_DEPTH         0
#define GEIF_DEFAULT_KAPPA             1.25     /**< Headroom for "Zero Kelvin" calibration */
#define GEIF_DEFAULT_ALPHA             1.0      /**< Metric depth density sensitivity */
#define GEIF_DEFAULT_CEILING_FACTOR    3        /**< Extra rows factor for reservoir ceiling */

/**
 * @brief Configuration parameters for training and evaluating GEIF.
 */
typedef struct geif_config {
    uint32_t tree_count;           /**< Number of trees in the forest (default: 100) */
    uint32_t samples_per_tree;     /**< Sub-sample size psi per tree (default: 256) */
    uint32_t max_depth;            /**< Hard tree depth cap (default: 16) */
    double   kappa;                /**< Absolute Zero Kelvin headroom factor (default: 1.25) */
    double   alpha;                /**< Metric depth density sensitivity (default: 1.0) */
    uint32_t ceiling_factor;       /**< Reservoir ceiling factor (default: 3) */
    uint32_t seed;                 /**< RNG seed (0 for auto / time-based) */
} geif_config_t;

/**
 * @brief Compact cache-aligned Voronoi node structure.
 */
typedef struct geif_node {
    int32_t  left_child;           /**< Index of left child in tree's node array (-1 if leaf) */
    int32_t  right_child;          /**< Index of right child in tree's node array (-1 if leaf) */
    uint32_t normal_offset;        /**< Offset into tree's normal pool: &normals_pool[normal_offset] */
    double   pdotn;                /**< Precomputed scalar threshold: sum(p_j * n_j) */
    double   step_weight;          /**< Continuous metric increment Delta H for this cut */
    double   delta_AB;             /**< Normalized generator separation ||B - A|| */
    int32_t  sample_count;         /**< Number of samples remaining in this node */
    uint32_t leaf_point_idx;       /**< Sample pool index of representative point (leaf only) */
    uint32_t leaf_sample_offset;   /**< Offset into tree's leaf_samples array where indices begin */
} geif_node_t;

/**
 * @brief Representation of an individual tree in contiguous memory.
 */
typedef struct geif_tree {
    geif_node_t *nodes;            /**< Contiguous array of nodes */
    size_t       node_count;       /**< Total nodes allocated in this tree */
    size_t       node_capacity;    /**< Allocated node capacity */
    double      *normals_pool;     /**< Contiguous buffer of normal vectors [node_count * dimensions] */
    size_t       normals_capacity; /**< Capacity of normals buffer */
    uint32_t    *leaf_samples;     /**< Contiguous buffer of leaf sample pool indices */
    size_t       leaf_samples_count;
    size_t       leaf_samples_capacity;
    double       max_path_depth;   /**< Longest metric path accumulated in this tree */
} geif_tree_t;

/**
 * @brief The complete Geometric Extended Isolation Forest.
 */
typedef struct geif_forest {
    uint32_t      dimensions;      /**< Dimensionality of feature space D */
    uint32_t      tree_count;      /**< Number of trees T */
    geif_config_t config;          /**< Forest configuration */
    geif_tree_t  *trees;           /**< Array of trees [tree_count] */

    // Geometric Envelope (Bounding Box & Scaling)
    double       *envelope_min;    /**< Minimum coordinate observed per dimension [dimensions] */
    double       *envelope_max;    /**< Maximum coordinate observed per dimension [dimensions] */
    double       *envelope_span;   /**< Physical span (max - min) per dimension [dimensions] */
    double       *effective_span;  /**< Regularized span floor for zero-variance features [dimensions] */
    uint8_t      *dim_active;      /**< Flag: 1 if dimension has variance >= epsilon, 0 if constant */
    int32_t       scale_range_idx; /**< Dimension index with largest range for isotropic scaling (-1 if unscaled) */
    double        avg_sample_dist; /**< Average scaled sample point distance for nearest distance analysis */
    double        c_factor;        /**< c(psi) average path length for sub-sample size */

    // Global Metrics & Calibration
    double        delta_nominal;   /**< Mean generator spacing across the ensemble */
    double        H_train_max;     /**< Deepest metric depth observed in training data */
    double        H_max;           /**< Calibrated universal scale: kappa * H_train_max */
    double        min_score;       /**< Empirical minimum anomaly score in training pool */
    double        max_score;       /**< Calibrated maximum anomaly score (theoretical outer limit) */
    double        average_score;   /**< Mean anomaly score across training pool */
    double        percentage_score;/**< Percentile threshold score based on training sample distribution */
    double       *averages;        /**< Running or calculated feature averages [dimensions] */

    // Reservoir Sample Pool
    double       *sample_pool;     /**< Contiguous sample matrix [pool_capacity * dimensions] */
    double       *scaled_pool;     /**< Ephemeral cached scaled sample pool [pool_capacity * dimensions] */
    size_t        pool_count;      /**< Current number of samples in the pool */
    size_t        pool_capacity;   /**< Maximum capacity of sample pool (tree_count * samples_per_tree) */
    uint64_t      total_rows_seen; /**< Total rows streamed through the reservoir */

    char          category[128];          /**< Optional category name for multi-tenant isolation */
    uint32_t      total_input_cols;       /**< Total raw columns in tabular input */
    char          label_dims_spec[128];   /**< Label columns spec (e.g. "1") */
    char          include_dims_spec[128]; /**< Included feature columns spec (e.g. "2-10") */
    char          ignore_dims_spec[128];  /**< Ignored columns spec (e.g. "12") */
    char          category_dims_spec[128];/**< Category columns spec (e.g. "12") */
    int           decimals;               /**< Decimal precision for serialization (-d, default: 6) */
    bool          scale_score;            /**< Flag indicating if anomaly scores are scaled [0..1] via min/max */
} geif_forest_t;

typedef struct {
    char           category[128];
    geif_forest_t *forest;
    time_t         last_updated;
    uint64_t       total_rows;
    bool           seen_in_analysis;  /**< Ephemeral tracking flag for missed category detection (-M) */
} geif_category_entry_t;

typedef struct geif_cat_hash_node {
    uint32_t entry_idx;
    struct geif_cat_hash_node *next;
} geif_cat_hash_node_t;

typedef struct {
    uint32_t       dimensions;             /**< Feature dimensions per sub-forest */
    geif_config_t  config;                 /**< Default training config */
    uint32_t       total_input_cols;       /**< Total raw columns in tabular input */
    char           label_dims_spec[128];   /**< Label columns spec (e.g. "1") */
    char           include_dims_spec[128]; /**< Included feature columns spec (e.g. "2-10") */
    char           ignore_dims_spec[128];  /**< Ignored columns spec (e.g. "12") */
    char           category_dims_spec[128];/**< Category columns spec (e.g. "12") */
    char           outlier_score_spec[64]; /**< Outlier score threshold spec (e.g. "0.5", "average", "80%", "0.65s") */
    int            decimals;               /**< Decimal precision for serialization (-d, default: 6) */
    bool           scale_score;            /**< Flag indicating if anomaly scores are scaled [0..1] via min/max */

    geif_category_entry_t *entries;        /**< Dynamic array of category sub-forests */
    size_t         count;                  /**< Number of active sub-forests */
    size_t         capacity;               /**< Allocated entry capacity */

    geif_cat_hash_node_t **hash_buckets;   /**< Hash table for O(1) category lookup */
    size_t         hash_size;              /**< Number of hash buckets */
} geif_ensemble_t;

#ifdef __cplusplus
}
#endif

#endif /* GEIF_TYPES_H */
