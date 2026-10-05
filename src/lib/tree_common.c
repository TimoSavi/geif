/**
 * @file tree_common.c
 * @brief Implementation of common tree memory allocation, traversal, and calibration.
 */

#include "tree_common.h"
#include "algo.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief Initializes dimension scales, bounding envelopes, and nominal cluster distances.
 *
 * Scans the reservoir sample pool across all dimensions to compute coordinate extrema
 * ([min, max]) and active spans. Identifies the dimension with the largest span to establish
 * a normalized isotropic coordinate space for distance-invariant calculations. Also pre-computes
 * average sample spacing (delta_nominal) and harmonic correction factor c(psi).
 *
 * @param[in,out] f Pointer to the forest instance.
 */
void init_dimension_scales(geif_forest_t *f)
{
    uint32_t d = f->dimensions;
    uint32_t psi = f->config.samples_per_tree;
    if (psi == 0) psi = 256;

    if (f->pool_count > 0 && f->sample_pool) {
        for (uint32_t j = 0; j < d; j++) {
            f->envelope_min[j] = f->sample_pool[j];
            f->envelope_max[j] = f->sample_pool[j];
        }
        for (size_t i = 1; i < f->pool_count; i++) {
            const double *pt = &f->sample_pool[i * d];
            for (uint32_t j = 0; j < d; j++) {
                if (pt[j] < f->envelope_min[j]) f->envelope_min[j] = pt[j];
                if (pt[j] > f->envelope_max[j]) f->envelope_max[j] = pt[j];
            }
        }
    }

    double max_span = 0.0;
    int32_t best_dim = -1;

    for (uint32_t j = 0; j < d; j++) {
        f->envelope_span[j] = f->envelope_max[j] - f->envelope_min[j];
        if (f->envelope_span[j] >= 1e-9) {
            f->dim_active[j] = 1;
            f->effective_span[j] = f->envelope_span[j];
            if (f->envelope_span[j] > max_span) {
                max_span = f->envelope_span[j];
                best_dim = (int32_t)j;
            }
        } else {
            f->dim_active[j] = 0;
            f->effective_span[j] = 0.0;
        }
    }

    if (best_dim < 0) best_dim = 0;
    if (max_span <= 0.0) max_span = 1.0;
    f->scale_range_idx = best_dim;

    if (!f->scaled_pool && f->pool_capacity > 0) {
        f->scaled_pool = (double *)malloc(f->pool_capacity * d * sizeof(double));
    }

    if (f->scaled_pool && f->pool_count > 0) {
        double target_range = max_span;
        double scale_min = f->envelope_min[best_dim];
        for (size_t i = 0; i < f->pool_count; i++) {
            for (uint32_t j = 0; j < d; j++) {
                f->scaled_pool[i * d + j] = geif_scale_value(f->sample_pool[i * d + j],
                                                             target_range, scale_min,
                                                             f->envelope_min[j], f->envelope_max[j]);
            }
        }
    }

    uint32_t n_eff = (f->pool_count < psi) ? (uint32_t)f->pool_count : psi;
    if (n_eff < 2) n_eff = 2;
    f->avg_sample_dist = sqrt(GEIF_DIST_AVG((double)d)) * (max_span / pow((double)n_eff, 1.0 / (double)d));

    geif_init_c_cache();
    f->c_factor = geif_c((double)n_eff);
    f->delta_nominal = f->avg_sample_dist;
}

/**
 * @brief Dynamically allocates a new node slot in the tree's contiguous node pool.
 *
 * Doubles capacity as needed starting from GEIF_INITIAL_TREE_NODES. Initializes children to -1.
 *
 * @param[in,out] tree Pointer to the tree.
 * @return Non-negative node index on success, or -1 on allocation failure.
 */
int32_t allocate_node(geif_tree_t *tree)
{
    if (tree->node_count >= tree->node_capacity) {
        size_t new_cap = (tree->node_capacity == 0) ? GEIF_INITIAL_TREE_NODES : tree->node_capacity * 2;
        geif_node_t *new_nodes = (geif_node_t *)realloc(tree->nodes, new_cap * sizeof(geif_node_t));
        if (!new_nodes) return -1;
        tree->nodes = new_nodes;
        tree->node_capacity = new_cap;
    }
    int32_t idx = (int32_t)tree->node_count++;
    memset(&tree->nodes[idx], 0, sizeof(geif_node_t));
    tree->nodes[idx].left_child = -1;
    tree->nodes[idx].right_child = -1;
    return idx;
}

/**
 * @brief Appends a normal vector or hypersphere center into the tree's contiguous float pool.
 *
 * Expands the tree->normals_pool buffer dynamically as needed.
 *
 * @param[in,out] tree   Pointer to the tree.
 * @param[in]     normal Normal or center coordinate vector of length d.
 * @param[in]     d      Dimensionality of the vector.
 * @return Offset in tree->normals_pool where the normal vector is stored.
 */
uint32_t append_normal(geif_tree_t *tree, const double *normal, uint32_t d)
{
    size_t needed = (tree->node_count + 1) * d;
    if (needed > tree->normals_capacity) {
        size_t new_cap = (tree->normals_capacity == 0) ? GEIF_INITIAL_NORMALS_COUNT * d : tree->normals_capacity * 2;
        if (new_cap < needed) new_cap = needed;
        double *new_pool = (double *)realloc(tree->normals_pool, new_cap * sizeof(double));
        if (!new_pool) return 0;
        tree->normals_pool = new_pool;
        tree->normals_capacity = new_cap;
    }
    uint32_t offset = (uint32_t)((tree->node_count - 1) * d);
    memcpy(&tree->normals_pool[offset], normal, d * sizeof(double));
    return offset;
}

/**
 * @brief Appends sample pool indices into the tree's leaf sample repository.
 *
 * Used for leaf-level nearest neighbor relative distance calculations. Expands
 * tree->leaf_samples dynamically starting from GEIF_INITIAL_LEAF_SAMPLES.
 *
 * @param[in,out] tree    Pointer to the tree.
 * @param[in]     samples Array of sample indices.
 * @param[in]     count   Number of sample indices.
 * @return Starting offset in tree->leaf_samples where indices are stored.
 */
uint32_t append_leaf_samples(geif_tree_t *tree, const uint32_t *samples, size_t count)
{
    if (count == 0) return 0;
    size_t needed = tree->leaf_samples_count + count;
    if (needed > tree->leaf_samples_capacity) {
        size_t new_cap = (tree->leaf_samples_capacity == 0) ? GEIF_INITIAL_LEAF_SAMPLES : tree->leaf_samples_capacity * 2;
        if (new_cap < needed) new_cap = needed;
        uint32_t *new_arr = (uint32_t *)realloc(tree->leaf_samples, new_cap * sizeof(uint32_t));
        if (!new_arr) return 0;
        tree->leaf_samples = new_arr;
        tree->leaf_samples_capacity = new_cap;
    }
    uint32_t offset = (uint32_t)tree->leaf_samples_count;
    memcpy(&tree->leaf_samples[offset], samples, count * sizeof(uint32_t));
    tree->leaf_samples_count += count;
    return offset;
}

/**
 * @brief Restores the max-heap property by sifting down the element at index i.
 *
 * Used to maintain the K smallest squared distances in the nearest-neighbor heap
 * without heap allocations.
 *
 * @param[in,out] heap Array representing the binary max-heap.
 * @param[in]     i    Index of the element to sift down.
 * @param[in]     n    Total count of elements currently in the heap.
 */
static inline void max_heap_sift_down(double *heap, uint32_t i, uint32_t n)
{
    double val = heap[i];
    while (1) {
        uint32_t left = 2 * i + 1;
        if (left >= n) break;
        uint32_t right = left + 1;
        uint32_t largest = (right < n && heap[right] > heap[left]) ? right : left;
        if (heap[largest] <= val) break;
        heap[i] = heap[largest];
        i = largest;
    }
    heap[i] = val;
}

/**
 * @brief Computes relative Euclidean distance from query point to nearest leaf samples.
 *
 * In GEIF, relative leaf distance (rel_dist) scales effective sample density in leaf nodes
 * to prevent false positives in high-density regions and detect sparse interior voids/cavities.
 * For low dimensions (d < 5), it searches up to 2^d samples (with floor GEIF_MIN_LEAF_SAMPLE_FLOOR);
 * for d >= 5, it caps the nearest neighbor set at GEIF_MAX_LEAF_NEAREST_SAMPLES (32) to bound
 * computational complexity. Uses a stack-allocated binary max-heap to maintain the K smallest
 * squared distances without heap allocations.
 *
 * @param[in] f            Pointer to the forest.
 * @param[in] tree         Pointer to the isolation tree.
 * @param[in] node         Leaf node containing sample indices.
 * @param[in] scaled_point Query point in normalized coordinate space.
 * @return Normalized relative distance factor (>= GEIF_REL_DIST_MULTIPLIER * GEIF_MIN_REL_DIST).
 */
double geif_calc_leaf_rel_dist(const geif_forest_t *f,
                              const geif_tree_t *tree,
                              const geif_node_t *node,
                              const double *scaled_point)
{
    if (!f || !tree || !node || node->sample_count <= 0 || !tree->leaf_samples ||
        f->avg_sample_dist <= 0.0) {
        return 1.0;
    }

    uint32_t d = f->dimensions;
    uint32_t max_nearest = ((1U << d) < GEIF_MIN_LEAF_SAMPLE_FLOOR) ? GEIF_MIN_LEAF_SAMPLE_FLOOR : ((d < GEIF_MAX_LEAF_NEAREST_DIM_CAP) ? (1U << d) : GEIF_MAX_LEAF_NEAREST_SAMPLES);
    const uint32_t *leaf_s = &tree->leaf_samples[node->leaf_sample_offset];
    int32_t n_samples = node->sample_count;

    /* Fast path: if leaf has <= max_nearest samples, all valid samples belong
       to the nearest bounding set. No heap, buffer tracking, or eviction needed. */
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
        double rel_dist = sqrt(mean_d) / f->avg_sample_dist;
        rel_dist = (rel_dist < GEIF_MIN_REL_DIST) ? GEIF_MIN_REL_DIST : rel_dist;
        return GEIF_REL_DIST_MULTIPLIER * rel_dist;
    }

    /* Heap path: maintain K smallest squared distances using a stack-allocated max-heap */
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

    if (filled == 0) return 1.0;
    if (filled < max_nearest) {
        double sum_d = 0.0;
        for (uint32_t j = 0; j < filled; j++) sum_d += heap[j];
        double mean_d = sum_d / (double)filled;
        double rel_dist = sqrt(mean_d) / f->avg_sample_dist;
        rel_dist = (rel_dist < GEIF_MIN_REL_DIST) ? GEIF_MIN_REL_DIST : rel_dist;
        return GEIF_REL_DIST_MULTIPLIER * rel_dist;
    }

    /* Build max-heap */
    for (int32_t p = (int32_t)(max_nearest >> 1) - 1; p >= 0; p--) {
        max_heap_sift_down(heap, (uint32_t)p, max_nearest);
    }

    /* Process remaining samples */
    for (; i < n_samples; i++) {
        uint32_t s_idx = leaf_s[i];
        if (s_idx < f->pool_count) {
            const double *sample = (f->scaled_pool) ? &f->scaled_pool[s_idx * d]
                                                   : &f->sample_pool[s_idx * d];
            double dsq = geif_dist_sq(scaled_point, sample, d);
            if (dsq < heap[0]) {
                heap[0] = dsq;
                max_heap_sift_down(heap, 0, max_nearest);
            }
        }
    }

    double sum_d = 0.0;
    for (uint32_t j = 0; j < max_nearest; j++) sum_d += heap[j];
    double mean_d = sum_d / (double)max_nearest;
    double rel_dist = sqrt(mean_d) / f->avg_sample_dist;
    rel_dist = (rel_dist < GEIF_MIN_REL_DIST) ? GEIF_MIN_REL_DIST : rel_dist;
    return GEIF_REL_DIST_MULTIPLIER * rel_dist;
}

/**
 * @brief Traverses a single isolation tree to evaluate continuous metric path depth.
 *
 * Recursively (or iteratively) navigates split hyperplanes (dot product test vs pdotn)
 * until reaching a terminal leaf. At the leaf, applies continuous metric depth estimation
 * adjusted by relative leaf distance (rel_dist) and the harmonic function c(n).
 *
 * @param[in] f            Pointer to the forest.
 * @param[in] tree         Pointer to the isolation tree.
 * @param[in] scaled_point Query point in normalized coordinate space.
 * @return Accumulated continuous depth including leaf residual completion.
 */
double evaluate_tree(const geif_forest_t *f,
                    const geif_tree_t *tree,
                    const double *scaled_point)
{
    if (tree->node_count == 0) {
        return 0.0;
    }

    int32_t curr = 0;
    double depth = 0.0;
    uint32_t d = f->dimensions;

    while (curr >= 0 && curr < (int32_t)tree->node_count) {
        const geif_node_t *node = &tree->nodes[curr];

        // Leaf node reached
        if (node->left_child == -1 && node->right_child == -1) {
            double leaf_c = 0.0;
            if (f->avg_sample_dist > 0.0 && node->sample_count > 0 && tree->leaf_samples &&
                (node->leaf_sample_offset + (size_t)node->sample_count <= tree->leaf_samples_count)) {
                double rel_dist = geif_calc_leaf_rel_dist(f, tree, node, scaled_point);
                double adjusted_n = (double)node->sample_count / rel_dist;
                leaf_c = geif_c(adjusted_n);
            } else if (node->sample_count > 1) {
                leaf_c = geif_c((double)node->sample_count);
            }
            return depth + leaf_c;
        }

        // Internal split node
        const double *normal = &tree->normals_pool[node->normal_offset];
        double dot_val = geif_dot(scaled_point, normal, d);

        if (dot_val < node->pdotn) {
            if (node->left_child == -1) return depth;
            curr = node->left_child;
        } else {
            if (node->right_child == -1) return depth;
            curr = node->right_child;
        }
        depth += 1.0;
    }

    return depth;
}

/**
 * @brief Evaluates average metric depth H across all trees in the forest.
 *
 * Normalizes query coordinates across active bounding envelope dimensions and evaluates
 * the average path depth across all isolation trees in the ensemble. If the query point
 * lies outside the training envelope bounding box, computes the Euclidean outer space distance
 * d_out in scaled units for subsequent outer decay scoring.
 *
 * @param[in]  f         Pointer to the forest.
 * @param[in]  point     Raw unscaled query point.
 * @param[out] d_out_out Optional pointer to receive outer space Euclidean distance.
 * @return Average continuous metric depth across ensemble.
 */
double geif_tree_evaluate_metric_depth(const geif_forest_t *f,
                                      const double *point,
                                      double *d_out_out)
{
    if (!f || !point || f->tree_count == 0) {
        if (d_out_out) *d_out_out = 0.0;
        return 0.0;
    }

    uint32_t d = f->dimensions;
    double stack_buf[GEIF_STACK_BUFFER_DIMS];
    double *scaled_point = (d <= GEIF_STACK_BUFFER_DIMS) ? stack_buf : (double *)malloc(d * sizeof(double));
    if (!scaled_point) {
        if (d_out_out) *d_out_out = 0.0;
        return 0.0;
    }

    if (f->scale_range_idx >= 0 && f->envelope_span && f->envelope_min && f->envelope_max) {
        double target_range = f->envelope_span[f->scale_range_idx];
        double scale_min = f->envelope_min[f->scale_range_idx];
        for (uint32_t j = 0; j < d; j++) {
            scaled_point[j] = geif_scale_value(point[j], target_range, scale_min,
                                               f->envelope_min[j], f->envelope_max[j]);
        }
    } else {
        memcpy(scaled_point, point, d * sizeof(double));
    }

    double sum_H = 0.0;
    for (uint32_t t = 0; t < f->tree_count; t++) {
        sum_H += evaluate_tree(f, &f->trees[t], scaled_point);
    }
    double H_avg = sum_H / (double)f->tree_count;

    // Outer space distance in scaled space
    double dist_out_sq = 0.0;
    double target_range = (f->scale_range_idx >= 0 && f->envelope_span) ? f->envelope_span[f->scale_range_idx] : 1.0;
    if (f->envelope_min && f->envelope_max) {
        for (uint32_t j = 0; j < d; j++) {
            double d_low = f->envelope_min[j] - point[j];
            double d_high = point[j] - f->envelope_max[j];
            double delta = 0.0;
            if (d_low > 0.0) delta = d_low;
            else if (d_high > 0.0) delta = d_high;

            if (delta > 0.0 && f->envelope_span && f->envelope_span[j] > 1e-12) {
                double norm = delta * (target_range / f->envelope_span[j]);
                dist_out_sq += norm * norm;
            }
        }
    }

    double d_out_scaled = sqrt(dist_out_sq);
    if (scaled_point != stack_buf) free(scaled_point);
    if (d_out_out) *d_out_out = d_out_scaled;
    return H_avg;
}

/**
 * @brief Computes calibrated anomaly score, metric depth, and outer space distance for a query point.
 *
 * Applies the canonical Isolation Forest exponential mapping s = 2^(-H / c(psi)), augmented by:
 * 1. Outer space smooth asymptotic decay: s' = 1 - (1 - s) * exp(-GEIF_OUTER_DECAY_RATE * d_norm).
 * 2. Zero Kelvin scale floor calibration: maps s_min (deepest observed training path) to 0.0.
 *
 * @param[in]  f               Pointer to the forest.
 * @param[in]  point           Raw unscaled query coordinate vector of length d.
 * @param[out] score_out       Pointer to receive anomaly score in [0..1).
 * @param[out] metric_depth_out Optional pointer to receive evaluated continuous metric depth H.
 * @param[out] d_out_out       Optional pointer to receive scaled Euclidean distance beyond envelope.
 * @return GEIF_OK on success, or GEIF_ERR_INVALID_ARG if invalid pointers are supplied.
 */
geif_status_t geif_tree_score_point(const geif_forest_t *f,
                                   const double *point,
                                   double *score_out,
                                   double *metric_depth_out,
                                   double *d_out_out)
{
    if (!f || !point || !score_out) {
        return GEIF_ERR_INVALID_ARG;
    }

    double d_out = 0.0;
    double H_final = geif_tree_evaluate_metric_depth(f, point, &d_out);

    if (metric_depth_out) *metric_depth_out = H_final;
    if (d_out_out) *d_out_out = d_out;

    double psi = (f->config.samples_per_tree > 0) ? (double)f->config.samples_per_tree : 256.0;
    double c_psi = (f->c_factor > 0.0) ? f->c_factor : geif_c(psi);
    if (c_psi <= 0.0) c_psi = 1.0;

    // Standard Isolation Forest score s = 1.0 / 2^(H / c)
    double score = 1.0 / pow(2.0, H_final / c_psi);

    // Outer space exponential attenuation: asymptotic convergence towards 1.0 without boundary wall
    if (d_out > 0.0) {
        double target_range = (f->scale_range_idx >= 0 && f->envelope_span) ? f->envelope_span[f->scale_range_idx] : 1.0;
        double d_norm = (target_range > 1e-12) ? (d_out / target_range) : d_out;
        score = 1.0 - (1.0 - score) * exp(-GEIF_OUTER_DECAY_RATE * d_norm);
    }
    if (score < 0.0) score = 0.0;
    if (score >= 1.0) score = 1.0 - 1e-6;

    if (f->scale_score) {
        double max_s = (f->max_score > 0.0) ? f->max_score : 1.0;
        double min_s = f->min_score;
        double scaled_score = score;
        if (max_s > min_s) {
            scaled_score = (score - min_s) / (max_s - min_s);
            if (scaled_score < 0.0) scaled_score = 0.0;
            if (scaled_score >= 1.0) scaled_score = 1.0 - 1e-6;
        }
        *score_out = scaled_score;
    } else {
        *score_out = score;
    }
    return GEIF_OK;
}

/**
 * @brief Recursively traverses an isolation tree to determine the maximum path depth.
 *
 * Explores all branches to the deepest terminal leaf node to find the theoretical
 * maximum height H_max achievable by any point falling into this tree.
 *
 * @param[in]     f        Pointer to the forest.
 * @param[in]     t        Pointer to the tree.
 * @param[in]     node_idx Current node index.
 * @param[in]     depth    Accumulated split depth from root.
 * @param[in,out] max_h    Pointer tracking the maximum depth discovered so far.
 */
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
            double rel_dist = GEIF_REL_DIST_MULTIPLIER * GEIF_MIN_REL_DIST;
            double adjusted_n = (double)node->sample_count / rel_dist;
            leaf_c = geif_c(adjusted_n);
        } else if (node->sample_count > 1) {
            leaf_c = geif_c((double)node->sample_count);
        }
        double h = depth + leaf_c;
        if (h > *max_h) {
            *max_h = h;
        }
        return;
    }

    if (node->left_child != -1) {
        geif_tree_find_max_height(f, t, node->left_child, depth + 1.0, max_h);
    } else {
        if (depth > *max_h) *max_h = depth;
    }
    if (node->right_child != -1) {
        geif_tree_find_max_height(f, t, node->right_child, depth + 1.0, max_h);
    } else {
        if (depth > *max_h) *max_h = depth;
    }
}

/**
 * @brief Calibrates "Zero Kelvin" baseline scale floor and ensemble average inlier score.
 *
 * Computes average maximum height H_train_max across all trained trees to establish
 * min_score = 2^(-H_train_max / c(psi)). Evaluates all training points in the sample pool
 * to compute the average baseline score and empirical dimension means.
 *
 * @param[in,out] f Pointer to the forest instance.
 */
void geif_tree_calibrate(geif_forest_t *f)
{
    f->min_score = 0.0;
    f->max_score = 1.0;

    double sum_max_h = 0.0;
    for (uint32_t t = 0; t < f->tree_count; t++) {
        double max_h = 0.0;
        geif_tree_find_max_height(f, &f->trees[t], 0, 0.0, &max_h);
        sum_max_h += max_h;
    }

    double avg_max_h = (f->tree_count > 0) ? (sum_max_h / (double)f->tree_count) : 0.0;
    if (avg_max_h < 1.0) avg_max_h = 1.0;

    double psi_d = (f->config.samples_per_tree > 0) ? (double)f->config.samples_per_tree : 256.0;
    double c_psi = (f->c_factor > 0.0) ? f->c_factor : geif_c(psi_d);
    if (c_psi <= 0.0) c_psi = 1.0;

    f->H_train_max = avg_max_h;
    f->H_max = avg_max_h;
    f->min_score = pow(2.0, -avg_max_h / c_psi);
    if (f->min_score < 0.0) f->min_score = 0.0;
    if (f->min_score > 1.0) f->min_score = 1.0;
    f->max_score = 1.0;

    const geif_algo_ops_t *ops = geif_algo_get_ops(f->config.algo);
    double sum_score = 0.0;
    for (size_t i = 0; i < f->pool_count; i++) {
        const double *pt = &f->sample_pool[i * f->dimensions];
        double s = 0.0;
        if (ops && ops->score) {
            ops->score(f, pt, &s, NULL, NULL);
        } else {
            geif_tree_score_point(f, pt, &s, NULL, NULL);
        }
        sum_score += s;
    }
    f->average_score = (f->pool_count > 0) ? (sum_score / (double)f->pool_count) : 0.5;

    if (f->averages && f->pool_count > 0 && f->sample_pool) {
        for (uint32_t j = 0; j < f->dimensions; j++) f->averages[j] = 0.0;
        for (size_t i = 0; i < f->pool_count; i++) {
            const double *pt = &f->sample_pool[i * f->dimensions];
            for (uint32_t j = 0; j < f->dimensions; j++) {
                f->averages[j] += pt[j];
            }
        }
        for (uint32_t j = 0; j < f->dimensions; j++) {
            f->averages[j] /= (double)f->pool_count;
        }
    }
}
