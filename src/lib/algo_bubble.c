/**
 * @file algo_bubble.c
 * @brief Bubble Algorithm: Hyperspherical cavity carving trees with empty void leaves.
 */

#include "algo.h"
#include "tree_common.h"
#include "geometry.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief Swaps two double-precision floating point values in-place.
 */
static inline void swap_doubles(double *a, double *b)
{
    double tmp = *a;
    *a = *b;
    *b = tmp;
}

/**
 * @brief Finds the k-th smallest element using Hoare's Quickselect in O(N) average time.
 *
 * Used to compute the median squared radius directly on squared Euclidean distances,
 * avoiding O(N log N) sorting and eliminating sqrt() operations entirely during training.
 *
 * @param arr Array of double values (partially reordered in-place).
 * @param n   Total number of elements in arr.
 * @param k   0-based index of the desired order statistic (e.g. n / 2 for median).
 * @return The value of the k-th smallest element.
 */
static double quickselect_median(double *arr, size_t n, size_t k)
{
    size_t left = 0, right = n - 1;
    while (left < right) {
        size_t mid = left + (right - left) / 2;
        double pivot = arr[mid];
        swap_doubles(&arr[mid], &arr[right]);
        size_t i = left;
        for (size_t j = left; j < right; j++) {
            if (arr[j] <= pivot) {
                swap_doubles(&arr[i], &arr[j]);
                i++;
            }
        }
        swap_doubles(&arr[i], &arr[right]);
        if (i == k) return arr[i];
        if (i < k) left = i + 1;
        else right = i - 1;
    }
    return arr[left];
}

/**
 * @brief Recursively constructs a hyperspherical Bubble tree node.
 *
 * Selects an exemplar point as center, computes squared distances for all samples,
 * finds the median squared radius via O(N) quickselect, and partitions samples
 * in-place into interior (<= R^2) and exterior (> R^2) child branches.
 * Non-convex internal cavities terminate into empty void leaves.
 *
 * @param f             Pointer to the forest.
 * @param tree          Pointer to the tree being built.
 * @param indices       Subarray of sample pool indices assigned to this node.
 * @param count         Number of samples in indices.
 * @param depth         Current tree depth from root.
 * @param max_depth     Maximum allowable tree depth.
 * @param dists_scratch Reusable scratchpad for squared distances.
 * @return Allocated node index, or -1 on allocation failure.
 */
static int32_t build_bubble_node(geif_forest_t *f,
                                geif_tree_t *tree,
                                uint32_t *indices,
                                size_t count,
                                uint32_t depth,
                                uint32_t max_depth,
                                double *dists_scratch)
{
    if (count < NODE_MIN_SAMPLE || depth >= max_depth) {
        if (depth == 0) {
            int32_t node_idx = allocate_node(tree);
            if (node_idx < 0) return -1;
            geif_node_t *node = &tree->nodes[node_idx];
            node->sample_count = (int32_t)count;
            node->leaf_point_idx = (count > 0) ? indices[0] : 0;
            node->left_child = -1;
            node->right_child = -1;
            node->leaf_sample_offset = append_leaf_samples(tree, indices, count);
            return node_idx;
        }
        return -1;
    }

    int32_t node_idx = allocate_node(tree);
    if (node_idx < 0) return -1;

    geif_node_t *node = &tree->nodes[node_idx];
    node->sample_count = (int32_t)count;
    node->leaf_point_idx = (count > 0) ? indices[0] : 0;
    node->left_child = -1;
    node->right_child = -1;
    node->leaf_sample_offset = 0;

    uint32_t d = f->dimensions;
    double stack_center[64];
    double *center = (d <= 64) ? stack_center : (double *)malloc(d * sizeof(double));
    if (!center) {
        node = &tree->nodes[node_idx];
        node->leaf_sample_offset = append_leaf_samples(tree, indices, count);
        return node_idx;
    }

    // Pick random exemplar as center
    size_t c_idx = (size_t)rand() % count;
    const double *c_pt = &f->scaled_pool[indices[c_idx] * d];
    memcpy(center, c_pt, d * sizeof(double));

    // Compute squared Euclidean distances directly (no sqrt required)
    for (size_t i = 0; i < count; i++) {
        const double *pt = &f->scaled_pool[indices[i] * d];
        dists_scratch[i] = geif_dist_sq(center, pt, d);
    }

    // O(N) Quickselect median squared radius
    double radius_sq = quickselect_median(dists_scratch, count, count / 2);
    if (radius_sq < 1e-12) radius_sq = 1e-6;

    // In-place partition of indices into left (<= R^2) and right (> R^2)
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

    if (left_count == 0 || right_count == 0) {
        if (d > 64) free(center);
        node = &tree->nodes[node_idx];
        node->leaf_sample_offset = append_leaf_samples(tree, indices, count);
        return node_idx;
    }

    node = &tree->nodes[node_idx];
    node->normal_offset = append_normal(tree, center, d);
    node->pdotn = radius_sq; // Store R^2 in pdotn
    node->step_weight = 1.0;
    node->leaf_sample_offset = 0;

    if (d > 64) free(center);

    int32_t left_child = -1;
    int32_t right_child = -1;

    if (left_count > 1) {
        left_child = build_bubble_node(f, tree, indices, left_count, depth + 1, max_depth, dists_scratch);
    }
    if (right_count > 1) {
        right_child = build_bubble_node(f, tree, &indices[left_count], right_count, depth + 1, max_depth, dists_scratch);
    }

    node = &tree->nodes[node_idx];
    node->left_child = left_child;
    node->right_child = right_child;

    if (left_child == -1 && right_child == -1) {
        node->leaf_sample_offset = append_leaf_samples(tree, indices, count);
    }

    return node_idx;
}

/**
 * @brief Traverses a Bubble tree to evaluate continuous metric depth for a point.
 *
 * Traverses inside the hypersphere if dist_sq(x, center) <= R^2, otherwise
 * branches outside. Applies leaf cavity relative distance damping at leaves.
 *
 * @param f            Pointer to the forest.
 * @param tree         Pointer to the isolation tree.
 * @param scaled_point Query point in normalized coordinate space.
 * @return Accumulated metric depth H.
 */
static double evaluate_bubble_tree(const geif_forest_t *f,
                                  const geif_tree_t *tree,
                                  const double *scaled_point)
{
    if (tree->node_count == 0) return 0.0;

    int32_t curr = 0;
    double depth = 0.0;
    uint32_t d = f->dimensions;

    while (curr >= 0 && curr < (int32_t)tree->node_count) {
        const geif_node_t *node = &tree->nodes[curr];

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

        const double *center = &tree->normals_pool[node->normal_offset];
        double dsq = geif_dist_sq(scaled_point, center, d);

        if (dsq <= node->pdotn) {
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
 * @brief Trains an ensemble of hyperspherical Bubble isolation trees.
 *
 * Normalizes dimension scales, draws subsamples per tree, and builds cavity-carving
 * trees with in-place median partitioning and Zero Kelvin universal calibration.
 *
 * @param f Pointer to the forest instance.
 * @return GEIF_OK on success, or error status.
 */
static geif_status_t geif_bubble_train(geif_forest_t *f)
{
    if (!f || f->pool_count == 0) return GEIF_ERR_EMPTY_DATASET;

    init_dimension_scales(f);

    uint32_t psi = f->config.samples_per_tree;
    if (psi > f->pool_count) psi = (uint32_t)f->pool_count;
    if (psi < 2) psi = (uint32_t)f->pool_count;

    uint32_t *subsample = (uint32_t *)malloc(psi * sizeof(uint32_t));
    double *dists_scratch = (double *)malloc(psi * sizeof(double));
    if (!subsample || !dists_scratch) {
        if (subsample) free(subsample);
        if (dists_scratch) free(dists_scratch);
        return GEIF_ERR_OUT_OF_MEMORY;
    }

    size_t current_pool_idx = 0;

    for (uint32_t t = 0; t < f->tree_count; t++) {
        geif_tree_t *tree = &f->trees[t];

        if (tree->nodes) { free(tree->nodes); tree->nodes = NULL; }
        if (tree->normals_pool) { free(tree->normals_pool); tree->normals_pool = NULL; }
        if (tree->leaf_samples) { free(tree->leaf_samples); tree->leaf_samples = NULL; }
        tree->node_count = 0;
        tree->node_capacity = 0;
        tree->normals_capacity = 0;
        tree->leaf_samples_count = 0;
        tree->leaf_samples_capacity = 0;

        if (f->pool_count <= psi) {
            for (uint32_t i = 0; i < psi; i++) {
                subsample[i] = (uint32_t)i;
            }
        } else {
            for (uint32_t i = 0; i < psi; i++) {
                subsample[i] = (uint32_t)((current_pool_idx + i) % f->pool_count);
            }
            current_pool_idx = (current_pool_idx + psi) % f->pool_count;
        }

        uint32_t tree_max_depth = (f->config.max_depth > 0) ? f->config.max_depth
                                                            : (uint32_t)(ceil(log2(psi)) + 1);
        build_bubble_node(f, tree, subsample, psi, 0, tree_max_depth, dists_scratch);
    }

    free(subsample);
    free(dists_scratch);

    geif_tree_calibrate(f);
    return GEIF_OK;
}

/**
 * @brief Scores a sample observation against the trained Bubble forest.
 *
 * Traverses all trees in the ensemble, accumulates average metric depth H_avg,
 * computes normalized outer space distance d_out to the bounding box, and applies
 * asymptotic exponential stadium attenuation: score = 1 - (1 - score)*exp(-0.10*d_out).
 *
 * @param[in]  f               Pointer to the forest.
 * @param[in]  point           Raw unscaled observation vector.
 * @param[out] score_out       Pointer to receive calibrated anomaly score.
 * @param[out] metric_depth_out Optional pointer to receive average depth H_avg.
 * @param[out] d_out_out       Optional pointer to receive outer distance d_out.
 * @return GEIF_OK on success, or error status.
 */
static geif_status_t geif_bubble_score(const geif_forest_t *f,
                                      const double *point,
                                      double *score_out,
                                      double *metric_depth_out,
                                      double *d_out_out)
{
    if (!f || !point || !score_out) return GEIF_ERR_INVALID_ARG;

    uint32_t d = f->dimensions;
    double stack_buf[64];
    double *scaled_point = (d <= 64) ? stack_buf : (double *)malloc(d * sizeof(double));
    if (!scaled_point) return GEIF_ERR_OUT_OF_MEMORY;

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
        sum_H += evaluate_bubble_tree(f, &f->trees[t], scaled_point);
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

    if (metric_depth_out) *metric_depth_out = H_avg;
    if (d_out_out) *d_out_out = d_out_scaled;

    double psi = (f->config.samples_per_tree > 0) ? (double)f->config.samples_per_tree : 256.0;
    double c_psi = (f->c_factor > 0.0) ? f->c_factor : geif_c(psi);
    if (c_psi <= 0.0) c_psi = 1.0;

    double score = 1.0 / pow(2.0, H_avg / c_psi);

    if (d_out_scaled > 0.0) {
        double d_norm = (target_range > 1e-12) ? (d_out_scaled / target_range) : d_out_scaled;
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

const geif_algo_ops_t geif_algo_ops_bubble = {
    .type = GEIF_ALGO_BUBBLE,
    .name = "bubble",
    .description = "Hyperspherical bubble cavity carving trees with empty void leaves",
    .train = geif_bubble_train,
    .score = geif_bubble_score,
    .destroy = NULL
};
