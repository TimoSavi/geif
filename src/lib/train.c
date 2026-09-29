/**
 * @file train.c
 * @brief Voronoi bisector hyperplane tree training and ensemble calibration.
 */

#include "geif/geif.h"
#include "geometry.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GEIF_MAX_TRIALS 5

static void init_dimension_scales(geif_forest_t *f)
{
    uint32_t d = f->dimensions;
    uint32_t psi = f->config.samples_per_tree;
    if (psi == 0) psi = 256;

    // Recalibrate bounding envelope from current sample pool if available
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

    // Allocate scaled_pool if needed
    if (!f->scaled_pool && f->pool_capacity > 0) {
        f->scaled_pool = (double *)malloc(f->pool_capacity * d * sizeof(double));
    }

    // Populate scaled_pool
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

    // Compute average scaled sample distance
    uint32_t n_eff = (f->pool_count < psi) ? (uint32_t)f->pool_count : psi;
    if (n_eff < 2) n_eff = 2;
    f->avg_sample_dist = sqrt(GEIF_DIST_AVG((double)d)) * (max_span / pow((double)n_eff, 1.0 / (double)d));

    geif_init_c_cache();
    f->c_factor = geif_c((double)n_eff);
    f->delta_nominal = f->avg_sample_dist;
}

static int32_t allocate_node(geif_tree_t *tree)
{
    if (tree->node_count >= tree->node_capacity) {
        size_t new_cap = (tree->node_capacity == 0) ? 64 : tree->node_capacity * 2;
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

static uint32_t append_normal(geif_tree_t *tree, const double *normal, uint32_t d)
{
    size_t needed = (tree->node_count + 1) * d;
    if (needed > tree->normals_capacity) {
        size_t new_cap = (tree->normals_capacity == 0) ? 64 * d : tree->normals_capacity * 2;
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

static uint32_t append_leaf_samples(geif_tree_t *tree, const uint32_t *samples, size_t count)
{
    if (count == 0) return 0;
    size_t needed = tree->leaf_samples_count + count;
    if (needed > tree->leaf_samples_capacity) {
        size_t new_cap = (tree->leaf_samples_capacity == 0) ? 128 : tree->leaf_samples_capacity * 2;
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

static int32_t build_tree_node(geif_forest_t *f,
                              geif_tree_t *tree,
                              uint32_t *indices,
                              size_t count,
                              uint32_t depth,
                              uint32_t max_depth)
{
    // Base condition: leaf node reached (NODE_MIN_SAMPLE = 3 or depth limit reached)
    if (count < 3 || depth >= max_depth) {
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
    double *p = (double *)malloc(d * sizeof(double));
    double *n = (double *)malloc(d * sizeof(double));
    uint32_t *left_indices = (uint32_t *)malloc(count * sizeof(uint32_t));
    uint32_t *right_indices = (uint32_t *)malloc(count * sizeof(uint32_t));

    if (!p || !n || !left_indices || !right_indices) {
        if (p) free(p);
        if (n) free(n);
        if (left_indices) free(left_indices);
        if (right_indices) free(right_indices);
        node = &tree->nodes[node_idx];
        node->leaf_sample_offset = append_leaf_samples(tree, indices, count);
        return node_idx;
    }

    // Average consecutive distance in scaled space
    double sum_consec = 0.0;
    for (size_t i = 1; i < count; i++) {
        const double *s1 = &f->scaled_pool[indices[i - 1] * d];
        const double *s2 = &f->scaled_pool[indices[i] * d];
        sum_consec += sqrt(geif_dist_sq(s1, s2, d));
    }
    double avg_consec_dist = (count > 1) ? (sum_consec / (double)(count - 1)) : 0.0;

    double height_ratio = (max_depth > 0) ? (1.0 - ((double)depth / (double)max_depth)) : 1.0;
    if (height_ratio < 0.0) height_ratio = 0.0;

    size_t left_count = 0;
    size_t right_count = 0;
    double best_pdotn = 0.0;
    bool split_found = false;

    for (int trial = 0; trial < 10; trial++) {
        size_t r1 = (size_t)rand() % count;
        size_t r2 = (size_t)rand() % count;
        if (r1 == r2) {
            r2 = (r1 + 1) % count;
        }

        uint32_t idx1 = indices[r1];
        uint32_t idx2 = indices[r2];
        const double *x1 = &f->scaled_pool[idx1 * d];
        const double *x2 = &f->scaled_pool[idx2 * d];

        double pair_dist = sqrt(geif_dist_sq(x1, x2, d));
        double pair_factor = 0.0;
        if (avg_consec_dist > 0.0) {
            pair_factor = 1.0 - (pair_dist / (2.0 * avg_consec_dist));
            if (pair_factor < 0.0) pair_factor = 0.0;
        }

        double margin = 2.5 * (height_ratio * height_ratio) * pair_factor;
        double u = -margin + ((double)rand() / (double)RAND_MAX) * (1.0 + 2.0 * margin);

        for (uint32_t j = 0; j < d; j++) {
            p[j] = x1[j] + u * (x2[j] - x1[j]);
            n[j] = geif_gaussrand();
        }

        double pdotn = geif_dot(p, n, d);

        left_count = 0;
        right_count = 0;
        for (size_t i = 0; i < count; i++) {
            const double *x = &f->scaled_pool[indices[i] * d];
            if (geif_dot(x, n, d) < pdotn) {
                left_indices[left_count++] = indices[i];
            } else {
                right_indices[right_count++] = indices[i];
            }
        }

        if (left_count > 0 && right_count > 0) {
            best_pdotn = pdotn;
            split_found = true;
            break;
        }
    }

    if (!split_found) {
        free(p);
        free(n);
        free(left_indices);
        free(right_indices);
        node = &tree->nodes[node_idx];
        node->left_child = -1;
        node->right_child = -1;
        node->leaf_sample_offset = append_leaf_samples(tree, indices, count);
        return node_idx;
    }

    uint32_t offset = append_normal(tree, n, d);
    free(p);
    free(n);

    node = &tree->nodes[node_idx];
    node->normal_offset = offset;
    node->pdotn = best_pdotn;
    node->step_weight = 1.0;

    int32_t left_child = -1;
    int32_t right_child = -1;

    if (left_count > 1) {
        left_child = build_tree_node(f, tree, left_indices, left_count, depth + 1, max_depth);
    }
    if (right_count > 1) {
        right_child = build_tree_node(f, tree, right_indices, right_count, depth + 1, max_depth);
    }

    free(left_indices);
    free(right_indices);

    node = &tree->nodes[node_idx];
    node->left_child = left_child;
    node->right_child = right_child;

    if (left_child == -1 && right_child == -1) {
        node->leaf_sample_offset = append_leaf_samples(tree, indices, count);
    }

    return node_idx;
}

geif_status_t geif_forest_train(geif_forest_t *f)
{
    if (!f || f->pool_count == 0) {
        return GEIF_ERR_EMPTY_DATASET;
    }

    init_dimension_scales(f);

    uint32_t psi = f->config.samples_per_tree;
    if (psi > f->pool_count) psi = (uint32_t)f->pool_count;
    if (psi < 2) psi = (uint32_t)f->pool_count;

    uint32_t *subsample = (uint32_t *)malloc(psi * sizeof(uint32_t));
    if (!subsample) return GEIF_ERR_OUT_OF_MEMORY;

    size_t current_pool_idx = 0;

    // Train each tree in the ensemble
    for (uint32_t t = 0; t < f->tree_count; t++) {
        geif_tree_t *tree = &f->trees[t];

        // Clean previous tree nodes if retraining
        if (tree->nodes) { free(tree->nodes); tree->nodes = NULL; }
        if (tree->normals_pool) { free(tree->normals_pool); tree->normals_pool = NULL; }
        if (tree->leaf_samples) { free(tree->leaf_samples); tree->leaf_samples = NULL; }
        tree->node_count = 0;
        tree->node_capacity = 0;
        tree->normals_capacity = 0;
        tree->leaf_samples_count = 0;
        tree->leaf_samples_capacity = 0;

        // Draw non-replacement circular window for this tree
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
        build_tree_node(f, tree, subsample, psi, 0, tree_max_depth);
    }

    free(subsample);

    // Calibrate min_score and average_score
    f->min_score = 0.0;
    f->max_score = 1.0;

    double min_raw = 1.0;
    double max_H = 0.0;
    double sum_H = 0.0;

    for (size_t i = 0; i < f->pool_count; i++) {
        const double *pt = &f->sample_pool[i * f->dimensions];
        double d_out = 0.0;
        double H = geif_forest_evaluate_metric_depth(f, pt, &d_out);
        sum_H += H;
        if (H > max_H) max_H = H;

        double psi_d = (f->config.samples_per_tree > 0) ? (double)f->config.samples_per_tree : 256.0;
        double c_psi = (f->c_factor > 0.0) ? f->c_factor : geif_c(psi_d);
        if (c_psi <= 0.0) c_psi = 1.0;
        double s_raw = pow(2.0, -H / c_psi);
        if (s_raw < min_raw) {
            min_raw = s_raw;
        }
    }

    f->min_score = min_raw;
    f->max_score = 1.0;
    f->H_train_max = (max_H < 1.0) ? 1.0 : max_H;
    f->H_max = f->config.kappa * f->H_train_max;

    // Compute average score across training pool
    double sum_score = 0.0;
    for (size_t i = 0; i < f->pool_count; i++) {
        const double *pt = &f->sample_pool[i * f->dimensions];
        double s = 0.0;
        geif_forest_score(f, pt, &s);
        sum_score += s;
    }
    f->average_score = (f->pool_count > 0) ? (sum_score / (double)f->pool_count) : 0.5;

    // Compute feature dimension averages
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

    return GEIF_OK;
}
