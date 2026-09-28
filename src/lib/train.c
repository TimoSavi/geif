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
    double sum_span = 0.0;
    uint32_t active_count = 0;

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

    for (uint32_t j = 0; j < d; j++) {
        f->envelope_span[j] = f->envelope_max[j] - f->envelope_min[j];
        if (f->envelope_span[j] >= 1e-9) {
            f->dim_active[j] = 1;
            f->effective_span[j] = f->envelope_span[j];
            sum_span += f->envelope_span[j];
            active_count++;
        } else {
            f->dim_active[j] = 0;
            f->effective_span[j] = 0.0;
        }
    }

    double mean_active_span = (active_count > 0) ? (sum_span / active_count) : 1.0;

    // Regularize inactive dimensions with adaptive floor
    for (uint32_t j = 0; j < d; j++) {
        if (!f->dim_active[j]) {
            double mean_val = 0.5 * (f->envelope_min[j] + f->envelope_max[j]);
            double floor_val = 0.01 * fabs(mean_val) + 1e-4 * mean_active_span;
            if (floor_val < 1e-6) floor_val = 1e-6;
            f->effective_span[j] = floor_val;
        }
    }

    // Default nominal spacing estimation based on active dimensions
    f->delta_nominal = (active_count > 0) ? (0.25 * sqrt((double)active_count)) : 1.0;
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

static int32_t build_tree_node(geif_forest_t *f,
                              geif_tree_t *tree,
                              uint32_t *indices,
                              size_t count,
                              uint32_t depth,
                              uint32_t max_depth)
{
    int32_t node_idx = allocate_node(tree);
    if (node_idx < 0) return -1;

    geif_node_t *node = &tree->nodes[node_idx];
    node->sample_count = (int32_t)count;
    node->leaf_point_idx = (count > 0) ? indices[0] : 0;

    // Base condition: leaf node
    if (count <= 1 || depth >= max_depth) {
        return node_idx;
    }

    uint32_t d = f->dimensions;
    double *candidate_normal = (double *)malloc(d * sizeof(double));
    if (!candidate_normal) return node_idx;

    uint32_t idx_A = 0, idx_B = 0;
    double pdotn = 0.0, delta = 0.0;
    bool bisector_valid = false;

    // Trial loop to pick valid distinct pair A != B
    for (int trial = 0; trial < GEIF_MAX_TRIALS; trial++) {
        size_t rA = (size_t)rand() % count;
        size_t rB = (size_t)rand() % count;
        if (rA == rB) continue;

        idx_A = indices[rA];
        idx_B = indices[rB];
        const double *A = &f->sample_pool[idx_A * d];
        const double *B = &f->sample_pool[idx_B * d];

        if (geif_compute_bisector(A, B, f->effective_span, f->dim_active, d,
                                 candidate_normal, &pdotn, &delta)) {
            bisector_valid = true;
            break;
        }
    }

    // Fallback: linear search for distinct sample if random draws hit duplicates
    if (!bisector_valid) {
        idx_A = indices[0];
        const double *A = &f->sample_pool[idx_A * d];
        for (size_t i = 1; i < count; i++) {
            idx_B = indices[i];
            const double *B = &f->sample_pool[idx_B * d];
            if (geif_compute_bisector(A, B, f->effective_span, f->dim_active, d,
                                     candidate_normal, &pdotn, &delta)) {
                bisector_valid = true;
                break;
            }
        }
    }

    // Inseparable leaf node: all samples in subset are identical!
    if (!bisector_valid) {
        free(candidate_normal);
        return node_idx;
    }

    // Partition samples into left and right subsets
    uint32_t *left_indices = (uint32_t *)malloc(count * sizeof(uint32_t));
    uint32_t *right_indices = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!left_indices || !right_indices) {
        free(candidate_normal);
        if (left_indices) free(left_indices);
        if (right_indices) free(right_indices);
        return node_idx;
    }

    size_t left_count = 0;
    size_t right_count = 0;

    for (size_t i = 0; i < count; i++) {
        uint32_t s_idx = indices[i];
        const double *x = &f->sample_pool[s_idx * d];
        double dot_val = geif_dot(x, candidate_normal, d);

        if (dot_val < pdotn) {
            left_indices[left_count++] = s_idx;
        } else {
            right_indices[right_count++] = s_idx;
        }
    }

    // Check for degenerate partition (peeled off 0 points)
    if (left_count == 0 || right_count == 0) {
        free(candidate_normal);
        free(left_indices);
        free(right_indices);
        return node_idx;
    }

    // Valid internal split node: record parameters
    uint32_t offset = append_normal(tree, candidate_normal, d);
    free(candidate_normal);

    // Re-acquire node pointer in case realloc moved the buffer
    node = &tree->nodes[node_idx];
    node->normal_offset = offset;
    node->pdotn = pdotn;
    node->delta_AB = delta;

    // Continuous metric depth increment Delta H
    double ratio = f->delta_nominal / (delta > 1e-12 ? delta : 1e-12);
    node->step_weight = 1.0 + f->config.alpha * log(1.0 + ratio);

    // Recursively build child subtrees
    int32_t left_child = build_tree_node(f, tree, left_indices, left_count, depth + 1, max_depth);
    int32_t right_child = build_tree_node(f, tree, right_indices, right_count, depth + 1, max_depth);

    free(left_indices);
    free(right_indices);

    // Re-acquire node pointer after recursion
    tree->nodes[node_idx].left_child = left_child;
    tree->nodes[node_idx].right_child = right_child;

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

    // Train each tree in the ensemble
    for (uint32_t t = 0; t < f->tree_count; t++) {
        geif_tree_t *tree = &f->trees[t];

        // Clean previous tree nodes if retraining
        if (tree->nodes) { free(tree->nodes); tree->nodes = NULL; }
        if (tree->normals_pool) { free(tree->normals_pool); tree->normals_pool = NULL; }
        tree->node_count = 0;
        tree->node_capacity = 0;
        tree->normals_capacity = 0;

        // Draw random reservoir sub-sample
        for (uint32_t i = 0; i < psi; i++) {
            subsample[i] = (uint32_t)((size_t)rand() % f->pool_count);
        }

        build_tree_node(f, tree, subsample, psi, 0, f->config.max_depth);
    }

    free(subsample);

    // Calibrate universal scale H_train_max, H_max, and average_score across the training pool
    double max_H = 0.0;
    double sum_H = 0.0;
    for (size_t i = 0; i < f->pool_count; i++) {
        const double *pt = &f->sample_pool[i * f->dimensions];
        double H_metric = geif_forest_evaluate_metric_depth(f, pt, NULL);
        sum_H += H_metric;
        if (H_metric > max_H) {
            max_H = H_metric;
        }
    }

    if (max_H < 1.0) max_H = 1.0;
    f->H_train_max = max_H;
    f->H_max = f->config.kappa * f->H_train_max;
    if (f->pool_count > 0 && f->H_max > 0.0) {
        double mean_H = sum_H / (double)f->pool_count;
        double avg_s = 1.0 - (mean_H / f->H_max);
        if (avg_s < 0.0) avg_s = 0.0;
        if (avg_s > 1.0) avg_s = 1.0;
        f->average_score = avg_s;
    } else {
        f->average_score = 0.5;
    }

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
