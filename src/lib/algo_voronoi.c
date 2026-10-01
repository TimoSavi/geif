/**
 * @file algo_voronoi.c
 * @brief Voronoi Algorithm: Pure perpendicular bisector splits between sample pairs.
 */

#include "algo.h"
#include "tree_common.h"
#include "geometry.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int32_t build_voronoi_node(geif_forest_t *f,
                                 geif_tree_t *tree,
                                 uint32_t *indices,
                                 size_t count,
                                 uint32_t depth,
                                 uint32_t max_depth)
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

        // Pure Voronoi perpendicular bisector:
        // n = (x2 - x1) / ||x2 - x1||
        // p = (x1 + x2) / 2
        double norm_sq = 0.0;
        for (uint32_t j = 0; j < d; j++) {
            n[j] = x2[j] - x1[j];
            norm_sq += n[j] * n[j];
            p[j] = 0.5 * (x1[j] + x2[j]);
        }

        double norm_len = sqrt(norm_sq);
        if (norm_len < 1e-12) continue;
        for (uint32_t j = 0; j < d; j++) {
            n[j] /= norm_len;
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
            split_found = true;
            best_pdotn = pdotn;
            break;
        }
    }

    if (!split_found) {
        free(p);
        free(n);
        free(left_indices);
        free(right_indices);
        node = &tree->nodes[node_idx];
        node->leaf_sample_offset = append_leaf_samples(tree, indices, count);
        return node_idx;
    }

    node = &tree->nodes[node_idx];
    node->normal_offset = append_normal(tree, n, d);
    node->pdotn = best_pdotn;
    node->step_weight = 1.0;
    node->leaf_sample_offset = 0;

    free(p);
    free(n);

    int32_t left_child = -1;
    int32_t right_child = -1;

    if (left_count > 1) {
        left_child = build_voronoi_node(f, tree, left_indices, left_count, depth + 1, max_depth);
    }
    if (right_count > 1) {
        right_child = build_voronoi_node(f, tree, right_indices, right_count, depth + 1, max_depth);
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

static geif_status_t geif_voronoi_train(geif_forest_t *f)
{
    if (!f || f->pool_count == 0) return GEIF_ERR_EMPTY_DATASET;

    init_dimension_scales(f);

    uint32_t psi = f->config.samples_per_tree;
    if (psi > f->pool_count) psi = (uint32_t)f->pool_count;
    if (psi < 2) psi = (uint32_t)f->pool_count;

    uint32_t *subsample = (uint32_t *)malloc(psi * sizeof(uint32_t));
    if (!subsample) return GEIF_ERR_OUT_OF_MEMORY;

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
        build_voronoi_node(f, tree, subsample, psi, 0, tree_max_depth);
    }

    free(subsample);

    geif_tree_calibrate(f);
    return GEIF_OK;
}

static geif_status_t geif_voronoi_score(const geif_forest_t *f,
                                      const double *point,
                                      double *score_out,
                                      double *metric_depth_out,
                                      double *d_out_out)
{
    return geif_tree_score_point(f, point, score_out, metric_depth_out, d_out_out);
}

const geif_algo_ops_t geif_algo_ops_voronoi = {
    .type = GEIF_ALGO_VORONOI,
    .name = "voronoi",
    .description = "Pure Voronoi perpendicular bisector splits between sample pairs",
    .train = geif_voronoi_train,
    .score = geif_voronoi_score,
    .destroy = NULL
};
