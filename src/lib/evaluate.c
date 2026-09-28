/**
 * @file evaluate.c
 * @brief Inference engine, continuous metric depth traversal, leaf void damping, and outer stadium decay.
 */

#include "geif/geif.h"
#include "geometry.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double evaluate_tree(const geif_forest_t *f,
                           const geif_tree_t *tree,
                           const double *point)
{
    if (tree->node_count == 0) {
        return 0.0;
    }

    int32_t curr = 0;
    double H_tree = 0.0;
    uint32_t d = f->dimensions;

    while (curr >= 0 && curr < (int32_t)tree->node_count) {
        const geif_node_t *node = &tree->nodes[curr];

        // Leaf node reached
        if (node->left_child == -1 && node->right_child == -1) {
            // Analytic completion bonus for duplicate / unpartitioned clusters in leaf
            if (node->sample_count > 1) {
                H_tree += log2((double)node->sample_count);
            }

            // Residual Cell Distance probe: test if query point is inside an empty void / donut hole
            const double *P_leaf = &f->sample_pool[node->leaf_point_idx * d];
            double d_residual = geif_residual_distance(point, P_leaf, f->effective_span, f->dim_active, d);

            // Cauchy-Lorentz void damping: attenuates depth if point is in an empty cavity
            double ratio = d_residual / (f->delta_nominal > 1e-12 ? f->delta_nominal : 1e-12);
            double void_damping = 1.0 / (1.0 + ratio * ratio);
            H_tree *= void_damping;

            break;
        }

        // Internal split node: accumulate metric depth
        H_tree += node->step_weight;

        const double *normal = &tree->normals_pool[node->normal_offset];
        double dot_val = geif_dot(point, normal, d);

        if (dot_val < node->pdotn) {
            curr = node->left_child;
        } else {
            curr = node->right_child;
        }
    }

    return H_tree;
}

double geif_forest_evaluate_metric_depth(const geif_forest_t *f,
                                         const double *point,
                                         double *d_out_out)
{
    if (!f || !point || f->tree_count == 0) {
        if (d_out_out) *d_out_out = 0.0;
        return 0.0;
    }

    // Step 1: Accumulate continuous metric depth across the ensemble
    double sum_H = 0.0;
    for (uint32_t t = 0; t < f->tree_count; t++) {
        sum_H += evaluate_tree(f, &f->trees[t], point);
    }
    double H_avg = sum_H / f->tree_count;

    // Step 2: Compute outer space stadium distance
    double d_out = geif_stadium_distance(point, f->envelope_min, f->envelope_max,
                                         f->effective_span, f->dimensions);

    // Step 3: Apply continuous outer space exponential damping (smooth stadium decay)
    double H_final = H_avg;
    if (d_out > 0.0) {
        H_final *= exp(-d_out);
    }

    if (d_out_out) *d_out_out = d_out;
    return H_final;
}

geif_status_t geif_forest_score_detailed(const geif_forest_t *f,
                                        const double *point,
                                        double *score_out,
                                        double *metric_depth_out,
                                        double *d_out_out)
{
    if (!f || !point || !score_out) {
        return GEIF_ERR_INVALID_ARG;
    }

    double d_out = 0.0;
    double H_final = geif_forest_evaluate_metric_depth(f, point, &d_out);

    if (metric_depth_out) *metric_depth_out = H_final;
    if (d_out_out) *d_out_out = d_out;

    if (f->H_max <= 0.0) {
        *score_out = 0.5;
        return GEIF_OK;
    }

    // Step 4: Compute normalized anomaly score relative to calibrated H_max
    double score = 1.0 - (H_final / f->H_max);

    // Clamp score to strict [0.0, 1.0] interval
    if (score < 0.0) score = 0.0;
    if (score > 1.0) score = 1.0;

    *score_out = score;
    return GEIF_OK;
}

geif_status_t geif_forest_score(const geif_forest_t *f,
                               const double *point,
                               double *score_out)
{
    return geif_forest_score_detailed(f, point, score_out, NULL, NULL);
}

geif_status_t geif_forest_get_averages(const geif_forest_t *forest,
                                      double *averages_out)
{
    if (!forest || !averages_out) return GEIF_ERR_INVALID_ARG;
    uint32_t d = forest->dimensions;
    if (d == 0) return GEIF_OK;

    if (forest->averages) {
        for (uint32_t j = 0; j < d; j++) {
            averages_out[j] = forest->averages[j];
        }
        return GEIF_OK;
    }

    if (forest->pool_count > 0 && forest->sample_pool) {
        for (uint32_t j = 0; j < d; j++) averages_out[j] = 0.0;
        for (size_t i = 0; i < forest->pool_count; i++) {
            const double *sp = &forest->sample_pool[i * d];
            for (uint32_t j = 0; j < d; j++) averages_out[j] += sp[j];
        }
        for (uint32_t j = 0; j < d; j++) averages_out[j] /= (double)forest->pool_count;
        return GEIF_OK;
    }

    if (forest->envelope_min && forest->envelope_max) {
        for (uint32_t j = 0; j < d; j++) {
            averages_out[j] = 0.5 * (forest->envelope_min[j] + forest->envelope_max[j]);
        }
        return GEIF_OK;
    }

    for (uint32_t j = 0; j < d; j++) averages_out[j] = 0.0;
    return GEIF_OK;
}

geif_status_t geif_forest_dimension_attribution(const geif_forest_t *forest,
                                               const double *point,
                                               double *attr_scores_out)
{
    if (!forest || !point || !attr_scores_out) return GEIF_ERR_INVALID_ARG;
    uint32_t d = forest->dimensions;
    if (d == 0) return GEIF_OK;

    double *baseline = (double *)malloc(d * sizeof(double));
    if (!baseline) return GEIF_ERR_OUT_OF_MEMORY;

    geif_status_t st = geif_forest_get_averages(forest, baseline);
    if (st != GEIF_OK) {
        free(baseline);
        return st;
    }

    double *test = (double *)malloc(d * sizeof(double));
    if (!test) {
        free(baseline);
        return GEIF_ERR_OUT_OF_MEMORY;
    }
    memcpy(test, baseline, d * sizeof(double));

    for (uint32_t j = 0; j < d; j++) {
        test[j] = point[j];
        double s = 0.0;
        geif_forest_score(forest, test, &s);
        attr_scores_out[j] = s;
        test[j] = baseline[j]; // restore to baseline coordinate
    }

    free(test);
    free(baseline);
    return GEIF_OK;
}

