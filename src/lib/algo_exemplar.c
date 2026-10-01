/**
 * @file algo_exemplar.c
 * @brief Exemplar Algorithm: Non-tree direct SIMD Cauchy density kernel on reservoir samples.
 */

#include "algo.h"
#include "tree_common.h"
#include "geometry.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EXEMPLAR_K 5

typedef struct {
    double *bandwidths;
    size_t  count;
    double  avg_sigma;
} geif_exemplar_state_t;

static void geif_exemplar_destroy(geif_forest_t *f)
{
    if (f && f->algo_data) {
        geif_exemplar_state_t *st = (geif_exemplar_state_t *)f->algo_data;
        if (st->bandwidths) free(st->bandwidths);
        free(st);
        f->algo_data = NULL;
    }
}

static geif_status_t geif_exemplar_train(geif_forest_t *f)
{
    if (!f || f->pool_count == 0) return GEIF_ERR_EMPTY_DATASET;

    init_dimension_scales(f);

    // Free any existing tree nodes since exemplar is tree-less
    for (uint32_t t = 0; t < f->tree_count; t++) {
        geif_tree_t *tree = &f->trees[t];
        if (tree->nodes) { free(tree->nodes); tree->nodes = NULL; }
        if (tree->normals_pool) { free(tree->normals_pool); tree->normals_pool = NULL; }
        if (tree->leaf_samples) { free(tree->leaf_samples); tree->leaf_samples = NULL; }
        tree->node_count = 0;
    }

    geif_exemplar_destroy(f);

    geif_exemplar_state_t *st = (geif_exemplar_state_t *)calloc(1, sizeof(geif_exemplar_state_t));
    if (!st) return GEIF_ERR_OUT_OF_MEMORY;

    st->count = f->pool_count;
    st->bandwidths = (double *)malloc(st->count * sizeof(double));
    if (!st->bandwidths) {
        free(st);
        return GEIF_ERR_OUT_OF_MEMORY;
    }

    uint32_t d = f->dimensions;
    double sum_sigma = 0.0;
    size_t k_nn = (f->pool_count < EXEMPLAR_K) ? f->pool_count : EXEMPLAR_K;
    if (k_nn < 2) k_nn = 2;

    // Precompute adaptive local bandwidth sigma_i for each sample in pool
    for (size_t i = 0; i < f->pool_count; i++) {
        const double *si = &f->scaled_pool[i * d];
        double best_dist_sq[EXEMPLAR_K];
        for (size_t k = 0; k < k_nn; k++) best_dist_sq[k] = 1e300;

        for (size_t j = 0; j < f->pool_count; j++) {
            if (i == j) continue;
            const double *sj = &f->scaled_pool[j * d];
            double dsq = geif_dist_sq(si, sj, d);
            if (dsq < best_dist_sq[k_nn - 1]) {
                best_dist_sq[k_nn - 1] = dsq;
                for (size_t m = k_nn - 1; m > 0; m--) {
                    if (best_dist_sq[m] < best_dist_sq[m - 1]) {
                        double tmp = best_dist_sq[m];
                        best_dist_sq[m] = best_dist_sq[m - 1];
                        best_dist_sq[m - 1] = tmp;
                    } else break;
                }
            }
        }

        double sum_d = 0.0;
        for (size_t k = 0; k < k_nn; k++) {
            sum_d += sqrt(best_dist_sq[k]);
        }
        double sigma = sum_d / (double)k_nn;
        if (sigma < 1e-6) sigma = 1e-6;
        st->bandwidths[i] = sigma;
        sum_sigma += sigma;
    }

    st->avg_sigma = (st->count > 0) ? (sum_sigma / (double)st->count) : 1.0;
    f->algo_data = st;

    // Calibrate min_score and score bounds
    f->min_score = 0.0;
    f->max_score = 1.0;

    double sum_score = 0.0;
    double min_s = 1.0;
    for (size_t i = 0; i < f->pool_count; i++) {
        const double *pt = &f->sample_pool[i * d];
        double s = 0.0;
        geif_algo_ops_exemplar.score(f, pt, &s, NULL, NULL);
        sum_score += s;
        if (s < min_s) min_s = s;
    }

    f->min_score = min_s;
    f->max_score = 1.0;
    f->average_score = (f->pool_count > 0) ? (sum_score / (double)f->pool_count) : 0.5;

    if (f->averages && f->pool_count > 0 && f->sample_pool) {
        for (uint32_t j = 0; j < d; j++) f->averages[j] = 0.0;
        for (size_t i = 0; i < f->pool_count; i++) {
            const double *pt = &f->sample_pool[i * d];
            for (uint32_t j = 0; j < d; j++) f->averages[j] += pt[j];
        }
        for (uint32_t j = 0; j < d; j++) f->averages[j] /= (double)f->pool_count;
    }

    return GEIF_OK;
}

static geif_status_t geif_exemplar_score(const geif_forest_t *f,
                                        const double *point,
                                        double *score_out,
                                        double *metric_depth_out,
                                        double *d_out_out)
{
    if (!f || !point || !score_out) return GEIF_ERR_INVALID_ARG;
    if (f->pool_count == 0) {
        *score_out = 1.0;
        return GEIF_OK;
    }

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

    // Outer space distance calculation
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
    double d_out = sqrt(dist_out_sq);
    if (d_out_out) *d_out_out = d_out;

    // Find K nearest neighbors in scaled pool
    size_t k_nn = (f->pool_count < EXEMPLAR_K) ? f->pool_count : EXEMPLAR_K;
    if (k_nn < 1) k_nn = 1;

    double best_dist_sq[EXEMPLAR_K];
    size_t best_idx[EXEMPLAR_K];
    for (size_t k = 0; k < k_nn; k++) {
        best_dist_sq[k] = 1e300;
        best_idx[k] = 0;
    }

    for (size_t i = 0; i < f->pool_count; i++) {
        const double *sample = (f->scaled_pool) ? &f->scaled_pool[i * d] : &f->sample_pool[i * d];
        double dsq = geif_dist_sq(scaled_point, sample, d);
        if (dsq < best_dist_sq[k_nn - 1]) {
            best_dist_sq[k_nn - 1] = dsq;
            best_idx[k_nn - 1] = i;
            for (size_t m = k_nn - 1; m > 0; m--) {
                if (best_dist_sq[m] < best_dist_sq[m - 1]) {
                    double tmp_d = best_dist_sq[m];
                    size_t tmp_i = best_idx[m];
                    best_dist_sq[m] = best_dist_sq[m - 1];
                    best_idx[m] = best_idx[m - 1];
                    best_dist_sq[m - 1] = tmp_d;
                    best_idx[m - 1] = tmp_i;
                } else break;
            }
        }
    }

    if (scaled_point != stack_buf) free(scaled_point);

    geif_exemplar_state_t *st = (geif_exemplar_state_t *)f->algo_data;
    double density_sum = 0.0;
    for (size_t k = 0; k < k_nn; k++) {
        double dist = sqrt(best_dist_sq[k]);
        size_t idx = best_idx[k];
        double sigma = (st && st->bandwidths && idx < st->count) ? st->bandwidths[idx] : f->avg_sample_dist;
        if (sigma < 1e-6) sigma = 1e-6;
        double ratio = dist / sigma;
        density_sum += 1.0 / (1.0 + ratio * ratio);
    }

    double D = density_sum / (double)k_nn;
    if (D > 1.0) D = 1.0;
    if (D < 0.0) D = 0.0;

    double score = 1.0 - D;
    if (metric_depth_out) *metric_depth_out = D * 10.0;

    // Outer space exponential attenuation
    if (d_out > 0.0) {
        double d_norm = (target_range > 1e-12) ? (d_out / target_range) : d_out;
        score = 1.0 - (1.0 - score) * exp(-GEIF_OUTER_DECAY_RATE * d_norm);
    }
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

const geif_algo_ops_t geif_algo_ops_exemplar = {
    .type = GEIF_ALGO_EXEMPLAR,
    .name = "exemplar",
    .description = "Direct non-tree SIMD Cauchy density kernel on reservoir samples",
    .train = geif_exemplar_train,
    .score = geif_exemplar_score,
    .destroy = geif_exemplar_destroy
};
