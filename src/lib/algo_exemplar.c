/**
 * @file algo_exemplar.c
 * @brief Exemplar Algorithm: Non-tree direct Cauchy density kernel on reservoir samples.
 *
 * Implements a unified, data-driven Cauchy kernel density formulation:
 * 1. Bandwidth Clamping: Mean and standard deviation of K=5 nearest-neighbor distances,
 *    clamping per-sample bandwidths sigma_i in [max(0.5 * avg, avg - std), avg + std].
 *    Handles identical-sample datasets (avg == 0) via fallback sigma = avg_val / 10.
 * 2. Credibility Weighting: Inverse-bandwidth weights w_i = (1 / sigma_i) / sum(1 / sigma_m).
 * 3. Cauchy Kernel Pooling & Fractional Decay:
 *      D(x) = sum_i w_i / (1 + (||x - x_i|| / sigma_i)^2)
 *      s(x) = 1 - (D(x) / D_max)^0.5
 * 4. Near-Zero Baseline Calibration: D_max = 1.05 * max_i D(x_i) (+5% headroom).
 *
 * Author / Maintainer: Timo Savinen (AI-assisted)
 */

#include "algo.h"
#include "tree_common.h"
#include "geometry.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EXEMPLAR_K          5
#define EXEMPLAR_STD_FACTOR 1.0
#define EXEMPLAR_MIN_FACTOR 0.5
#define EXEMPLAR_DMAX_MULT  1.05

typedef struct {
    double *bandwidths;          /* Clamped bandwidth sigma_i [count] */
    double *inv_bandwidths_sq;   /* Precomputed 1.0 / (sigma_i * sigma_i) [count] */
    double *weights;             /* Normalized credibility weights w_i [count] */
    size_t  count;               /* Number of reservoir exemplars (N) */
    double  d_max;               /* Calibrated peak density (1.05 * max_i D(x_i)) */
} geif_exemplar_state_t;

/**
 * @brief Releases heap-allocated state for the exemplar model.
 *
 * @param f Pointer to the forest instance.
 */
static void geif_exemplar_destroy(geif_forest_t *f)
{
    if (f && f->algo_data) {
        geif_exemplar_state_t *st = (geif_exemplar_state_t *)f->algo_data;
        if (st->bandwidths) free(st->bandwidths);
        if (st->inv_bandwidths_sq) free(st->inv_bandwidths_sq);
        if (st->weights) free(st->weights);
        free(st);
        f->algo_data = NULL;
    }
}

/**
 * @brief Computes pooled Cauchy kernel density D(x) across all reservoir samples.
 *
 * D(x) = sum_i w_i / (1 + ||x - x_i||^2 / sigma_i^2)
 *
 * @param st           Exemplar internal state (with normalized weights sum(w_i) = 1).
 * @param scaled_point Scaled observation vector.
 * @param scaled_pool  Array of scaled reservoir sample coordinates.
 * @param count        Number of reservoir samples.
 * @param d            Dimensionality.
 * @return Pooled Cauchy density D(x) in [0, 1].
 */
static inline double geif_exemplar_compute_density(const geif_exemplar_state_t *st,
                                                   const double *scaled_point,
                                                   const double *scaled_pool,
                                                   size_t count,
                                                   uint32_t d)
{
    double density_sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        if (st->weights[i] <= 0.0) continue;
        const double *sample = &scaled_pool[i * d];
        double dsq = geif_dist_sq(scaled_point, sample, d);
        density_sum += st->weights[i] / (1.0 + dsq * st->inv_bandwidths_sq[i]);
    }
    return density_sum;
}

/**
 * @brief Trains the Exemplar model by computing clamped bandwidths, 1/sigma weights, and D_max.
 *
 * @param f Pointer to the forest instance.
 * @return GEIF_OK on success, or error status.
 */
static geif_status_t geif_exemplar_train(geif_forest_t *f)
{
    if (!f || f->pool_count == 0) return GEIF_ERR_EMPTY_DATASET;

    init_dimension_scales(f);

    /* Free any existing tree nodes since exemplar is tree-less */
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

    size_t n = f->pool_count;
    st->count = n;
    st->bandwidths = (double *)malloc(n * sizeof(double));
    st->inv_bandwidths_sq = (double *)malloc(n * sizeof(double));
    st->weights = (double *)malloc(n * sizeof(double));
    double *raw_d_k = (double *)malloc(n * sizeof(double));
    if (!st->bandwidths || !st->inv_bandwidths_sq || !st->weights || !raw_d_k) {
        if (raw_d_k) free(raw_d_k);
        if (st->bandwidths) free(st->bandwidths);
        if (st->inv_bandwidths_sq) free(st->inv_bandwidths_sq);
        if (st->weights) free(st->weights);
        free(st);
        return GEIF_ERR_OUT_OF_MEMORY;
    }

    uint32_t d = f->dimensions;
    const double *scaled_pool = f->scaled_pool ? f->scaled_pool : f->sample_pool;

    /* 1. Identify unique locations so duplicates are not summed multiple times in D(x) */
    int32_t *dup_of = (int32_t *)malloc(n * sizeof(int32_t));
    if (!dup_of) {
        free(raw_d_k);
        free(st->bandwidths);
        free(st->inv_bandwidths_sq);
        free(st->weights);
        free(st);
        return GEIF_ERR_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < n; i++) {
        dup_of[i] = -1;
        const double *si = &scaled_pool[i * d];
        for (size_t j = 0; j < i; j++) {
            if (dup_of[j] != -1) continue;
            const double *sj = &scaled_pool[j * d];
            if (geif_dist_sq(si, sj, d) <= 1e-12) {
                dup_of[i] = (int32_t)j;
                break;
            }
        }
    }

    /* 2. Compute mean distance to K nearest neighbors across all N samples */
    if (n <= 1) {
        for (size_t i = 0; i < n; i++) raw_d_k[i] = 0.0;
    } else {
        size_t k_nn = (n - 1 < EXEMPLAR_K) ? (n - 1) : EXEMPLAR_K;
        double best_dist_sq[EXEMPLAR_K];

        for (size_t i = 0; i < n; i++) {
            if (dup_of[i] != -1) {
                raw_d_k[i] = raw_d_k[dup_of[i]];
                continue;
            }
            const double *si = &scaled_pool[i * d];
            for (size_t k = 0; k < k_nn; k++) best_dist_sq[k] = 1e300;

            for (size_t j = 0; j < n; j++) {
                if (i == j) continue;
                const double *sj = &scaled_pool[j * d];
                double dsq = geif_dist_sq(si, sj, d);
                if (dsq < best_dist_sq[k_nn - 1]) {
                    best_dist_sq[k_nn - 1] = dsq;
                    for (size_t m = k_nn - 1; m > 0; m--) {
                        if (best_dist_sq[m] < best_dist_sq[m - 1]) {
                            double tmp = best_dist_sq[m];
                            best_dist_sq[m] = best_dist_sq[m - 1];
                            best_dist_sq[m - 1] = tmp;
                        } else {
                            break;
                        }
                    }
                }
            }

            double sum_d = 0.0;
            for (size_t k = 0; k < k_nn; k++) {
                sum_d += sqrt(best_dist_sq[k]);
            }
            raw_d_k[i] = sum_d / (double)k_nn;
        }
    }

    /* 3. Compute mean (avg_sigma) and standard deviation (std_sigma) of K-NN distances */
    double sum_d = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum_d += raw_d_k[i];
    }
    double avg_sigma = sum_d / (double)n;

    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = raw_d_k[i] - avg_sigma;
        sum_sq += diff * diff;
    }
    double std_sigma = sqrt(sum_sq / (double)n);

    /* 4. Determine bandwidth clamping bounds (handling identical-samples corner case) */
    double min_sigma, max_sigma;
    if (avg_sigma <= 1e-12) {
        /* All samples are identical: set bandwidth to avg_val / 10.0 */
        double avg_mag = 0.0;
        for (uint32_t j = 0; j < d; j++) {
            avg_mag += fabs(f->sample_pool[j]);
        }
        avg_mag /= (double)d;
        double fallback_sigma = avg_mag / 10.0;
        if (fallback_sigma < 1e-6) fallback_sigma = 0.1;
        min_sigma = fallback_sigma;
        max_sigma = fallback_sigma;
    } else {
        double floor_sigma = EXEMPLAR_MIN_FACTOR * avg_sigma;
        double lower = avg_sigma - EXEMPLAR_STD_FACTOR * std_sigma;
        min_sigma = (lower > floor_sigma) ? lower : floor_sigma;
        if (min_sigma < 1e-6) min_sigma = 1e-6;
        max_sigma = avg_sigma + EXEMPLAR_STD_FACTOR * std_sigma;
        if (max_sigma < min_sigma) max_sigma = min_sigma;
    }

    /* 5. Clamp bandwidths and assign 1/sigma credibility weights to unique locations only */
    double weight_sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double sigma = raw_d_k[i];
        if (sigma < min_sigma) sigma = min_sigma;
        if (sigma > max_sigma) sigma = max_sigma;

        st->bandwidths[i] = sigma;
        st->inv_bandwidths_sq[i] = 1.0 / (sigma * sigma);
        if (dup_of[i] == -1) {
            st->weights[i] = 1.0 / sigma;
            weight_sum += st->weights[i];
        } else {
            st->weights[i] = 0.0;
        }
    }
    free(dup_of);
    free(raw_d_k);

    if (weight_sum > 0.0) {
        for (size_t i = 0; i < n; i++) {
            st->weights[i] /= weight_sum;
        }
    }

    /* 6. Near-zero baseline calibration: D_max = 1.05 * max_i D(x_i) */
    double max_d = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (st->weights[i] <= 0.0) continue;
        const double *si = &scaled_pool[i * d];
        double d_val = geif_exemplar_compute_density(st, si, scaled_pool, n, d);
        if (d_val > max_d) max_d = d_val;
    }
    if (max_d < 1e-12) max_d = 1.0;
    st->d_max = max_d * EXEMPLAR_DMAX_MULT;
    f->algo_data = st;

    /* 6. Calibrate forest score statistics and feature averages */
    f->min_score = 0.0;
    f->max_score = 1.0;

    double sum_score = 0.0;
    double min_s = 1.0;
    for (size_t i = 0; i < n; i++) {
        const double *pt = &f->sample_pool[i * d];
        double s = 0.0;
        geif_algo_ops_exemplar.score(f, pt, &s, NULL, NULL);
        sum_score += s;
        if (s < min_s) min_s = s;
    }

    f->min_score = min_s;
    f->max_score = 1.0;
    f->average_score = (n > 0) ? (sum_score / (double)n) : 0.5;

    if (f->averages && n > 0 && f->sample_pool) {
        for (uint32_t j = 0; j < d; j++) f->averages[j] = 0.0;
        for (size_t i = 0; i < n; i++) {
            const double *pt = &f->sample_pool[i * d];
            for (uint32_t j = 0; j < d; j++) f->averages[j] += pt[j];
        }
        for (uint32_t j = 0; j < d; j++) f->averages[j] /= (double)n;
    }

    return GEIF_OK;
}

/**
 * @brief Scores a query point via Cauchy kernel pooling and fractional decay.
 *
 * D(x) = sum_i w_i / (1 + (||x - x_i|| / sigma_i)^2)
 * s(x) = 1 - (D(x) / D_max)^0.5
 *
 * @param[in]  f                Pointer to the forest.
 * @param[in]  point            Raw unscaled observation vector.
 * @param[out] score_out        Pointer to receive calibrated anomaly score.
 * @param[out] metric_depth_out Optional pointer to receive density-based depth.
 * @param[out] d_out_out        Optional pointer to receive outer distance (0.0, handled by Cauchy tail).
 * @return GEIF_OK on success, or error status.
 */
static geif_status_t geif_exemplar_score(const geif_forest_t *f,
                                         const double *point,
                                         double *score_out,
                                         double *metric_depth_out,
                                         double *d_out_out)
{
    if (!f || !point || !score_out) return GEIF_ERR_INVALID_ARG;
    if (f->pool_count == 0) {
        *score_out = 1.0 - 1e-6;
        return GEIF_OK;
    }

    geif_exemplar_state_t *st = (geif_exemplar_state_t *)f->algo_data;
    if (!st || !st->bandwidths || !st->inv_bandwidths_sq || !st->weights) {
        *score_out = 1.0 - 1e-6;
        return GEIF_OK;
    }

    uint32_t d = f->dimensions;
    double stack_buf[GEIF_STACK_BUFFER_DIMS];
    double *scaled_point = (d <= GEIF_STACK_BUFFER_DIMS) ? stack_buf : (double *)malloc(d * sizeof(double));
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

    if (d_out_out) *d_out_out = 0.0;

    const double *scaled_pool = f->scaled_pool ? f->scaled_pool : f->sample_pool;
    double D = geif_exemplar_compute_density(st, scaled_point, scaled_pool, f->pool_count, d);

    if (scaled_point != stack_buf) free(scaled_point);

    double ratio = (st->d_max > 1e-12) ? (D / st->d_max) : 0.0;
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;

    double decay = (f->config.decay > 0.0) ? f->config.decay : GEIF_DEFAULT_DECAY;
    double score = (fabs(decay - 0.5) < 1e-9) ? (1.0 - sqrt(ratio)) : (1.0 - pow(ratio, decay));
    if (score < 0.0) score = 0.0;
    if (score >= 1.0) score = 1.0 - 1e-6;

    if (metric_depth_out) {
        *metric_depth_out = (1.0 - score) * 10.0;
    }

    *score_out = score;
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
