/**
 * @file algo_exemplar.c
 * @brief Exemplar Algorithm: Non-tree direct Cauchy density kernel on reservoir samples.
 *
 * Implements a simple, unified Cauchy kernel density formulation over unique
 * exemplar locations u_i with duplicate counts c_i:
 * 1. Bandwidth via Cap-5 K-NN: In K=5 nearest-neighbor search, each duplicate set
 *    contributes at most 5 samples (at most 4 self-duplicates at distance 0), so
 *    the 5th neighbor always comes from a distinct location (b_i > 0).
 * 2. Low-End Bandwidth Floor Only: Each bandwidth b_i is floored at
 *    b_min = max(0.5 * avg_b, avg_b - std_b). No high-end clamping is applied:
 *    isolated outliers naturally get large b_i and negligible peak density (1 / b_i -> 0).
 * 3. Sublinear Duplicate Growth & Unified Cauchy Density:
 *      D(x) = sum_i (sqrt(c_i) * b_i) / (b_i^2 + ||x - u_i||^2)
 *      D_max = 1.05 * max_i D(u_i)
 *      s(x) = 1 - (D(x) / D_max)^decay
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
#define EXEMPLAR_MAX_DUPS   5
#define EXEMPLAR_MIN_FACTOR 0.5
#define EXEMPLAR_DMAX_MULT  1.05

typedef struct {
    double *u_coords;   /* Unique exemplar coordinates [u_count * d] */
    double *num;        /* Precomputed numerator sqrt(c_i) * b_i [u_count] */
    double *b_sq;       /* Precomputed denominator offset b_i^2 [u_count] */
    size_t  u_count;    /* Number of unique exemplar locations */
    double  d_max;      /* Calibrated peak density (1.05 * max_i D(u_i)) */
} geif_exemplar_state_t;

/**
 * @brief Releases heap-allocated state for the exemplar model.
 */
static void geif_exemplar_destroy(geif_forest_t *f)
{
    if (f && f->algo_data) {
        geif_exemplar_state_t *st = (geif_exemplar_state_t *)f->algo_data;
        free(st->u_coords);
        free(st->num);
        free(st->b_sq);
        free(st);
        f->algo_data = NULL;
    }
}

/**
 * @brief Computes pooled Cauchy kernel density D(x) across unique exemplars.
 *
 * D(x) = sum_i (sqrt(c_i) * b_i) / (b_i^2 + ||x - u_i||^2)
 */
static inline double geif_exemplar_compute_density(const geif_exemplar_state_t *st,
                                                   const double *scaled_point,
                                                   uint32_t d)
{
    double density_sum = 0.0;
    size_t u_count = st->u_count;
    const double *u_coords = st->u_coords;
    const double *num = st->num;
    const double *b_sq = st->b_sq;

    for (size_t i = 0; i < u_count; i++) {
        double dsq = geif_dist_sq(scaled_point, &u_coords[i * d], d);
        density_sum += num[i] / (b_sq[i] + dsq);
    }
    return density_sum;
}

/**
 * @brief Trains the Exemplar model by extracting unique locations, computing b_i, and calibrating D_max.
 */
static geif_status_t geif_exemplar_train(geif_forest_t *f)
{
    if (!f || f->pool_count == 0) return GEIF_ERR_EMPTY_DATASET;

    init_dimension_scales(f);
    geif_exemplar_destroy(f);

    geif_exemplar_state_t *st = (geif_exemplar_state_t *)calloc(1, sizeof(geif_exemplar_state_t));
    if (!st) return GEIF_ERR_OUT_OF_MEMORY;

    size_t n = f->pool_count;
    uint32_t d = f->dimensions;
    const double *scaled_pool = f->scaled_pool ? f->scaled_pool : f->sample_pool;

    st->u_coords  = (double *)malloc(n * d * sizeof(double));
    st->num       = (double *)malloc(n * sizeof(double));
    st->b_sq      = (double *)malloc(n * sizeof(double));
    size_t *counts = (size_t *)malloc(n * sizeof(size_t));
    if (!st->u_coords || !st->num || !st->b_sq || !counts) {
        free(counts);
        free(st->u_coords);
        free(st->num);
        free(st->b_sq);
        free(st);
        return GEIF_ERR_OUT_OF_MEMORY;
    }

    /* 1. Extract unique locations u_i and their duplicate counts c_i */
    size_t u_count = 0;
    size_t knn_total = 0;
    for (size_t i = 0; i < n; i++) {
        const double *si = &scaled_pool[i * d];
        bool found = false;
        for (size_t j = 0; j < u_count; j++) {
            if (geif_dist_sq(si, &st->u_coords[j * d], d) <= 1e-12) {
                counts[j]++;
                if (counts[j] <= EXEMPLAR_MAX_DUPS) knn_total++;
                found = true;
                break;
            }
        }
        if (!found) {
            memcpy(&st->u_coords[u_count * d], si, d * sizeof(double));
            counts[u_count] = 1;
            u_count++;
            knn_total++;
        }
    }
    st->u_count = u_count;

    /* 2. Compute mean distance b_i to K=5 nearest neighbors (with each duplicate set capped at 5) */
    if (knn_total <= 1) {
        for (size_t i = 0; i < u_count; i++) st->num[i] = 0.0;
    } else {
        size_t k_nn = (knn_total - 1 < EXEMPLAR_K) ? (knn_total - 1) : EXEMPLAR_K;
        double best_dist_sq[EXEMPLAR_K];

        for (size_t i = 0; i < u_count; i++) {
            const double *ui = &st->u_coords[i * d];
            for (size_t k = 0; k < k_nn; k++) best_dist_sq[k] = 1e300;

            /* Self-duplicates contribute min(c_i, 5) - 1 neighbors at distance 0 */
            size_t self_dups = (counts[i] < EXEMPLAR_MAX_DUPS ? counts[i] : EXEMPLAR_MAX_DUPS) - 1;
            for (size_t s = 0; s < self_dups && s < k_nn; s++) {
                best_dist_sq[s] = 0.0;
            }

            /* Other unique locations j contribute min(c_j, 5) neighbors at distance ||u_i - u_j|| */
            for (size_t j = 0; j < u_count; j++) {
                if (i == j) continue;
                double dsq = geif_dist_sq(ui, &st->u_coords[j * d], d);
                if (dsq >= best_dist_sq[k_nn - 1]) continue;

                size_t copies = (counts[j] < EXEMPLAR_MAX_DUPS) ? counts[j] : EXEMPLAR_MAX_DUPS;
                for (size_t c = 0; c < copies; c++) {
                    if (dsq < best_dist_sq[k_nn - 1]) {
                        best_dist_sq[k_nn - 1] = dsq;
                        for (size_t m = k_nn - 1; m > 0 && best_dist_sq[m] < best_dist_sq[m - 1]; m--) {
                            double tmp = best_dist_sq[m];
                            best_dist_sq[m] = best_dist_sq[m - 1];
                            best_dist_sq[m - 1] = tmp;
                        }
                    } else {
                        break;
                    }
                }
            }

            double sum_d = 0.0;
            for (size_t k = 0; k < k_nn; k++) {
                sum_d += sqrt(best_dist_sq[k]);
            }
            st->num[i] = sum_d / (double)k_nn; /* temporarily holds raw b_i */
        }
    }

    /* 3. Compute low-end bandwidth floor b_min weighted over the capped K-NN sample set */
    double sum_b = 0.0;
    for (size_t i = 0; i < u_count; i++) {
        size_t w = (counts[i] < EXEMPLAR_MAX_DUPS) ? counts[i] : EXEMPLAR_MAX_DUPS;
        sum_b += (double)w * st->num[i];
    }
    double avg_b = sum_b / (double)knn_total;

    double sum_sq = 0.0;
    for (size_t i = 0; i < u_count; i++) {
        size_t w = (counts[i] < EXEMPLAR_MAX_DUPS) ? counts[i] : EXEMPLAR_MAX_DUPS;
        double diff = st->num[i] - avg_b;
        sum_sq += (double)w * diff * diff;
    }
    double std_b = sqrt(sum_sq / (double)knn_total);

    double min_b;
    if (avg_b <= 1e-12) {
        /* All samples are identical: fallback to mean coordinate magnitude / 10 */
        double avg_mag = 0.0;
        for (uint32_t j = 0; j < d; j++) avg_mag += fabs(f->sample_pool[j]);
        min_b = (avg_mag / (double)d) / 10.0;
        if (min_b < 1e-6) min_b = 0.1;
    } else {
        double floor_b = EXEMPLAR_MIN_FACTOR * avg_b;
        double lower_b = avg_b - std_b;
        min_b = (lower_b > floor_b) ? lower_b : floor_b;
        if (min_b < 1e-6) min_b = 1e-6;
    }

    /* Store b_sq[i] = b_i^2 and num[i] = sqrt(c_i) * b_i */
    for (size_t i = 0; i < u_count; i++) {
        double b_i = st->num[i];
        if (b_i < min_b) b_i = min_b;
        st->b_sq[i] = b_i * b_i;
        st->num[i]  = sqrt((double)counts[i]) * b_i;
    }
    free(counts);

    /* 4. Calibrate peak density D_max = 1.05 * max_i D(u_i) */
    double max_d = 0.0;
    for (size_t i = 0; i < u_count; i++) {
        double d_val = geif_exemplar_compute_density(st, &st->u_coords[i * d], d);
        if (d_val > max_d) max_d = d_val;
    }
    st->d_max = (max_d > 1e-12) ? (max_d * EXEMPLAR_DMAX_MULT) : 1.0;
    f->algo_data = st;

    /* 5. Calibrate forest score statistics and feature averages */
    double sum_score = 0.0;
    double min_s = 1.0;
    for (size_t i = 0; i < n; i++) {
        double s = 0.0;
        geif_algo_ops_exemplar.score(f, &f->sample_pool[i * d], &s, NULL, NULL);
        sum_score += s;
        if (s < min_s) min_s = s;
    }
    f->min_score = min_s;
    f->max_score = 1.0;
    f->average_score = (n > 0) ? (sum_score / (double)n) : 0.5;

    if (f->averages && f->sample_pool) {
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
 * D(x) = sum_i (sqrt(c_i) * b_i) / (b_i^2 + ||x - u_i||^2)
 * s(x) = 1 - (D(x) / D_max)^decay
 */
static geif_status_t geif_exemplar_score(const geif_forest_t *f,
                                         const double *point,
                                         double *score_out,
                                         double *metric_depth_out,
                                         double *d_out_out)
{
    if (!f || !point || !score_out) return GEIF_ERR_INVALID_ARG;

    const geif_exemplar_state_t *st = (const geif_exemplar_state_t *)f->algo_data;
    if (!st || st->u_count == 0) {
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

    double D = geif_exemplar_compute_density(st, scaled_point, d);
    if (scaled_point != stack_buf) free(scaled_point);

    double ratio = D / st->d_max;
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
