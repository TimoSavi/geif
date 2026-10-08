/**
 * @file algo_exemplar.c
 * @brief Exemplar Algorithm: Non-tree direct SIMD Cauchy density kernel on reservoir samples.
 *
 * Implements the 5-Pillar Architectural Remedy:
 * 1. Robust Bandwidth Regularization (Clamped sigma_i based on global median spacing)
 * 2. Pilot Density Credibility Weighting (w_i in (0, 1] suppressing isolated noise outliers)
 * 3. Smooth Continuous Kernel Pooling (C-infinity all-reservoir evaluation, eliminating caustic ridges & starburst rays)
 * 4. Asymptotic Outer Stadium Decay (Monotonic Euclidean bounding box attenuation)
 * 5. Zero Kelvin Baseline Calibration (Exact 0.000 anchor at core cluster centroid)
 *
 * Author / Maintainer: Timo Savinen (AI-assisted)
 */

#include "algo.h"
#include "tree_common.h"
#include "geometry.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EXEMPLAR_K 5

typedef struct {
    double *bandwidths;          /* Clamped bandwidth sigma_i [count] */
    double *inv_bandwidths_sq;   /* Precomputed 1.0 / (sigma_i * sigma_i) [count] */
    double *weights;             /* Pilot density credibility weights w_i [count] */
    size_t  count;               /* Number of reservoir exemplars (N) */
    double  sigma_median;        /* Global robust median spacing */
    double  weight_sum;          /* Precomputed sum of all weights w_i */
    double  max_core_density;    /* Peak observed density (D_max) for Zero Kelvin floor */
    double  min_core_density;    /* Lowest observed training density */
} geif_exemplar_state_t;

/**
 * @brief Releases heap-allocated adaptive bandwidth state for the exemplar model.
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
 * @brief Double comparison callback for qsort.
 */
static int compare_doubles(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/**
 * @brief Computes smooth all-reservoir Cauchy kernel density D(x).
 *
 * D(x) = sum_i(w_i / (1 + dist_i^2 / sigma_i^2)) / sum_i(w_i)
 *
 * Infinitely differentiable (C-infinity) everywhere. Eliminates order-K Voronoi
 * boundary derivative jumps (caustic fringes) and starburst rays.
 *
 * @param st           Exemplar internal state.
 * @param scaled_point Scaled observation vector.
 * @param scaled_pool  Array of reservoir sample coordinates.
 * @param count        Number of reservoir samples.
 * @param d            Dimensionality.
 * @return Aggregated normalized kernel density D in [0, 1].
 */
static inline double geif_exemplar_compute_density(const geif_exemplar_state_t *st,
                                                   const double *scaled_point,
                                                   const double *scaled_pool,
                                                   size_t count,
                                                   uint32_t d)
{
    double density_sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        const double *sample = &scaled_pool[i * d];
        double dsq = 0.0;
        for (uint32_t j = 0; j < d; j++) {
            double diff = scaled_point[j] - sample[j];
            dsq += diff * diff;
        }
        double ratio_sq = dsq * st->inv_bandwidths_sq[i];
        density_sum += st->weights[i] / (1.0 + ratio_sq);
    }
    return (st->weight_sum > 0.0) ? (density_sum / st->weight_sum) : 0.0;
}

/**
 * @brief Trains the Exemplar model by computing regularized bandwidths and credibility weights.
 *
 * Bypasses binary tree construction entirely. For each sample in the reservoir pool:
 * 1. Computes mean distance to K=5 nearest neighbors in scaled space.
 * 2. Computes global robust median spacing sigma_median.
 * 3. Clamps bandwidths: sigma_i in [0.5 * sigma_median, 1.5 * sigma_median].
 * 4. Computes pilot density credibility weights w_i = 1 / (1 + (excess / sigma_median)^2).
 * 5. Precomputes max core density D_max for Zero Kelvin calibration.
 *
 * @param f Pointer to the forest instance.
 * @return GEIF_OK on success, or error status.
 */
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

    size_t n = f->pool_count;
    st->count = n;
    st->bandwidths = (double *)malloc(n * sizeof(double));
    st->inv_bandwidths_sq = (double *)malloc(n * sizeof(double));
    st->weights = (double *)malloc(n * sizeof(double));
    if (!st->bandwidths || !st->inv_bandwidths_sq || !st->weights) {
        if (st->bandwidths) free(st->bandwidths);
        if (st->inv_bandwidths_sq) free(st->inv_bandwidths_sq);
        if (st->weights) free(st->weights);
        free(st);
        return GEIF_ERR_OUT_OF_MEMORY;
    }

    uint32_t d = f->dimensions;
    double *raw_d_k = (double *)malloc(n * sizeof(double));
    double *sorted_d_k = (double *)malloc(n * sizeof(double));
    if (!raw_d_k || !sorted_d_k) {
        if (raw_d_k) free(raw_d_k);
        if (sorted_d_k) free(sorted_d_k);
        if (st->bandwidths) free(st->bandwidths);
        if (st->inv_bandwidths_sq) free(st->inv_bandwidths_sq);
        if (st->weights) free(st->weights);
        free(st);
        return GEIF_ERR_OUT_OF_MEMORY;
    }

    size_t k_nn = (n < EXEMPLAR_K) ? n : EXEMPLAR_K;
    if (k_nn < 2 && n > 1) k_nn = 2;

    const double *scaled_pool = f->scaled_pool ? f->scaled_pool : f->sample_pool;

    if (n <= 1) {
        raw_d_k[0] = (f->avg_sample_dist > 1e-6) ? f->avg_sample_dist : 1.0;
        sorted_d_k[0] = raw_d_k[0];
    } else {
        double best_dist_sq[EXEMPLAR_K];
        for (size_t i = 0; i < n; i++) {
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
                        } else break;
                    }
                }
            }

            double sum_d = 0.0;
            size_t valid_k = 0;
            for (size_t k = 0; k < k_nn; k++) {
                if (best_dist_sq[k] < 1e299) {
                    sum_d += sqrt(best_dist_sq[k]);
                    valid_k++;
                }
            }
            double mean_d = (valid_k > 0) ? (sum_d / (double)valid_k) : ((f->avg_sample_dist > 1e-6) ? f->avg_sample_dist : 1.0);
            if (mean_d < 1e-6) mean_d = 1e-6;
            raw_d_k[i] = mean_d;
            sorted_d_k[i] = mean_d;
        }
    }

    // Compute global robust median spacing
    qsort(sorted_d_k, n, sizeof(double), compare_doubles);
    double sigma_med;
    if (n % 2 == 1) {
        sigma_med = sorted_d_k[n / 2];
    } else {
        sigma_med = 0.5 * (sorted_d_k[n / 2 - 1] + sorted_d_k[n / 2]);
    }
    if (sigma_med < 1e-6) sigma_med = 1e-6;
    st->sigma_median = sigma_med;
    free(sorted_d_k);

    // Pillar 1: Clamped Bandwidth & Pillar 2: Pilot Credibility Weighting
    double min_sigma = 0.5 * sigma_med;
    double max_sigma = 1.5 * sigma_med;
    double weight_sum = 0.0;

    for (size_t i = 0; i < n; i++) {
        double d_k_val = raw_d_k[i];
        double sigma = d_k_val;
        if (sigma < min_sigma) sigma = min_sigma;
        if (sigma > max_sigma) sigma = max_sigma;
        if (sigma < 1e-6) sigma = 1e-6;

        st->bandwidths[i] = sigma;
        st->inv_bandwidths_sq[i] = 1.0 / (sigma * sigma);

        double excess = (d_k_val > sigma_med) ? (d_k_val - sigma_med) : 0.0;
        double ratio = excess / sigma_med;
        double w = 1.0 / (1.0 + ratio * ratio);
        st->weights[i] = w;
        weight_sum += w;
    }
    free(raw_d_k);

    if (weight_sum < 1e-12) weight_sum = 1.0;
    st->weight_sum = weight_sum;
    f->algo_data = st;

    // Pillar 5: Zero Kelvin baseline calibration
    double max_d = 0.0;
    double min_d = 1e300;
    for (size_t i = 0; i < n; i++) {
        const double *si = &scaled_pool[i * d];
        double d_val = geif_exemplar_compute_density(st, si, scaled_pool, n, d);
        if (d_val > max_d) max_d = d_val;
        if (d_val < min_d) min_d = d_val;
    }
    if (max_d < 1e-6) max_d = 1.0;
    st->max_core_density = max_d;
    st->min_core_density = (min_d < 1e299) ? min_d : 0.0;

    // Calibrate min_score and score bounds
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
 * @brief Scores a query point via direct Cauchy kernel density estimation.
 *
 * Smoothly aggregates kernel density across all reservoir exemplars:
 * D(x) = sum_i(w_i / (1 + (d_i / sigma_i)^2)) / sum_i(w_i).
 * Anomaly score is evaluated via Zero Kelvin calibration:
 * s_raw = (D_max - D(x)) / D_max, followed by outer space stadium attenuation.
 *
 * @param[in]  f               Pointer to the forest.
 * @param[in]  point           Raw unscaled observation vector.
 * @param[out] score_out       Pointer to receive calibrated anomaly score.
 * @param[out] metric_depth_out Optional pointer to receive density-based depth.
 * @param[out] d_out_out       Optional pointer to receive outer distance d_out.
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

    // Outer space Euclidean distance calculation
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

    // Pillar 3: Smooth All-Reservoir Pooling
    const double *scaled_pool = f->scaled_pool ? f->scaled_pool : f->sample_pool;
    double D = geif_exemplar_compute_density(st, scaled_point, scaled_pool, f->pool_count, d);

    if (scaled_point != stack_buf) free(scaled_point);

    // Pillar 5: Zero Kelvin Baseline Calibration
    double max_core = (st->max_core_density > 1e-12) ? st->max_core_density : 1.0;
    double raw_score = (max_core - D) / max_core;
    if (raw_score < 0.0) raw_score = 0.0;
    if (raw_score > 1.0) raw_score = 1.0;

    double score = raw_score;

    // Pillar 4: Asymptotic Outer Stadium Decay
    if (d_out > 0.0) {
        double d_norm = (target_range > 1e-12) ? (d_out / target_range) : d_out;
        score = 1.0 - (1.0 - score) * exp(-GEIF_OUTER_DECAY_RATE * d_norm);
    }
    if (score < 0.0) score = 0.0;
    if (score >= 1.0) score = 1.0 - 1e-6;

    if (metric_depth_out) {
        *metric_depth_out = (1.0 - score) * 10.0;
    }

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
