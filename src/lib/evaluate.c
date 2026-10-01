/**
 * @file evaluate.c
 * @brief Inference engine, continuous metric depth traversal, leaf void damping, and outer stadium decay.
 */

#include "geif/geif.h"
#include "geometry.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double s_fast_c_cache[GEIF_FAST_C_SAMPLES];
static bool s_c_cache_initialized = false;

void geif_init_c_cache(void)
{
    if (s_c_cache_initialized) return;
    s_fast_c_cache[0] = 0.0;
    s_fast_c_cache[1] = 0.0;
    double H = 0.0;
    for (int n = 2; n < GEIF_FAST_C_SAMPLES; n++) {
        H += 1.0 / (double)(n - 1);
        s_fast_c_cache[n] = 2.0 * H - (2.0 * (double)(n - 1) / (double)n);
    }
    s_c_cache_initialized = true;
}

double geif_c(double n)
{
    if (!s_c_cache_initialized) geif_init_c_cache();
    if (n <= 1.0) return 0.0;
    int idx = (int)n;
    if (idx < GEIF_FAST_C_SAMPLES) {
        return s_fast_c_cache[idx];
    }
    double m = n - 1.0;
    double H = log(m) + 0.5772156649015328606 + (1.0 / (2.0 * m)) - (1.0 / (12.0 * m * m));
    return 2.0 * H - (2.0 * m / n);
}

static double evaluate_tree(const geif_forest_t *f,
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
            if (f->avg_sample_dist > 0.0 && node->sample_count > 0 && tree->leaf_samples) {
                // Find nearest sample in this leaf
                double min_dist_sq = 1e300;
                const uint32_t *leaf_s = &tree->leaf_samples[node->leaf_sample_offset];
                for (int32_t i = 0; i < node->sample_count; i++) {
                    uint32_t s_idx = leaf_s[i];
                    const double *sample = (f->scaled_pool) ? &f->scaled_pool[s_idx * d]
                                                           : &f->sample_pool[s_idx * d];
                    double dist_sq = geif_dist_sq(scaled_point, sample, d);
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                    }
                }
                double rel_dist = (sqrt(min_dist_sq) / f->avg_sample_dist) + MIN_REL_DIST;
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

double geif_forest_evaluate_metric_depth(const geif_forest_t *f,
                                         const double *point,
                                         double *d_out_out)
{
    if (!f || !point || f->tree_count == 0) {
        if (d_out_out) *d_out_out = 0.0;
        return 0.0;
    }

    uint32_t d = f->dimensions;
    double stack_buf[64];
    double *scaled_point = (d <= 64) ? stack_buf : (double *)malloc(d * sizeof(double));
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

    // Step 1: Accumulate path length across the ensemble
    double sum_H = 0.0;
    for (uint32_t t = 0; t < f->tree_count; t++) {
        sum_H += evaluate_tree(f, &f->trees[t], scaled_point);
    }
    double H_avg = sum_H / (double)f->tree_count;

    // Step 2: Compute outer space distance in scaled space
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

    double psi = (f->config.samples_per_tree > 0) ? (double)f->config.samples_per_tree : 256.0;
    double c_psi = (f->c_factor > 0.0) ? f->c_factor : geif_c(psi);
    if (c_psi <= 0.0) c_psi = 1.0;

    // Standard Isolation Forest score s = 1.0 / 2^(H / c)
    double score = 1.0 / pow(2.0, H_final / c_psi);

    // Outer space exponential attenuation from CEIF:
    // score = 1.0 - (1.0 - score) * exp(-OUTER_DECAY_RATE * d_norm)
    if (d_out > 0.0) {
        double target_range = (f->scale_range_idx >= 0 && f->envelope_span) ? f->envelope_span[f->scale_range_idx] : 1.0;
        double d_norm = (target_range > 1e-12) ? (d_out / target_range) : d_out;
        score = 1.0 - (1.0 - score) * exp(-GEIF_OUTER_DECAY_RATE * d_norm);
    }
    if (score >= 1.0) score = 1.0 - 1e-6;

    if (f->scale_score) {
        // Scale score to [0, 1] using calibrated min_score and max_score
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

static int pscore_cmp(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

double geif_forest_calculate_percentile_score(const geif_forest_t *forest, double percentile)
{
    if (!forest || forest->pool_count == 0 || !forest->sample_pool) {
        return (forest && forest->average_score > 0.0) ? forest->average_score : 0.5;
    }
    if (percentile < 0.0) percentile = 0.0;
    if (percentile > 100.0) percentile = 100.0;

    double *scores = (double *)malloc(forest->pool_count * sizeof(double));
    if (!scores) {
        return (forest->average_score > 0.0) ? forest->average_score : 0.5;
    }

    for (size_t i = 0; i < forest->pool_count; i++) {
        const double *sample = &forest->sample_pool[i * forest->dimensions];
        geif_forest_score(forest, sample, &scores[i]);
    }

    qsort(scores, forest->pool_count, sizeof(double), pscore_cmp);

    size_t idx = (size_t)((double)(forest->pool_count - 1) * (percentile / 100.0));
    if (idx >= forest->pool_count) idx = forest->pool_count - 1;
    double score = scores[idx];
    free(scores);

    return score;
}

