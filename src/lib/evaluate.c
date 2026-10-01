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

#include "algo.h"
#include "tree_common.h"

double geif_forest_evaluate_metric_depth(const geif_forest_t *f,
                                         const double *point,
                                         double *d_out_out)
{
    if (!f || !point) {
        if (d_out_out) *d_out_out = 0.0;
        return 0.0;
    }
    double depth = 0.0;
    double score = 0.0;
    const geif_algo_ops_t *ops = geif_algo_get_ops(f->config.algo);
    if (ops && ops->score) {
        ops->score(f, point, &score, &depth, d_out_out);
        return depth;
    }
    return geif_tree_evaluate_metric_depth(f, point, d_out_out);
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
    const geif_algo_ops_t *ops = geif_algo_get_ops(f->config.algo);
    if (!ops || !ops->score) {
        return GEIF_ERR_NOT_SUPPORTED;
    }
    return ops->score(f, point, score_out, metric_depth_out, d_out_out);
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

