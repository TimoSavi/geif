/**
 * @file forest.c
 * @brief Lifecycle management, initialization, and deallocation for GEIF forests.
 */

#include "geif/geif.h"
#include "algo.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

geif_config_t geif_config_default(void)
{
    geif_config_t cfg;
    cfg.tree_count        = GEIF_DEFAULT_TREE_COUNT;
    cfg.samples_per_tree  = GEIF_DEFAULT_SAMPLES_PER_TREE;
    cfg.max_depth         = GEIF_DEFAULT_MAX_DEPTH;
    cfg.kappa             = GEIF_DEFAULT_KAPPA;
    cfg.alpha             = GEIF_DEFAULT_ALPHA;
    cfg.ceiling_factor    = GEIF_DEFAULT_CEILING_FACTOR;
    cfg.seed              = 0;
    cfg.algo              = GEIF_ALGO_DEFAULT;
    return cfg;
}

geif_status_t geif_forest_create(geif_forest_t **forest_out,
                                uint32_t dimensions,
                                const geif_config_t *config)
{
    if (!forest_out || dimensions == 0) {
        return GEIF_ERR_INVALID_ARG;
    }

    geif_forest_t *f = (geif_forest_t *)calloc(1, sizeof(geif_forest_t));
    if (!f) {
        return GEIF_ERR_OUT_OF_MEMORY;
    }

    f->dimensions = dimensions;
    f->config = config ? *config : geif_config_default();

    if (f->config.tree_count == 0) f->config.tree_count = GEIF_DEFAULT_TREE_COUNT;
    if (f->config.samples_per_tree == 0) f->config.samples_per_tree = GEIF_DEFAULT_SAMPLES_PER_TREE;
    if (f->config.max_depth == 0) f->config.max_depth = GEIF_DEFAULT_MAX_DEPTH;
    if (f->config.kappa <= 1.0) f->config.kappa = GEIF_DEFAULT_KAPPA;
    if (f->config.alpha < 0.0) f->config.alpha = GEIF_DEFAULT_ALPHA;

    f->tree_count = f->config.tree_count;
    f->pool_capacity = (size_t)f->tree_count * f->config.samples_per_tree;

    // Allocate coordinate envelope arrays
    f->envelope_min   = (double *)malloc(dimensions * sizeof(double));
    f->envelope_max   = (double *)malloc(dimensions * sizeof(double));
    f->envelope_span  = (double *)malloc(dimensions * sizeof(double));
    f->effective_span = (double *)malloc(dimensions * sizeof(double));
    f->dim_active     = (uint8_t *)malloc(dimensions * sizeof(uint8_t));

    // Allocate contiguous reservoir sample pool: [pool_capacity * dimensions]
    f->sample_pool    = (double *)malloc(f->pool_capacity * dimensions * sizeof(double));

    // Allocate dimension averages [dimensions]
    f->averages       = (double *)calloc(dimensions, sizeof(double));

    // Allocate tree headers
    f->trees          = (geif_tree_t *)calloc(f->tree_count, sizeof(geif_tree_t));

    if (!f->envelope_min || !f->envelope_max || !f->envelope_span ||
        !f->effective_span || !f->dim_active || !f->sample_pool || !f->averages || !f->trees) {
        geif_forest_destroy(f);
        return GEIF_ERR_OUT_OF_MEMORY;
    }

    // Initialize envelopes to extreme values
    for (uint32_t j = 0; j < dimensions; j++) {
        f->envelope_min[j]   = 1e300;
        f->envelope_max[j]   = -1e300;
        f->envelope_span[j]  = 0.0;
        f->effective_span[j] = 1.0;
        f->dim_active[j]     = 1;
    }

    f->decimals = 6;

    if (f->config.seed == 0) {
        srand((unsigned int)(time(NULL) ^ 0x5DEECE66DULL));
    } else {
        srand(f->config.seed);
    }

    *forest_out = f;
    return GEIF_OK;
}

void geif_forest_destroy(geif_forest_t *f)
{
    if (!f) return;

    if (f->trees) {
        for (uint32_t t = 0; t < f->tree_count; t++) {
            if (f->trees[t].nodes) {
                free(f->trees[t].nodes);
            }
            if (f->trees[t].normals_pool) {
                free(f->trees[t].normals_pool);
            }
            if (f->trees[t].leaf_samples) {
                free(f->trees[t].leaf_samples);
            }
        }
        free(f->trees);
    }

    const geif_algo_ops_t *ops = geif_algo_get_ops(f->config.algo);
    if (ops && ops->destroy) {
        ops->destroy(f);
    }

    if (f->envelope_min)   free(f->envelope_min);
    if (f->envelope_max)   free(f->envelope_max);
    if (f->envelope_span)  free(f->envelope_span);
    if (f->effective_span) free(f->effective_span);
    if (f->dim_active)     free(f->dim_active);
    if (f->sample_pool)    free(f->sample_pool);
    if (f->scaled_pool)    free(f->scaled_pool);
    if (f->averages)       free(f->averages);

    free(f);
}

void geif_forest_summary(const geif_forest_t *f, char *buf, size_t size)
{
    if (!f || !buf || size == 0) return;

    uint32_t active_dims = 0;
    for (uint32_t j = 0; j < f->dimensions; j++) {
        if (f->dim_active[j]) active_dims++;
    }

    snprintf(buf, size,
             "GEIF Forest Summary:\n"
             "  Dimensions:          %u (Active: %u, Inactive/Constant: %u)\n"
             "  Trees:               %u\n"
             "  Samples/Tree (psi):  %u\n"
             "  Max Depth Cap:       %u\n"
             "  Sample Pool Count:   %zu / %zu (Stream rows seen: %llu)\n"
             "  Nominal Spacing:     %.6f\n"
             "  H_train_max:         %.6f\n"
             "  H_max (Scale):       %.6f (Headroom: %.2f)\n",
             f->dimensions, active_dims, f->dimensions - active_dims,
             f->tree_count, f->config.samples_per_tree, f->config.max_depth,
             f->pool_count, f->pool_capacity, (unsigned long long)f->total_rows_seen,
             f->delta_nominal, f->H_train_max, f->H_max, f->config.kappa);
}
