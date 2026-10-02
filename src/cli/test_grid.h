/**
 * @file test_grid.h
 * @brief Population drift visualization and test grid generator for GEIF.
 */

#ifndef GEIF_TEST_GRID_H
#define GEIF_TEST_GRID_H

#include "geif/geif.h"
#include "template.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <regex.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEIF_MAX_CAT_FILTERS 32

typedef struct {
    regex_t regex;
    bool    invert;
    bool    compiled;
} geif_cat_filter_entry_t;

typedef struct {
    bool                    active;
    size_t                  count;
    geif_cat_filter_entry_t entries[GEIF_MAX_CAT_FILTERS];
} cat_filter_t;

bool geif_cat_filter_add(cat_filter_t *cf, const char *arg);
bool geif_cat_filter_allows(const cat_filter_t *cf, const char *category);
void geif_cat_filter_free(cat_filter_t *cf);

/**
 * @brief Generate a synthetic evaluation grid and training sample scatter points
 * for population drift visualization.
 *
 * @param ens Ensemble model containing categories and sub-forests
 * @param test_extension_factor Factor to extend grid beyond bounding box (e.g. 0.1)
 * @param test_sample_interval Number of grid subdivisions per dimension (default 256)
 * @param filter Optional category filter regex
 * @param threshold_is_average If true, filter grid points using forest average score
 * @param threshold_is_percentage If true, filter grid points using forest percentage score
 * @param point_tmpl Template for formatting points (e.g. "%d,0x%x")
 * @param decimals Decimal precision
 * @param list_sep Separator for list elements (e.g. ',')
 * @param low_rgb Low score RGB hex color
 * @param high_rgb High score RGB hex color
 * @param printf_format Custom printf format
 * @param print_dimension Dimension template (%j)
 * @param out_fp Output file stream
 */
void geif_generate_test_grid(const geif_ensemble_t *ens,
                             double test_extension_factor,
                             int test_sample_interval,
                             const cat_filter_t *filter,
                             double threshold,
                             bool threshold_is_average,
                             bool threshold_is_percentage,
                             const char *point_tmpl,
                             int decimals,
                             char list_sep,
                             uint32_t low_rgb,
                             uint32_t high_rgb,
                             const char *printf_format,
                             const char *print_dimension,
                             FILE *out_fp);

#ifdef __cplusplus
}
#endif

#endif // GEIF_TEST_GRID_H
