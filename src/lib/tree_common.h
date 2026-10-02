/**
 * @file tree_common.h
 * @brief Common tree manipulation, memory allocation, and evaluation utilities.
 */

#ifndef GEIF_TREE_COMMON_H
#define GEIF_TREE_COMMON_H

#include "geif/geif.h"
#include "geometry.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_dimension_scales(geif_forest_t *f);
int32_t allocate_node(geif_tree_t *tree);
uint32_t append_normal(geif_tree_t *tree, const double *normal, uint32_t d);
uint32_t append_leaf_samples(geif_tree_t *tree, const uint32_t *samples, size_t count);

double geif_calc_leaf_rel_dist(const geif_forest_t *f,
                              const geif_tree_t *tree,
                              const geif_node_t *node,
                              const double *scaled_point);

double evaluate_tree(const geif_forest_t *f,
                    const geif_tree_t *tree,
                    const double *scaled_point);

double geif_tree_evaluate_metric_depth(const geif_forest_t *f,
                                      const double *point,
                                      double *d_out_out);

geif_status_t geif_tree_score_point(const geif_forest_t *f,
                                   const double *point,
                                   double *score_out,
                                   double *metric_depth_out,
                                   double *d_out_out);

void geif_tree_find_max_height(const geif_forest_t *f,
                               const geif_tree_t *t,
                               int32_t node_idx,
                               double depth,
                               double *max_h);

void geif_tree_calibrate(geif_forest_t *f);

#ifdef __cplusplus
}
#endif

#endif /* GEIF_TREE_COMMON_H */
