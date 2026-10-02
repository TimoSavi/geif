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

/**
 * @brief Initialize dimension scales, bounding envelopes, and nominal distances.
 *
 * Scans the sample pool to compute coordinate minimums, maximums, and active
 * dimensions. Normalizes features across disparate aspect ratios.
 *
 * @param f Pointer to the forest instance.
 */
void init_dimension_scales(geif_forest_t *f);

/**
 * @brief Dynamically allocates a new node slot in the tree's node pool.
 *
 * Grows tree->nodes buffer exponentially as needed.
 *
 * @param tree Pointer to the isolation tree.
 * @return 0-based node index on success, or -1 on allocation failure.
 */
int32_t allocate_node(geif_tree_t *tree);

/**
 * @brief Appends a normal vector or hypersphere center to the tree's float pool.
 *
 * @param tree   Pointer to the isolation tree.
 * @param normal Coordinate vector of length d.
 * @param d      Dimensionality of the feature vector.
 * @return Offset in tree->normals_pool where vector was stored.
 */
uint32_t append_normal(geif_tree_t *tree, const double *normal, uint32_t d);

/**
 * @brief Stores indices of samples terminating in a leaf node.
 *
 * @param tree    Pointer to the isolation tree.
 * @param samples Array of sample indices in the forest pool.
 * @param count   Number of sample indices to append.
 * @return Offset in tree->leaf_samples where indices start.
 */
uint32_t append_leaf_samples(geif_tree_t *tree, const uint32_t *samples, size_t count);

/**
 * @brief Computes relative distance from query point to nearest leaf samples.
 *
 * Evaluates Euclidean distance to the 2^D nearest leaf samples (or heap-bounded
 * subset) relative to nominal cluster density (avg_sample_dist) for cavity damping.
 *
 * @param f            Pointer to the forest.
 * @param tree         Pointer to the isolation tree.
 * @param node         Leaf node containing sample indices.
 * @param scaled_point Query point in normalized coordinate space.
 * @return Relative distance factor (>= MIN_REL_DIST).
 */
double geif_calc_leaf_rel_dist(const geif_forest_t *f,
                              const geif_tree_t *tree,
                              const geif_node_t *node,
                              const double *scaled_point);

/**
 * @brief Traverses a single isolation tree to evaluate continuous metric depth.
 *
 * @param f            Pointer to the forest.
 * @param tree         Pointer to the tree.
 * @param scaled_point Query point in normalized coordinate space.
 * @return Accumulated continuous depth including leaf residual completion.
 */
double evaluate_tree(const geif_forest_t *f,
                    const geif_tree_t *tree,
                    const double *scaled_point);

/**
 * @brief Evaluates average metric depth H across all trees in the forest.
 *
 * @param[in]  f         Pointer to the forest.
 * @param[in]  point     Raw unscaled query point.
 * @param[out] d_out_out Optional pointer to receive outer space Euclidean distance.
 * @return Average continuous metric depth across ensemble.
 */
double geif_tree_evaluate_metric_depth(const geif_forest_t *f,
                                      const double *point,
                                      double *d_out_out);

/**
 * @brief Evaluates calibrated anomaly score, metric depth, and outer distance.
 *
 * @param[in]  f               Pointer to the forest.
 * @param[in]  point           Raw unscaled query point.
 * @param[out] score_out       Pointer to receive calibrated anomaly score [0..1).
 * @param[out] metric_depth_out Optional pointer to receive average depth H.
 * @param[out] d_out_out       Optional pointer to receive outer space distance.
 * @return GEIF_OK on success, or error status.
 */
geif_status_t geif_tree_score_point(const geif_forest_t *f,
                                   const double *point,
                                   double *score_out,
                                   double *metric_depth_out,
                                   double *d_out_out);

/**
 * @brief Recursively traverses tree to find the maximum possible leaf height.
 *
 * Used during Zero Kelvin calibration to determine the theoretical deepest path.
 *
 * @param[in]     f        Pointer to the forest.
 * @param[in]     t        Pointer to the tree.
 * @param[in]     node_idx Current node index.
 * @param[in]     depth    Accumulated depth from root.
 * @param[in,out] max_h    Pointer tracking maximum encountered path height.
 */
void geif_tree_find_max_height(const geif_forest_t *f,
                               const geif_tree_t *t,
                               int32_t node_idx,
                               double depth,
                               double *max_h);

/**
 * @brief Calibrates Zero Kelvin universal scale floor and average score.
 *
 * Traverses deepest leaf paths across all trees to calibrate H_max, s_min,
 * and ensemble average inlier baseline.
 *
 * @param f Pointer to the forest instance.
 */
void geif_tree_calibrate(geif_forest_t *f);

#ifdef __cplusplus
}
#endif

#endif /* GEIF_TREE_COMMON_H */
