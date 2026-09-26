/**
 * @file geif.h
 * @brief Public C17 API for Geometric Extended Isolation Forest (GEIF).
 */

#ifndef GEIF_H
#define GEIF_H

#include "geif/types.h"
#include "geif/error.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns the default configuration for GEIF.
 */
geif_config_t geif_config_default(void);

/**
 * @brief Allocates and initializes a new GEIF forest.
 *
 * @param[out] forest_out Pointer to receive the allocated forest.
 * @param[in]  dimensions Dimensionality of the input features (D > 0).
 * @param[in]  config     Optional configuration (NULL for defaults).
 * @return GEIF_OK on success, or an error code.
 */
geif_status_t geif_forest_create(geif_forest_t **forest_out,
                                uint32_t dimensions,
                                const geif_config_t *config);

/**
 * @brief Frees all resources associated with a forest.
 *
 * @param forest Forest to free (safe to call with NULL).
 */
void geif_forest_destroy(geif_forest_t *forest);

/**
 * @brief Feeds a single observation vector into the forest's reservoir sample pool.
 *
 * @param forest Forest instance.
 * @param point  Feature vector of length dimensions.
 * @return GEIF_OK on success, or an error code.
 */
geif_status_t geif_forest_feed(geif_forest_t *forest, const double *point);

/**
 * @brief Trains all trees in the forest using the collected sample pool.
 *
 * @param forest Forest instance with collected samples.
 * @return GEIF_OK on success, or an error code.
 */
geif_status_t geif_forest_train(geif_forest_t *forest);

/**
 * @brief Evaluates raw unnormalized continuous metric depth H(x) for a point.
 *
 * @param f         Forest instance.
 * @param point     Observation vector of length dimensions.
 * @param d_out_out Optional pointer to receive outer stadium distance.
 * @return Accumulated continuous metric depth.
 */
double geif_forest_evaluate_metric_depth(const geif_forest_t *f,
                                         const double *point,
                                         double *d_out_out);

/**
 * @brief Evaluates an observation and computes its normalized anomaly score in [0.0, 1.0].
 *
 * Score semantics:
 *   - 0.000000: Asymptotic absolute inlier ("Zero Kelvin")
 *   - 0.10 - 0.35: Strong nominal cluster inlier
 *   - 0.50: Boundary threshold
 *   - 0.85 - 1.000000: Outlier / Anomaly / Outer Space
 *
 * @param[in]  forest    Trained forest instance.
 * @param[in]  point     Query vector of length dimensions.
 * @param[out] score_out Pointer to receive the computed anomaly score.
 * @return GEIF_OK on success, or an error code.
 */
geif_status_t geif_forest_score(const geif_forest_t *forest,
                               const double *point,
                               double *score_out);

/**
 * @brief Evaluates an observation with detailed diagnostic metric outputs.
 *
 * @param[in]  forest           Trained forest instance.
 * @param[in]  point            Query vector of length dimensions.
 * @param[out] score_out        Final anomaly score in [0.0, 1.0].
 * @param[out] metric_depth_out Accumulated continuous metric depth H.
 * @param[out] d_out_out        Relative distance outside the envelope stadium.
 * @return GEIF_OK on success, or an error code.
 */
geif_status_t geif_forest_score_detailed(const geif_forest_t *forest,
                                        const double *point,
                                        double *score_out,
                                        double *metric_depth_out,
                                        double *d_out_out);

/**
 * @brief Serializes a trained GEIF forest to a JSON model file.
 *
 * @param forest Forest to save.
 * @param path   Target filesystem path.
 * @return GEIF_OK on success, or an error code.
 */
geif_status_t geif_forest_save_json(const geif_forest_t *forest, const char *path);

/**
 * @brief Deserializes a GEIF forest from a JSON model file.
 *
 * @param[out] forest_out Pointer to receive the loaded forest.
 * @param[in]  path       Source filesystem path.
 * @return GEIF_OK on success, or an error code.
 */
geif_status_t geif_forest_load_json(geif_forest_t **forest_out, const char *path);

/**
 * @brief Formats a human-readable diagnostic summary of the forest.
 *
 * @param forest      Forest instance.
 * @param buffer      Destination text buffer.
 * @param buffer_size Size of destination buffer.
 */
void geif_forest_summary(const geif_forest_t *forest, char *buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif /* GEIF_H */
