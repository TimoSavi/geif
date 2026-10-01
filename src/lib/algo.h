/**
 * @file algo.h
 * @brief Internal algorithm abstraction interface and registry for GEIF.
 */

#ifndef GEIF_ALGO_H
#define GEIF_ALGO_H

#include "geif/geif.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Unified operations interface for geometric anomaly detection algorithms.
 */
typedef struct geif_algo_ops {
    geif_algo_type_t type;
    const char      *name;
    const char      *description;
    geif_status_t  (*train)(geif_forest_t *f);
    geif_status_t  (*score)(const geif_forest_t *f,
                            const double *point,
                            double *score_out,
                            double *metric_depth_out,
                            double *d_out_out);
    void           (*destroy)(geif_forest_t *f);
} geif_algo_ops_t;

/**
 * @brief Retrieves the operations table for the specified algorithm type.
 */
const geif_algo_ops_t *geif_algo_get_ops(geif_algo_type_t algo);

/* Individual algorithm operational tables */
extern const geif_algo_ops_t geif_algo_ops_ceif;
extern const geif_algo_ops_t geif_algo_ops_bubble;
extern const geif_algo_ops_t geif_algo_ops_exemplar;
extern const geif_algo_ops_t geif_algo_ops_voronoi;

#ifdef __cplusplus
}
#endif

#endif /* GEIF_ALGO_H */
