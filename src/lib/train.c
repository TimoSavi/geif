/**
 * @file train.c
 * @brief Unified training dispatcher for GEIF algorithms.
 */

#include "geif/geif.h"
#include "algo.h"
#include <stdlib.h>

/**
 * @brief Trains all trees in the forest using the algorithm configured in forest config.
 *
 * Validates forest presence and ensures the reservoir pool is non-empty. Dispatches
 * training via the polymorphic algorithm operations table (geif_algo_ops_t) matching
 * f->config.algo (e.g., bubble, voronoi, exemplar, ceif).
 *
 * @param[in,out] f Forest instance containing populated reservoir samples.
 * @return GEIF_OK on success, GEIF_ERR_EMPTY_DATASET if no samples, or an error status code.
 */
geif_status_t geif_forest_train(geif_forest_t *f)
{
    if (!f) {
        return GEIF_ERR_INVALID_ARG;
    }
    if (f->is_trained) {
        return GEIF_OK;
    }
    if (f->pool_count == 0) {
        return GEIF_ERR_EMPTY_DATASET;
    }

    if (f->config.seed != 0) {
        srand(f->config.seed);
    }

    const geif_algo_ops_t *ops = geif_algo_get_ops(f->config.algo);
    if (!ops || !ops->train) {
        return GEIF_ERR_NOT_SUPPORTED;
    }

    geif_status_t status = ops->train(f);
    if (status == GEIF_OK) {
        f->is_trained = true;
    }
    return status;
}
