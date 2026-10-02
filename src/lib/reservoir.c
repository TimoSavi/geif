/**
 * @file reservoir.c
 * @brief Streaming reservoir sampling with adaptive ceiling factor.
 */

#include "geif/geif.h"
#include <stdlib.h>
#include <string.h>

/**
 * @brief Ingests an observation vector into the forest's streaming reservoir sample pool.
 *
 * Implements Algorithm R reservoir sampling with an optional ceiling factor cap:
 * 1. Increment total_rows_seen and expands the coordinate bounding box envelope.
 * 2. If pool_count < pool_capacity (Phase 1), point is stored directly in the pool.
 * 3. Once capacity is reached (Phase 2), uses a 64-bit pseudo-random replacement
 *    index modulo effective_seen (bounded by ceiling_factor * capacity to prevent
 *    sample freezing over very long streams). If slot falls within capacity, the
 *    existing sample in that slot is replaced.
 *
 * @param[in,out] f     Forest instance.
 * @param[in]     point Array of double coordinates of length f->dimensions.
 * @return GEIF_OK on success, or GEIF_ERR_INVALID_ARG if arguments are NULL.
 */
geif_status_t geif_forest_feed(geif_forest_t *f, const double *point)
{
    if (!f || !point) {
        return GEIF_ERR_INVALID_ARG;
    }

    uint32_t d = f->dimensions;
    f->total_rows_seen++;

    // Update bounding box coordinates
    for (uint32_t j = 0; j < d; j++) {
        if (point[j] < f->envelope_min[j]) f->envelope_min[j] = point[j];
        if (point[j] > f->envelope_max[j]) f->envelope_max[j] = point[j];
    }

    // Phase 1: Reservoir not yet full
    if (f->pool_count < f->pool_capacity) {
        size_t offset = f->pool_count * d;
        memcpy(&f->sample_pool[offset], point, d * sizeof(double));
        f->pool_count++;
        return GEIF_OK;
    }

    // Phase 2: Reservoir full - perform bounded reservoir replacement
    uint64_t max_effective = (uint64_t)(f->config.ceiling_factor + 1) * f->pool_capacity;
    uint64_t effective_seen = f->total_rows_seen;
    if (effective_seen > max_effective) {
        effective_seen = max_effective;
    }

    // 64-bit random number for large stream row counts
    uint64_t rand64 = (((uint64_t)rand() << 32) ^ (uint64_t)rand()) & 0x7FFFFFFFFFFFFFFFULL;
    uint64_t slot = rand64 % effective_seen;

    if (slot < f->pool_capacity) {
        size_t offset = slot * d;
        memcpy(&f->sample_pool[offset], point, d * sizeof(double));
    }

    return GEIF_OK;
}
