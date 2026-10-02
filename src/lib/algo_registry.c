/**
 * @file algo_registry.c
 * @brief Algorithm registry and dispatching table for GEIF.
 */

#include "algo.h"
#include <string.h>
#include <strings.h>

/**
 * @brief Returns the canonical string identifier for an algorithm type.
 *
 * @param[in] algo Algorithm enum identifier (e.g. GEIF_ALGO_BUBBLE).
 * @return Constant string: "ceif", "bubble", "exemplar", or "voronoi".
 */
const char *geif_algo_name(geif_algo_type_t algo)
{
    switch (algo) {
    case GEIF_ALGO_CEIF:
        return "ceif";
    case GEIF_ALGO_BUBBLE:
        return "bubble";
    case GEIF_ALGO_EXEMPLAR:
        return "exemplar";
    case GEIF_ALGO_VORONOI:
        return "voronoi";
    default:
        return "ceif";
    }
}

/**
 * @brief Parses a string algorithm name or alias into a geif_algo_type_t enum.
 *
 * Supports canonical names and aliases (case-insensitive):
 *  - "bubble", "spherical" -> GEIF_ALGO_BUBBLE
 *  - "voronoi"             -> GEIF_ALGO_VORONOI
 *  - "exemplar", "knn", "density" -> GEIF_ALGO_EXEMPLAR
 *  - "ceif", "eif", "gaussian"    -> GEIF_ALGO_CEIF
 *
 * @param[in] name Algorithm string identifier.
 * @return Parsed geif_algo_type_t enum (defaults to GEIF_ALGO_DEFAULT if unknown).
 */
geif_algo_type_t geif_algo_from_name(const char *name)
{
    if (!name || name[0] == '\0') {
        return GEIF_ALGO_DEFAULT;
    }

    if (strcasecmp(name, "ceif") == 0 ||
        strcasecmp(name, "eif") == 0 ||
        strcasecmp(name, "gaussian") == 0) {
        return GEIF_ALGO_CEIF;
    }
    if (strcasecmp(name, "bubble") == 0 ||
        strcasecmp(name, "spherical") == 0) {
        return GEIF_ALGO_BUBBLE;
    }
    if (strcasecmp(name, "exemplar") == 0 ||
        strcasecmp(name, "knn") == 0 ||
        strcasecmp(name, "density") == 0) {
        return GEIF_ALGO_EXEMPLAR;
    }
    if (strcasecmp(name, "voronoi") == 0) {
        return GEIF_ALGO_VORONOI;
    }

    return GEIF_ALGO_DEFAULT;
}

/**
 * @brief Retrieves the polymorphic operations dispatch table for an algorithm.
 *
 * Returns pointer to the static geif_algo_ops_t table containing function pointers
 * for train, score, serialize, deserialize, and destroy.
 *
 * @param[in] algo Algorithm type enum.
 * @return Pointer to algorithm operations table.
 */
const geif_algo_ops_t *geif_algo_get_ops(geif_algo_type_t algo)
{
    switch (algo) {
    case GEIF_ALGO_CEIF:
        return &geif_algo_ops_ceif;
    case GEIF_ALGO_BUBBLE:
        return &geif_algo_ops_bubble;
    case GEIF_ALGO_EXEMPLAR:
        return &geif_algo_ops_exemplar;
    case GEIF_ALGO_VORONOI:
        return &geif_algo_ops_voronoi;
    default:
        return &geif_algo_ops_ceif;
    }
}
