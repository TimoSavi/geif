/**
 * @file algo_registry.c
 * @brief Algorithm registry and dispatching table for GEIF.
 */

#include "algo.h"
#include <string.h>
#include <strings.h>

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
