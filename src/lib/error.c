/**
 * @file error.c
 * @brief Error code string formatting.
 */

#include "geif/error.h"

/**
 * @brief Translates a geif_status_t numeric error code into a human-readable English description.
 *
 * @param[in] status Status code to convert.
 * @return Static string pointer containing diagnostic message.
 */
const char *geif_status_str(geif_status_t status)
{
    switch (status) {
    case GEIF_OK:
        return "Success";
    case GEIF_ERR_INVALID_ARG:
        return "Invalid argument or null pointer";
    case GEIF_ERR_OUT_OF_MEMORY:
        return "Out of memory";
    case GEIF_ERR_IO:
        return "File I/O error";
    case GEIF_ERR_FORMAT_CORRUPT:
        return "Model file or stream format corrupted";
    case GEIF_ERR_EMPTY_DATASET:
        return "Insufficient samples in dataset";
    case GEIF_ERR_DIM_MISMATCH:
        return "Feature dimension count mismatch";
    case GEIF_ERR_DEGENERATE_DATA:
        return "All dataset points are degenerate or identical";
    case GEIF_ERR_NOT_SUPPORTED:
        return "Requested algorithm or feature is not supported";
    default:
        return "Unknown error";
    }
}
