/**
 * @file error.h
 * @brief Diagnostic error codes and status indicators for GEIF.
 */

#ifndef GEIF_ERROR_H
#define GEIF_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status and error codes returned by GEIF functions.
 */
typedef enum geif_status {
    GEIF_OK                  =  0,  /**< Operation completed successfully */
    GEIF_ERR_INVALID_ARG     = -1,  /**< Invalid argument or null pointer */
    GEIF_ERR_OUT_OF_MEMORY   = -2,  /**< Memory allocation failure */
    GEIF_ERR_IO              = -3,  /**< File I/O read/write error */
    GEIF_ERR_FORMAT_CORRUPT  = -4,  /**< Input format or model file corrupted */
    GEIF_ERR_EMPTY_DATASET   = -5,  /**< Dataset is empty or insufficient samples */
    GEIF_ERR_DIM_MISMATCH    = -6,  /**< Feature dimension count mismatch */
    GEIF_ERR_DEGENERATE_DATA = -7,  /**< All data points are completely degenerate */
    GEIF_ERR_NOT_SUPPORTED   = -8   /**< Requested algorithm or feature is not supported */
} geif_status_t;

/**
 * @brief Returns a static human-readable description of a status code.
 *
 * @param status The status code.
 * @return A constant string describing the status.
 */
const char *geif_status_str(geif_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* GEIF_ERROR_H */
