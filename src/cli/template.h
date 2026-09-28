/**
 * @file template.h
 * @brief Output templating engine for GEIF CLI (-p, -v, -N, -M).
 */

#ifndef GEIF_TEMPLATE_H
#define GEIF_TEMPLATE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char   *orig_line;     /**< %a: Original raw input line */
    const char   *label;         /**< %l: Extracted label string */
    const char   *category;      /**< %c, %C: Extracted category string */
    double        score;         /**< %s: Anomaly score (0.0 .. 1.0) */
    double        metric_depth;  /**< %m: Metric tree depth */
    double        d_out;         /**< %d: Outer stadium distance */
    double        H_max;         /**< %h: Calibrated scale H_max */
    int           is_outlier;    /**< %o: Outlier indicator (0 or 1) */
    time_t        timestamp;     /**< %t: Epoch timestamp */
    const double *vector;        /**< %v: Numeric feature vector */
    uint32_t      vector_dim;    /**< Dimensionality of feature vector */
} geif_template_context_t;

/**
 * @brief Formats a template string by replacing % specifiers with context values.
 *
 * Supported specifiers:
 *   %s - Anomaly score (%.6f)
 *   %S - Anomaly score percentage (%.2f%%)
 *   %l - Label string
 *   %c - Category string
 *   %C - Category string
 *   %m - Metric depth (%.6f)
 *   %d - Outer stadium distance (%.6f)
 *   %h - Forest H_max universal scale (%.6f)
 *   %o - Outlier flag (0 or 1)
 *   %t - Epoch timestamp (%ld)
 *   %a - Entire original input line
 *   %v - Numeric feature vector values
 *   %% - Literal percent sign
 *
 * @param out Output buffer
 * @param out_size Capacity of output buffer
 * @param tmpl Template string
 * @param ctx Context values
 * @return Number of characters written (excluding null terminator)
 */
size_t geif_format_template(char *out,
                            size_t out_size,
                            const char *tmpl,
                            const geif_template_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* GEIF_TEMPLATE_H */
