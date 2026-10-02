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
    const char   *orig_line;         /**< %a / %v fallback: Original raw input line */
    const char   *label;             /**< %l: Extracted label string */
    const char   *category;          /**< %C: Model / assigned category string */
    const char   *input_category;    /**< %c: Category extracted from input data */
    double        score;             /**< %s: Anomaly score (0.0 .. 1.0) */
    double        metric_depth;      /**< %m: Metric tree depth (when -j not provided) */
    double        d_out;             /**< %d (when vector is NULL): Outer stadium distance */
    double        H_max;             /**< %h: Calibrated scale H_max */
    int           is_outlier;        /**< %o: Outlier indicator (0 or 1) */
    time_t        timestamp;         /**< %t: Epoch timestamp */
    uint64_t      total_rows;        /**< %n: Category total rows */
    uint64_t      analyzed_rows;     /**< %o: Analyzed rows count */
    uint64_t      row_idx;           /**< %r: 1-based row number */
    const double *vector;            /**< %d (in -p) / %d (in -j): Numeric feature vector */
    const double *averages;          /**< %a: Category feature averages */
    const double *attr_scores;       /**< %e: Single-dimension attribution / impact scores */
    uint32_t      vector_dim;        /**< Dimensionality of feature vector */
    char        **raw_values;        /**< %v: Raw token string array */
    uint32_t      raw_value_count;   /**< Total raw tokens in raw_values */
    char          list_separator;    /**< Delimiter between dimensions / values (-e) */
    int           decimals;          /**< Decimal precision (-d, default 6) */
    const char   *printf_format;     /**< Custom printf format (from -m "%...") */
    const char   *print_dimension;   /**< Custom template for %m dimension iteration (-j) */
    uint32_t      low_rgb;           /**< Hex color for score 0.0 (default 0x20FF20 or 0xFFFF00) */
    uint32_t      high_rgb;          /**< Hex color for score 1.0 (default 0xFF0000) */
} geif_template_context_t;

/**
 * @brief Formats a template string by replacing % specifiers with context values.
 *
 * Supported specifiers:
 *   %s - Anomaly score (formatted with precision -d)
 *   %S - Anomaly score percentage (%.2f%%)
 *   %l - Label string
 *   %c - Category string
 *   %C - Category string
 *   %m - Dimension expansion with -j template, or continuous metric depth H
 *   %d - Feature dimension values list (joined by -e), or outer stadium distance
 *   %e - Single-dimension impact / attribution scores list (joined by -e)
 *   %a - Category dimension averages list (joined by -e), or raw input line
 *   %v - Raw token column values (joined by -e), or raw input line
 *   %x - 6-hex RGB color interpolated by score between low_rgb and high_rgb
 *   %h - Forest H_max universal scale
 *   %o - Outlier flag (0 or 1)
 *   %n - Category total training rows
 *   %t - Epoch timestamp (%ld)
 *   %% - Literal percent sign
 *
 * Inside dimension template (-j, expanded by %m):
 *   %d - Feature value for current dimension (formatted with -m printf_format or -d)
 *   %a - Category average value for current dimension
 *   %e - Single-dimension impact / attribution score for current dimension
 *   %i - 1-based feature dimension index
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
