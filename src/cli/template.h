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
    double        score;             /**< %s: Anomaly score (0.0 .. 1.0) or threshold in -v */
    double        test_average_score;/**< %S: Average anomaly score for analyzed data */
    double        metric_depth;      /**< %m: Metric tree depth (internal) */
    double        d_out;             /**< %d (when vector is NULL): Outer stadium distance */
    double        H_max;             /**< %h: Calibrated scale H_max */
    int           is_outlier;        /**< %o: Outlier indicator (0 or 1 in row mode) */
    time_t        timestamp;         /**< %t: Human-readable timestamp, %u / %T: Epoch timestamp */
    uint64_t      total_rows;        /**< %n: Category total rows */
    uint64_t      analyzed_rows;     /**< %o: Analyzed rows count */
    uint64_t      high_analyzed_rows;/**< %h: High analyzed rows count (score > threshold) */
    uint64_t      row_idx;           /**< %r: 1-based row number */
    const double *vector;            /**< %d (in -p) / %d (in -j): Numeric feature vector */
    const double *averages;          /**< %a: Category feature averages */
    const double *attr_scores;       /**< %e: Single-dimension attribution / impact scores */
    uint32_t      vector_dim;        /**< Dimensionality of feature vector */
    char        **raw_values;        /**< %v: Raw token string array */
    uint32_t      raw_value_count;   /**< Total raw tokens in raw_values */
    char          list_separator;    /**< Delimiter between dimensions / values (-e) */
    char          category_sep;      /**< %: Category separator */
    char          label_sep;         /**< %. Label separator */
    int           decimals;          /**< Decimal precision (-d, default 6) */
    const char   *printf_format;     /**< Custom printf format (from -m "%...") */
    const char   *print_dimension;   /**< Custom template for %m dimension iteration (-j) */
    uint32_t      low_rgb;           /**< Hex color for score 0.0 (default 0x20FF20 or 0xFFFF00) */
    uint32_t      high_rgb;          /**< Hex color for score 1.0 (default 0xFF0000) */
    bool          is_summary;        /**< Summary mode flag (-v): switches %o, %h, %S to category aggregates */
} geif_template_context_t;

/**
 * @brief Formats a template string by replacing % specifiers with context values.
 *
 * Supported specifiers (matching CEIF):
 *   %r - Current input file row number
 *   %s - Anomaly score (or outlier score threshold in -v summary)
 *   %S - Average anomaly score for analyzed data
 *   %l - Label string
 *   %c - Category string from input data
 *   %C - Forest / model category string
 *   %m - Separated list of dimension metrics (-j template)
 *   %d - Feature dimension values list (joined by -e), or outer stadium distance
 *   %e - Single-dimension impact / attribution scores list (joined by -e)
 *   %a - Separated list of dimension average values (joined by -e)
 *   %v - Raw token column values (joined by -e), or raw input line
 *   %x - 6-hex RGB color interpolated by score between low_rgb and high_rgb
 *   %h - Number of analyzed rows with score > threshold (or H_max scale)
 *   %o - Number of analyzed rows for category in summary (-v) or outlier flag in row mode
 *   %n - Category total training rows
 *   %t - Timestamp when category was last updated, human-readable (strftime "%c")
 *   %u - Unix epoch timestamp in seconds (%ld)
 *   %T - Unix epoch timestamp in seconds (%ld, alias of %u)
 *   %: - Category value separator
 *   %. - Label value separator
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

/**
 * @brief Checks if a template string (or its %m dimension sub-template) uses a given specifier.
 */
bool geif_template_uses_spec(const char *tmpl,
                             const char *print_dimension,
                             char spec);

#ifdef __cplusplus
}
#endif

#endif /* GEIF_TEMPLATE_H */
