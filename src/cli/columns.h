/**
 * @file columns.h
 * @brief Column range specification parsing and feature masking for GEIF CLI.
 */

#ifndef GEIF_COLUMNS_H
#define GEIF_COLUMNS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define GEIF_MAX_COLUMNS 1024

typedef struct {
    char *ignore_spec;       /**< Raw string passed to -I (e.g. "1,4-6") */
    char *include_spec;      /**< Raw string passed to -U (e.g. "2-10") */
    char *label_spec;        /**< Raw string passed to -L (e.g. "1") */
    char *category_spec;     /**< Raw string passed to -C (e.g. "12") */

    uint32_t ignore_indices[GEIF_MAX_COLUMNS];
    uint32_t ignore_count;

    uint32_t include_indices[GEIF_MAX_COLUMNS];
    uint32_t include_count;

    uint32_t label_indices[GEIF_MAX_COLUMNS];
    uint32_t label_count;

    uint32_t category_indices[GEIF_MAX_COLUMNS];
    uint32_t category_count;

    /* Computed mapping for a given input row of N columns */
    uint32_t total_input_cols;
    uint32_t feature_dim_count;
    uint32_t feature_cols[GEIF_MAX_COLUMNS]; /**< Mapping: feature_idx (0..D-1) -> input_col_idx */
} geif_column_config_t;

/**
 * @brief Parse a 1-based comma/hyphen range string into 0-based indices.
 * Example: "1,3,5-7" -> [0, 2, 4, 5, 6], count = 5.
 * @return Number of unique parsed indices, or -1 on error.
 */
int geif_parse_dim_spec(const char *spec, uint32_t *indices, uint32_t max_indices);

/**
 * @brief Initialize column configuration from CLI option strings.
 */
void geif_column_config_init(geif_column_config_t *cfg,
                             const char *ignore_spec,
                             const char *include_spec,
                             const char *label_spec,
                             const char *category_spec);

/**
 * @brief Free any heap-allocated strings in column configuration.
 */
void geif_column_config_free(geif_column_config_t *cfg);

/**
 * @brief Compute the feature column mapping for a row with total_cols columns.
 * Excludes label, category, ignored, and non-included columns.
 * Sets cfg->feature_dim_count and cfg->feature_cols[].
 */
bool geif_column_config_resolve(geif_column_config_t *cfg, uint32_t total_cols);

/**
 * @brief Check if a 0-based column index is present in an index list.
 */
bool geif_has_col_index(uint32_t col_idx, const uint32_t *list, uint32_t count);

/**
 * @brief Extract feature vector from row tokens according to column config.
 * Handles NaNs, infinities, and missing fields by defaulting to 0.0.
 */
bool geif_extract_features(const geif_column_config_t *cfg,
                           char **tokens,
                           uint32_t total_cols,
                           double *vec);

/**
 * @brief Build label string from row tokens according to cfg->label_indices.
 * Concatenates label tokens with sep. Buffer out_buf is populated.
 */
void geif_extract_label(const geif_column_config_t *cfg,
                        char **tokens,
                        uint32_t total_cols,
                        char sep,
                        char *out_buf,
                        size_t max_len);

#endif /* GEIF_COLUMNS_H */
