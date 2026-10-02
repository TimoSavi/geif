/**
 * @file columns.c
 * @brief Column range specification parsing and feature masking for GEIF CLI.
 */

#include "columns.h"
#include "xmalloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

/**
 * @brief Trims leading and trailing ASCII whitespace in-place.
 *
 * @param[in,out] str Input string.
 * @return Pointer to first non-whitespace character in str.
 */
static char *trim_whitespace(char *str)
{
    while (isspace((unsigned char)*str)) str++;
    if (*str == 0) return str;
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return str;
}

/**
 * @brief Parses a comma-separated 1-based column range string into 0-based indices.
 *
 * Parses tokens such as "1,3,5-8" into an array of distinct 0-based indices
 * [0, 2, 4, 5, 6, 7] while discarding duplicates.
 *
 * @param[in]  spec        Column specification string (e.g. "1-4,7").
 * @param[out] indices     Destination array receiving 0-based indices.
 * @param[in]  max_indices Maximum capacity of indices array.
 * @return Number of resolved column indices, or -1 on syntax/range error.
 */
int geif_parse_dim_spec(const char *spec, uint32_t *indices, uint32_t max_indices)
{
    if (!spec || !indices || max_indices == 0) return 0;

    char *copy = xstrdup(spec);
    uint32_t count = 0;
    char *saveptr1 = NULL;
    char *token = strtok_r(copy, ",", &saveptr1);

    while (token) {
        char *trimmed = trim_whitespace(token);
        if (*trimmed != '\0') {
            char *hyphen = strchr(trimmed, '-');
            if (hyphen) {
                *hyphen = '\0';
                char *start_str = trim_whitespace(trimmed);
                char *end_str = trim_whitespace(hyphen + 1);

                int start_val = atoi(start_str);
                int end_val = atoi(end_str);

                if (start_val < 1 || end_val < start_val) {
                    fprintf(stderr, "geif: error: invalid column range '%s-%s' (1-based index required)\n",
                            start_str, end_str);
                    xfree(copy);
                    return -1;
                }

                for (int j = start_val; j <= end_val; j++) {
                    uint32_t idx = (uint32_t)(j - 1);
                    if (!geif_has_col_index(idx, indices, count)) {
                        if (count >= max_indices) {
                            fprintf(stderr, "geif: error: exceeded maximum column limit (%u)\n", max_indices);
                            xfree(copy);
                            return -1;
                        }
                        indices[count++] = idx;
                    }
                }
            } else {
                int val = atoi(trimmed);
                if (val < 1) {
                    fprintf(stderr, "geif: error: invalid column index '%s' (1-based index required)\n", trimmed);
                    xfree(copy);
                    return -1;
                }
                uint32_t idx = (uint32_t)(val - 1);
                if (!geif_has_col_index(idx, indices, count)) {
                    if (count >= max_indices) {
                        fprintf(stderr, "geif: error: exceeded maximum column limit (%u)\n", max_indices);
                        xfree(copy);
                        return -1;
                    }
                    indices[count++] = idx;
                }
            }
        }
        token = strtok_r(NULL, ",", &saveptr1);
    }

    xfree(copy);
    return (int)count;
}

/**
 * @brief Tests whether a given column index exists within an index array.
 *
 * @param[in] col_idx Column index to query.
 * @param[in] list    Array of column indices.
 * @param[in] count   Number of elements in list.
 * @return True if col_idx is present, false otherwise.
 */
bool geif_has_col_index(uint32_t col_idx, const uint32_t *list, uint32_t count)
{
    if (!list) return false;
    for (uint32_t i = 0; i < count; i++) {
        if (list[i] == col_idx) return true;
    }
    return false;
}

/**
 * @brief Initializes a geif_column_config_t structure from CLI argument strings.
 *
 * Copies and parses the raw column specification strings for ignore (-I),
 * include (-U), label (-L), and category (-C).
 *
 * @param[out] cfg           Column configuration object.
 * @param[in]  ignore_spec   Specification string for ignored columns (or NULL).
 * @param[in]  include_spec  Specification string for included columns (or NULL).
 * @param[in]  label_spec    Specification string for label columns (or NULL).
 * @param[in]  category_spec Specification string for category columns (or NULL).
 */
void geif_column_config_init(geif_column_config_t *cfg,
                             const char *ignore_spec,
                             const char *include_spec,
                             const char *label_spec,
                             const char *category_spec)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));

    if (ignore_spec) {
        cfg->ignore_spec = xstrdup(ignore_spec);
        int n = geif_parse_dim_spec(ignore_spec, cfg->ignore_indices, GEIF_MAX_COLUMNS);
        cfg->ignore_count = (n > 0) ? (uint32_t)n : 0;
    }
    if (include_spec) {
        cfg->include_spec = xstrdup(include_spec);
        int n = geif_parse_dim_spec(include_spec, cfg->include_indices, GEIF_MAX_COLUMNS);
        cfg->include_count = (n > 0) ? (uint32_t)n : 0;
    }
    if (label_spec) {
        cfg->label_spec = xstrdup(label_spec);
        int n = geif_parse_dim_spec(label_spec, cfg->label_indices, GEIF_MAX_COLUMNS);
        cfg->label_count = (n > 0) ? (uint32_t)n : 0;
    }
    if (category_spec) {
        cfg->category_spec = xstrdup(category_spec);
        int n = geif_parse_dim_spec(category_spec, cfg->category_indices, GEIF_MAX_COLUMNS);
        cfg->category_count = (n > 0) ? (uint32_t)n : 0;
    }
}

/**
 * @brief Frees dynamically allocated specification strings in column configuration.
 *
 * @param[in,out] cfg Column configuration instance to release.
 */
void geif_column_config_free(geif_column_config_t *cfg)
{
    if (!cfg) return;
    if (cfg->ignore_spec)   { xfree(cfg->ignore_spec);   cfg->ignore_spec = NULL; }
    if (cfg->include_spec)  { xfree(cfg->include_spec);  cfg->include_spec = NULL; }
    if (cfg->label_spec)    { xfree(cfg->label_spec);    cfg->label_spec = NULL; }
    if (cfg->category_spec) { xfree(cfg->category_spec); cfg->category_spec = NULL; }
}

/**
 * @brief Resolves active feature column indices based on total input columns.
 *
 * Determines the set of numerical feature columns by applying precedence:
 *  1. Label columns (-L) are excluded.
 *  2. Category columns (-C) are excluded.
 *  3. Included columns (-U), if specified, must contain the column.
 *  4. Ignored columns (-I) are excluded.
 *
 * @param[in,out] cfg        Column configuration instance.
 * @param[in]     total_cols Total number of fields detected in CSV header/row.
 * @return True if at least one feature column was resolved, false otherwise.
 */
bool geif_column_config_resolve(geif_column_config_t *cfg, uint32_t total_cols)
{
    if (!cfg || total_cols == 0) return false;

    cfg->total_input_cols = total_cols;
    cfg->feature_dim_count = 0;

    for (uint32_t i = 0; i < total_cols; i++) {
        /* 1. Label columns are never feature dimensions */
        if (geif_has_col_index(i, cfg->label_indices, cfg->label_count)) {
            continue;
        }

        /* 2. Category columns are never feature dimensions */
        if (geif_has_col_index(i, cfg->category_indices, cfg->category_count)) {
            continue;
        }

        /* 3. If explicit include list (-U) provided, column must be in it */
        if (cfg->include_count > 0 && !geif_has_col_index(i, cfg->include_indices, cfg->include_count)) {
            continue;
        }

        /* 4. Ignored columns (-I) are excluded */
        if (geif_has_col_index(i, cfg->ignore_indices, cfg->ignore_count)) {
            continue;
        }

        /* Active feature dimension */
        cfg->feature_cols[cfg->feature_dim_count++] = i;
    }

    return (cfg->feature_dim_count > 0);
}

/**
 * @brief Extracts continuous numerical feature values from parsed text tokens into a double vector.
 *
 * Iterates through active feature column indices, parsing each field with strtod().
 * Replaces non-numeric tokens, NaNs, and infinities with 0.0.
 *
 * @param[in]  cfg        Resolved column configuration.
 * @param[in]  tokens     Array of string pointers representing fields in current row.
 * @param[in]  total_cols Number of tokens available in row.
 * @param[out] vec        Destination double array (size >= cfg->feature_dim_count).
 * @return True on success, false if input arguments are invalid.
 */
bool geif_extract_features(const geif_column_config_t *cfg,
                           char **tokens,
                           uint32_t total_cols,
                           double *vec)
{
    if (!cfg || !tokens || !vec) return false;
    if (total_cols < cfg->total_input_cols && cfg->total_input_cols > 0) return false;

    for (uint32_t d = 0; d < cfg->feature_dim_count; d++) {
        uint32_t col_idx = cfg->feature_cols[d];
        if (col_idx >= total_cols || !tokens[col_idx] || tokens[col_idx][0] == '\0') {
            vec[d] = 0.0;
            continue;
        }

        char *endptr;
        double val = strtod(tokens[col_idx], &endptr);
        if (isnan(val) || isinf(val) || endptr == tokens[col_idx]) {
            vec[d] = 0.0;
        } else {
            vec[d] = val;
        }
    }
    return true;
}

/**
 * @brief Concatenates values of configured label columns (-L) into an output string.
 *
 * Joins label fields using the specified separator character.
 *
 * @param[in]  cfg        Column configuration instance.
 * @param[in]  tokens     Parsed string tokens from row.
 * @param[in]  total_cols Total tokens available in row.
 * @param[in]  sep        Delimiter character (e.g. '/' or '_').
 * @param[out] out_buf    Destination buffer for composite label string.
 * @param[in]  max_len    Capacity of destination buffer.
 */
void geif_extract_label(const geif_column_config_t *cfg,
                        char **tokens,
                        uint32_t total_cols,
                        char sep,
                        char *out_buf,
                        size_t max_len)
{
    if (!out_buf || max_len == 0) return;
    out_buf[0] = '\0';

    if (!cfg || cfg->label_count == 0 || !tokens) return;

    size_t written = 0;
    for (uint32_t i = 0; i < cfg->label_count; i++) {
        uint32_t col = cfg->label_indices[i];
        const char *val = (col < total_cols && tokens[col]) ? tokens[col] : "";

        if (i > 0 && written + 1 < max_len) {
            out_buf[written++] = sep;
            out_buf[written] = '\0';
        }

        size_t vlen = strlen(val);
        if (written + vlen < max_len) {
            memcpy(out_buf + written, val, vlen);
            written += vlen;
            out_buf[written] = '\0';
        } else {
            size_t rem = max_len - written - 1;
            memcpy(out_buf + written, val, rem);
            written += rem;
            out_buf[written] = '\0';
            break;
        }
    }
}

/**
 * @brief Concatenates values of configured category columns (-C) into a routing key.
 *
 * Joins category fields using the specified separator character to form
 * the lookup key for ensemble sub-forest dispatching.
 *
 * @param[in]  cfg        Column configuration instance.
 * @param[in]  tokens     Parsed string tokens from row.
 * @param[in]  total_cols Total tokens available in row.
 * @param[in]  sep        Delimiter character.
 * @param[out] out_buf    Destination buffer for composite category string.
 * @param[in]  max_len    Capacity of destination buffer.
 */
void geif_extract_category(const geif_column_config_t *cfg,
                            char **tokens,
                            uint32_t total_cols,
                            char sep,
                            char *out_buf,
                            size_t max_len)
{
    if (!out_buf || max_len == 0) return;
    out_buf[0] = '\0';

    if (!cfg || cfg->category_count == 0 || !tokens) return;

    size_t written = 0;
    for (uint32_t i = 0; i < cfg->category_count; i++) {
        uint32_t col = cfg->category_indices[i];
        const char *val = (col < total_cols && tokens[col]) ? tokens[col] : "";

        if (i > 0 && written + 1 < max_len) {
            out_buf[written++] = sep;
            out_buf[written] = '\0';
        }

        size_t vlen = strlen(val);
        if (written + vlen < max_len) {
            memcpy(out_buf + written, val, vlen);
            written += vlen;
            out_buf[written] = '\0';
        } else {
            size_t rem = max_len - written - 1;
            memcpy(out_buf + written, val, rem);
            written += rem;
            out_buf[written] = '\0';
            break;
        }
    }
}
