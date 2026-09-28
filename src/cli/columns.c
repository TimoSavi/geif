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

static char *trim_whitespace(char *str)
{
    while (isspace((unsigned char)*str)) str++;
    if (*str == 0) return str;
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return str;
}

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

bool geif_has_col_index(uint32_t col_idx, const uint32_t *list, uint32_t count)
{
    if (!list) return false;
    for (uint32_t i = 0; i < count; i++) {
        if (list[i] == col_idx) return true;
    }
    return false;
}

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

void geif_column_config_free(geif_column_config_t *cfg)
{
    if (!cfg) return;
    if (cfg->ignore_spec)   { xfree(cfg->ignore_spec);   cfg->ignore_spec = NULL; }
    if (cfg->include_spec)  { xfree(cfg->include_spec);  cfg->include_spec = NULL; }
    if (cfg->label_spec)    { xfree(cfg->label_spec);    cfg->label_spec = NULL; }
    if (cfg->category_spec) { xfree(cfg->category_spec); cfg->category_spec = NULL; }
}

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
