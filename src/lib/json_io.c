/**
 * @file json_io.c
 * @brief JSON serialization and deserialization for GEIF models and ensembles.
 */

#include "geif/geif.h"
#include "tree_common.h"
#include <json-c/json.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Formats a floating-point value to JSON with trimmed trailing zeros.
 *
 * Avoids precision bloat in saved JSON files by formatting with specified decimals
 * and removing trailing decimal zeros and negative zeros ("-0").
 *
 * @param[in] val      Floating point value.
 * @param[in] decimals Maximum decimal precision.
 * @return Allocated json_object representing the double value.
 */
static struct json_object *geif_clean_double_json(double val, int decimals)
{
    char buf[64];
    int dec = (decimals >= 0) ? decimals : 6;
    snprintf(buf, sizeof(buf), "%.*f", dec, val);
    if (strchr(buf, '.')) {
        char *p = buf + strlen(buf) - 1;
        while (p > buf && *p == '0') {
            *p-- = '\0';
        }
        if (*p == '.') *p = '\0';
    }
    if (strcmp(buf, "-0") == 0) {
        strcpy(buf, "0");
    }
    return json_object_new_double_s(val, buf);
}

/**
 * @brief Serializes a single category sub-forest entry into a minimal json_object.
 *
 * Emits only per-forest non-derived fields: category, last_updated, total_rows,
 * and sample_pool. Global parameters are stored once at the ensemble root level.
 *
 * @param[in] f            Forest instance to serialize.
 * @param[in] category     Category key string.
 * @param[in] last_updated Last update timestamp.
 * @param[in] total_rows   Total rows observed for this category.
 * @param[in] decimals     Decimal formatting precision.
 * @return Pointer to newly allocated json_object, or NULL on error.
 */
static struct json_object *geif_forest_entry_to_json_object(const geif_forest_t *f,
                                                            const char *category,
                                                            time_t last_updated,
                                                            uint64_t total_rows,
                                                            int decimals)
{
    if (!f) return NULL;

    struct json_object *jf = json_object_new_object();
    if (!jf) return NULL;

    int dec = (decimals >= 0) ? decimals : ((f->decimals >= 0) ? f->decimals : 6);
    uint64_t rows = (total_rows > 0) ? total_rows : f->total_rows_seen;
    if (rows < f->pool_count) rows = f->pool_count;

    json_object_object_add(jf, "category", json_object_new_string(category ? category : f->category));
    json_object_object_add(jf, "last_updated", json_object_new_int64((int64_t)last_updated));
    json_object_object_add(jf, "total_rows", json_object_new_int64((int64_t)rows));

    struct json_object *j_pool = json_object_new_array();
    size_t total_pool_coords = f->pool_count * f->dimensions;
    for (size_t i = 0; i < total_pool_coords; i++) {
        json_object_array_add(j_pool, geif_clean_double_json(f->sample_pool[i], dec));
    }
    json_object_object_add(jf, "sample_pool", j_pool);

    return jf;
}

/**
 * @brief Deserializes a json-c json_object into a populated geif_forest_t.
 *
 * Inherits ensemble-level dimensions and configuration, with optional per-object
 * fallback for backward compatibility with legacy single-forest JSON files.
 * Unknown JSON fields are silently ignored.
 *
 * @param[out] forest_out   Pointer receiving the newly allocated forest.
 * @param[in]  root         json-c object representing the sub-forest (or root).
 * @param[in]  default_dims Default feature dimensions from parent ensemble.
 * @param[in]  default_cfg  Default configuration from parent ensemble.
 * @return GEIF_OK on success, or GEIF_ERR_FORMAT_CORRUPT / GEIF_ERR_OUT_OF_MEMORY.
 */
static geif_status_t geif_forest_from_json_object(geif_forest_t **forest_out,
                                                  struct json_object *root,
                                                  uint32_t default_dims,
                                                  const geif_config_t *default_cfg)
{
    if (!forest_out || !root) return GEIF_ERR_INVALID_ARG;

    struct json_object *j_arr = NULL;
    if (!json_object_object_get_ex(root, "sample_pool", &j_arr)) {
        return GEIF_ERR_FORMAT_CORRUPT;
    }

    struct json_object *j_val;
    uint32_t dimensions = default_dims;
    if (json_object_object_get_ex(root, "dimensions", &j_val)) {
        dimensions = (uint32_t)json_object_get_int(j_val);
    }
    if (dimensions == 0) {
        return GEIF_ERR_FORMAT_CORRUPT;
    }

    geif_config_t cfg = default_cfg ? *default_cfg : geif_config_default();
    if (json_object_object_get_ex(root, "algorithm", &j_val)) {
        cfg.algo = geif_algo_from_name(json_object_get_string(j_val));
    }
    if (json_object_object_get_ex(root, "tree_count", &j_val)) {
        cfg.tree_count = (uint32_t)json_object_get_int(j_val);
    }
    if (json_object_object_get_ex(root, "samples_per_tree", &j_val)) {
        cfg.samples_per_tree = (uint32_t)json_object_get_int(j_val);
    }

    geif_forest_t *f = NULL;
    geif_status_t status = geif_forest_create(&f, dimensions, &cfg);
    if (status != GEIF_OK) {
        return status;
    }

    if (json_object_object_get_ex(root, "category", &j_val)) {
        const char *cat = json_object_get_string(j_val);
        if (cat) {
            strncpy(f->category, cat, sizeof(f->category) - 1);
            f->category[sizeof(f->category) - 1] = '\0';
        }
    }

    // Derive pool_count directly from sample_pool length / dimensions
    size_t len = (size_t)json_object_array_length(j_arr);
    size_t saved_count = len / dimensions;
    if (json_object_object_get_ex(root, "pool_count", &j_val)) {
        size_t pc = (size_t)json_object_get_int(j_val);
        if (pc < saved_count) saved_count = pc;
    }
    if (saved_count > f->pool_capacity) {
        double *new_pool = (double *)realloc(f->sample_pool, saved_count * dimensions * sizeof(double));
        if (!new_pool) {
            geif_forest_destroy(f);
            return GEIF_ERR_OUT_OF_MEMORY;
        }
        f->sample_pool = new_pool;
        f->pool_capacity = saved_count;
    }
    f->pool_count = saved_count;

    size_t coord_count = saved_count * dimensions;
    for (size_t i = 0; i < coord_count; i++) {
        f->sample_pool[i] = json_object_get_double(json_object_array_get_idx(j_arr, i));
    }

    if (json_object_object_get_ex(root, "total_rows", &j_val) ||
        json_object_object_get_ex(root, "total_rows_seen", &j_val)) {
        f->total_rows_seen = (uint64_t)json_object_get_int64(j_val);
    }
    if (f->total_rows_seen < f->pool_count) {
        f->total_rows_seen = f->pool_count;
    }

    f->is_trained = false;
    *forest_out = f;
    return GEIF_OK;
}

/**
 * @brief Serializes a single forest and writes it to a file or stdout in formatted JSON.
 *
 * Wraps the forest in a temporary single-entry ensemble so all saved models share
 * the same unified GEIF-1.0 schema.
 *
 * @param[in] f    Forest instance to save.
 * @param[in] path File destination path (or "-" for stdout).
 * @return GEIF_OK on success, or GEIF_ERR_IO / GEIF_ERR_INVALID_ARG on failure.
 */
geif_status_t geif_forest_save_json(const geif_forest_t *f, const char *path)
{
    if (!f || !path) return GEIF_ERR_INVALID_ARG;

    geif_category_entry_t entry;
    memset(&entry, 0, sizeof(entry));
    memcpy(entry.category, f->category, sizeof(entry.category));
    entry.category[sizeof(entry.category) - 1] = '\0';
    entry.forest = (geif_forest_t *)f;
    entry.last_updated = time(NULL);
    entry.total_rows = f->total_rows_seen;

    geif_ensemble_t ens;
    memset(&ens, 0, sizeof(ens));
    ens.dimensions = f->dimensions;
    ens.config = f->config;
    ens.total_input_cols = f->total_input_cols;
    memcpy(ens.label_dims_spec, f->label_dims_spec, sizeof(ens.label_dims_spec));
    memcpy(ens.include_dims_spec, f->include_dims_spec, sizeof(ens.include_dims_spec));
    memcpy(ens.ignore_dims_spec, f->ignore_dims_spec, sizeof(ens.ignore_dims_spec));
    memcpy(ens.category_dims_spec, f->category_dims_spec, sizeof(ens.category_dims_spec));
    ens.decimals = (f->decimals >= 0) ? f->decimals : 6;
    ens.scale_score = f->scale_score;
    ens.input_delimiter = f->input_delimiter ? f->input_delimiter : ',';
    ens.has_header = f->has_header;
    ens.aggregate = f->aggregate;
    ens.entries = &entry;
    ens.count = 1;
    ens.capacity = 1;

    return geif_ensemble_save_json(&ens, path);
}

/**
 * @brief Serializes an ensemble containing multiple category sub-forests to JSON.
 *
 * Emits global ensemble configuration once at the root level and a minimal array
 * of sub-forest objects containing only category, last_updated, total_rows, and sample_pool.
 *
 * @param[in] ens  Ensemble instance to save.
 * @param[in] path File destination path (or "-" for stdout).
 * @return GEIF_OK on success, or GEIF_ERR_IO / GEIF_ERR_INVALID_ARG on failure.
 */
geif_status_t geif_ensemble_save_json(const geif_ensemble_t *ens, const char *path)
{
    if (!ens || !path) return GEIF_ERR_INVALID_ARG;

    struct json_object *root = json_object_new_object();
    if (!root) return GEIF_ERR_OUT_OF_MEMORY;

    json_object_object_add(root, "format", json_object_new_string("GEIF-1.0"));
    json_object_object_add(root, "algorithm", json_object_new_string(geif_algo_name(ens->config.algo)));
    json_object_object_add(root, "dimensions", json_object_new_int((int)ens->dimensions));

    uint32_t tree_cnt = ens->config.tree_count;
    uint32_t samples_cnt = ens->config.samples_per_tree;
    if (ens->count > 0 && ens->entries[0].forest) {
        if (tree_cnt == 0) tree_cnt = ens->entries[0].forest->tree_count;
        if (samples_cnt == 0) samples_cnt = ens->entries[0].forest->config.samples_per_tree;
    }
    if (ens->config.algo != GEIF_ALGO_EXEMPLAR && tree_cnt > 0) {
        json_object_object_add(root, "tree_count", json_object_new_int((int)tree_cnt));
    }
    if (samples_cnt > 0) {
        json_object_object_add(root, "samples_per_tree", json_object_new_int((int)samples_cnt));
    }

    json_object_object_add(root, "total_input_cols", json_object_new_int((int)ens->total_input_cols));
    json_object_object_add(root, "label_dims", json_object_new_string(ens->label_dims_spec));
    json_object_object_add(root, "include_dims", json_object_new_string(ens->include_dims_spec));
    json_object_object_add(root, "ignore_dims", json_object_new_string(ens->ignore_dims_spec));
    json_object_object_add(root, "category_dims", json_object_new_string(ens->category_dims_spec));
    int ens_dec = (ens->decimals >= 0) ? ens->decimals : 6;

    char sep_str[2] = { ens->input_delimiter ? ens->input_delimiter : ',', '\0' };
    json_object_object_add(root, "input_separator", json_object_new_string(sep_str));
    json_object_object_add(root, "has_header", json_object_new_boolean(ens->has_header));
    json_object_object_add(root, "aggregate", json_object_new_boolean(ens->aggregate));

    // Save sub-forests array with only per-forest state
    struct json_object *j_forests = json_object_new_array();
    for (size_t i = 0; i < ens->count; i++) {
        if (ens->entries[i].forest) {
            struct json_object *jf = geif_forest_entry_to_json_object(
                ens->entries[i].forest,
                ens->entries[i].category,
                ens->entries[i].last_updated,
                ens->entries[i].total_rows,
                ens_dec);
            if (jf) {
                json_object_array_add(j_forests, jf);
            }
        }
    }
    json_object_object_add(root, "forests", j_forests);

    int ret;
    if (strcmp(path, "-") == 0) {
        const char *json_str = json_object_to_json_string_ext(root, JSON_C_TO_STRING_PRETTY);
        ret = (json_str && fputs(json_str, stdout) >= 0 && fputc('\n', stdout) >= 0) ? 0 : -1;
    } else {
        ret = json_object_to_file_ext(path, root, JSON_C_TO_STRING_PRETTY);
    }
    json_object_put(root);

    return (ret == 0) ? GEIF_OK : GEIF_ERR_IO;
}

/**
 * @brief Reads and parses a JSON model file or stdin into a geif_ensemble_t.
 *
 * Supports native GEIF models, multi-category ensemble JSON files, and legacy CEIF
 * model schemas, dynamically reconstructing memory structures and building trees.
 *
 * @param[out] ensemble_out Pointer receiving the loaded ensemble.
 * @param[in]  path         File path or "-" to read from stdin.
 * @return GEIF_OK on success, or an error status code.
 */
geif_status_t geif_ensemble_load_json(geif_ensemble_t **ensemble_out, const char *path)
{
    if (!ensemble_out || !path) return GEIF_ERR_INVALID_ARG;

    struct json_object *root = NULL;
    if (strcmp(path, "-") == 0) {
        struct json_tokener *tok = json_tokener_new();
        if (!tok) return GEIF_ERR_OUT_OF_MEMORY;
        char buffer[4096];
        size_t bytes_read;
        enum json_tokener_error jerr = json_tokener_continue;
        while ((bytes_read = fread(buffer, 1, sizeof(buffer), stdin)) > 0) {
            root = json_tokener_parse_ex(tok, buffer, (int)bytes_read);
            jerr = json_tokener_get_error(tok);
            if (root || jerr != json_tokener_continue) break;
        }
        json_tokener_free(tok);
        if (!root) return GEIF_ERR_FORMAT_CORRUPT;
    } else {
        root = json_object_from_file(path);
        if (!root) return GEIF_ERR_IO;
    }

    struct json_object *j_val;
    uint32_t dimensions = 0;
    if (json_object_object_get_ex(root, "dimensions", &j_val)) {
        dimensions = (uint32_t)json_object_get_int(j_val);
    }

    geif_config_t cfg = geif_config_default();
    if (json_object_object_get_ex(root, "tree_count", &j_val)) cfg.tree_count = (uint32_t)json_object_get_int(j_val);
    if (json_object_object_get_ex(root, "samples_per_tree", &j_val)) cfg.samples_per_tree = (uint32_t)json_object_get_int(j_val);
    if (json_object_object_get_ex(root, "max_depth", &j_val)) cfg.max_depth = (uint32_t)json_object_get_int(j_val);
    if (json_object_object_get_ex(root, "kappa", &j_val)) cfg.kappa = json_object_get_double(j_val);
    if (json_object_object_get_ex(root, "alpha", &j_val)) cfg.alpha = json_object_get_double(j_val);
    if (json_object_object_get_ex(root, "algorithm", &j_val)) {
        cfg.algo = geif_algo_from_name(json_object_get_string(j_val));
    }

    struct json_object *globals = NULL;
    json_object_object_get_ex(root, "globals", &globals);
    if (!json_object_object_get_ex(root, "algorithm", NULL) && globals != NULL) {
        cfg.algo = GEIF_ALGO_CEIF;
    }

    struct json_object *j_forests = NULL;
    bool has_forests = json_object_object_get_ex(root, "forests", &j_forests);

    if (dimensions == 0 && globals) {
        if (json_object_object_get_ex(globals, "dimensions", &j_val)) {
            dimensions = (uint32_t)json_object_get_int(j_val);
        }
    }

    if (dimensions == 0 && has_forests && json_object_array_length(j_forests) > 0) {
        struct json_object *first_f = json_object_array_get_idx(j_forests, 0);
        if (json_object_object_get_ex(first_f, "dimensions", &j_val)) {
            dimensions = (uint32_t)json_object_get_int(j_val);
        }
    }

    if (dimensions == 0) {
        json_object_put(root);
        return GEIF_ERR_FORMAT_CORRUPT;
    }

    if (globals) {
        if (json_object_object_get_ex(globals, "treeCount", &j_val)) {
            cfg.tree_count = (uint32_t)json_object_get_int(j_val);
        }
        if (json_object_object_get_ex(globals, "samplesMax", &j_val)) {
            cfg.samples_per_tree = (uint32_t)json_object_get_int(j_val);
        }
    }

    geif_ensemble_t *ens = NULL;
    geif_status_t status = geif_ensemble_create(&ens, dimensions, &cfg);
    if (status != GEIF_OK) {
        json_object_put(root);
        return status;
    }

    if (json_object_object_get_ex(root, "total_input_cols", &j_val)) ens->total_input_cols = (uint32_t)json_object_get_int(j_val);
    if (json_object_object_get_ex(root, "label_dims", &j_val)) {
        strncpy(ens->label_dims_spec, json_object_get_string(j_val), sizeof(ens->label_dims_spec) - 1);
    }
    if (json_object_object_get_ex(root, "include_dims", &j_val)) {
        strncpy(ens->include_dims_spec, json_object_get_string(j_val), sizeof(ens->include_dims_spec) - 1);
    }
    if (json_object_object_get_ex(root, "ignore_dims", &j_val)) {
        strncpy(ens->ignore_dims_spec, json_object_get_string(j_val), sizeof(ens->ignore_dims_spec) - 1);
    }
    if (json_object_object_get_ex(root, "category_dims", &j_val)) {
        strncpy(ens->category_dims_spec, json_object_get_string(j_val), sizeof(ens->category_dims_spec) - 1);
    }
    ens->scale_score = true;

    if (globals) {
        if (ens->label_dims_spec[0] == '\0' && json_object_object_get_ex(globals, "labelDims", &j_val)) {
            strncpy(ens->label_dims_spec, json_object_get_string(j_val), sizeof(ens->label_dims_spec) - 1);
        }
        if (ens->include_dims_spec[0] == '\0' && json_object_object_get_ex(globals, "includeDims", &j_val)) {
            strncpy(ens->include_dims_spec, json_object_get_string(j_val), sizeof(ens->include_dims_spec) - 1);
        }
        if (ens->ignore_dims_spec[0] == '\0' && json_object_object_get_ex(globals, "ignoreDims", &j_val)) {
            strncpy(ens->ignore_dims_spec, json_object_get_string(j_val), sizeof(ens->ignore_dims_spec) - 1);
        }
        if (ens->category_dims_spec[0] == '\0' && json_object_object_get_ex(globals, "categoryDims", &j_val)) {
            strncpy(ens->category_dims_spec, json_object_get_string(j_val), sizeof(ens->category_dims_spec) - 1);
        }
        if ((ens->input_delimiter == '\0' || ens->input_delimiter == ',') &&
            (json_object_object_get_ex(globals, "inputSeparator", &j_val) ||
             json_object_object_get_ex(globals, "input_separator", &j_val))) {
            const char *s = json_object_get_string(j_val);
            if (s && s[0] != '\0') ens->input_delimiter = s[0];
        }
        if (json_object_object_get_ex(globals, "header", &j_val)) {
            ens->has_header = json_object_get_boolean(j_val);
        }
        if (json_object_object_get_ex(globals, "aggregate", &j_val)) {
            ens->aggregate = json_object_get_boolean(j_val);
        }
    }

    if (json_object_object_get_ex(root, "input_separator", &j_val) ||
        json_object_object_get_ex(root, "inputSeparator", &j_val)) {
        const char *s = json_object_get_string(j_val);
        if (s && s[0] != '\0') ens->input_delimiter = s[0];
    }
    if (json_object_object_get_ex(root, "has_header", &j_val) ||
        json_object_object_get_ex(root, "header", &j_val)) {
        ens->has_header = json_object_get_boolean(j_val);
    }
    if (json_object_object_get_ex(root, "aggregate", &j_val)) {
        ens->aggregate = json_object_get_boolean(j_val);
    }

    if (has_forests) {
        int f_count = json_object_array_length(j_forests);
        for (int i = 0; i < f_count; i++) {
            struct json_object *jf = json_object_array_get_idx(j_forests, i);
            geif_forest_t *sub = NULL;
            struct json_object *j_samples = NULL;
            if (geif_forest_from_json_object(&sub, jf, dimensions, &cfg) == GEIF_OK && sub) {
                // Native GEIF format
            } else if (json_object_object_get_ex(jf, "samples", &j_samples)) {
                // CEIF format sub-forest with raw samples
                geif_status_t fstat = geif_forest_create(&sub, dimensions, &cfg);
                if (fstat == GEIF_OK && sub) {
                    if (json_object_object_get_ex(jf, "category", &j_val)) {
                        strncpy(sub->category, json_object_get_string(j_val), sizeof(sub->category) - 1);
                    }
                    int s_count = json_object_array_length(j_samples);
                    double *pt = (double *)calloc(dimensions, sizeof(double));
                    if (pt) {
                        for (int s = 0; s < s_count; s++) {
                            struct json_object *js = json_object_array_get_idx(j_samples, s);
                            int d_cnt = json_object_array_length(js);
                            for (uint32_t d = 0; d < dimensions; d++) {
                                pt[d] = (d < (uint32_t)d_cnt) ? json_object_get_double(json_object_array_get_idx(js, d)) : 0.0;
                            }
                            geif_forest_feed(sub, pt);
                        }
                        free(pt);
                    }
                    if (json_object_object_get_ex(jf, "extraRows", &j_val)) {
                        sub->total_rows_seen += (uint64_t)json_object_get_int64(j_val);
                    }
                    sub->aggregate = ens->aggregate;
                    sub->summary_slot = -1;
                }
            }

            if (sub) {
                const char *cat_str = sub->category;
                time_t last_updated = time(NULL);
                if (json_object_object_get_ex(jf, "last_updated", &j_val)) {
                    last_updated = (time_t)json_object_get_int64(j_val);
                } else if (json_object_object_get_ex(jf, "lastUpdated", &j_val)) {
                    last_updated = (time_t)json_object_get_int64(j_val);
                }
                uint64_t total_rows = sub->total_rows_seen;
                if (json_object_object_get_ex(jf, "total_rows", &j_val)) {
                    total_rows = (uint64_t)json_object_get_int64(j_val);
                    sub->total_rows_seen = total_rows;
                }
                geif_ensemble_add_forest(ens, cat_str, sub, total_rows, last_updated);
            }
        }
    } else {
        // Single-forest fallback
        geif_forest_t *sub = NULL;
        if (geif_forest_from_json_object(&sub, root, dimensions, &cfg) == GEIF_OK && sub) {
            const char *cat_str = sub->category;
            geif_ensemble_add_forest(ens, cat_str, sub, sub->total_rows_seen, time(NULL));
        }
    }

    if (ens->input_delimiter == ',' && ens->count > 0 && ens->entries[0].forest && ens->entries[0].forest->input_delimiter != '\0') {
        ens->input_delimiter = ens->entries[0].forest->input_delimiter;
    }
    if (!ens->has_header && ens->count > 0 && ens->entries[0].forest && ens->entries[0].forest->has_header) {
        ens->has_header = ens->entries[0].forest->has_header;
    }
    if (!ens->aggregate && ens->count > 0 && ens->entries[0].forest && ens->entries[0].forest->aggregate) {
        ens->aggregate = ens->entries[0].forest->aggregate;
    }

    for (size_t i = 0; i < ens->count; i++) {
        if (ens->entries[i].forest) {
            ens->entries[i].forest->scale_score = ens->scale_score;
            ens->entries[i].forest->input_delimiter = ens->input_delimiter;
            ens->entries[i].forest->has_header = ens->has_header;
            ens->entries[i].forest->aggregate = ens->aggregate;
            ens->entries[i].forest->summary_slot = -1;
        }
    }

    json_object_put(root);
    *ensemble_out = ens;
    return GEIF_OK;
}

/**
 * @brief Reads and parses a single forest model from a JSON file.
 *
 * Convenience function that loads the model via geif_ensemble_load_json()
 * and extracts its primary/first sub-forest.
 *
 * @param[out] forest_out Pointer receiving the loaded forest.
 * @param[in]  path       File path or "-" to read from stdin.
 * @return GEIF_OK on success, or an error status code.
 */
geif_status_t geif_forest_load_json(geif_forest_t **forest_out, const char *path)
{
    geif_ensemble_t *ens = NULL;
    geif_status_t status = geif_ensemble_load_json(&ens, path);
    if (status != GEIF_OK || !ens) return status;
    if (ens->count == 0 || !ens->entries[0].forest) {
        geif_ensemble_destroy(ens);
        return GEIF_ERR_FORMAT_CORRUPT;
    }

    *forest_out = ens->entries[0].forest;
    ens->entries[0].forest = NULL;
    geif_ensemble_destroy(ens);
    return GEIF_OK;
}
