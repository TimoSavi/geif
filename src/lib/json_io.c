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

struct json_object *geif_forest_to_json_object(const geif_forest_t *f);
geif_status_t geif_forest_from_json_object(geif_forest_t **forest_out, struct json_object *root);

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
 * @brief Serializes a geif_forest_t into a json-c json_object.
 *
 * Encodes forest hyperparameters, coordinate envelopes, dimension metadata,
 * calibration parameters, and the raw reservoir sample pool.
 *
 * @param[in] f Forest instance to serialize.
 * @return Pointer to newly allocated json_object root, or NULL on error.
 */
struct json_object *geif_forest_to_json_object(const geif_forest_t *f)
{
    if (!f) return NULL;

    struct json_object *root = json_object_new_object();
    if (!root) return NULL;

    int dec = (f->decimals >= 0) ? f->decimals : 6;

    json_object_object_add(root, "format", json_object_new_string("GEIF-1.0"));
    json_object_object_add(root, "algorithm", json_object_new_string(geif_algo_name(f->config.algo)));
    json_object_object_add(root, "dimensions", json_object_new_int((int)f->dimensions));
    json_object_object_add(root, "tree_count", json_object_new_int((int)f->tree_count));
    json_object_object_add(root, "samples_per_tree", json_object_new_int((int)f->config.samples_per_tree));
    json_object_object_add(root, "max_depth", json_object_new_int((int)f->config.max_depth));
    json_object_object_add(root, "kappa", json_object_new_double(f->config.kappa));
    json_object_object_add(root, "alpha", json_object_new_double(f->config.alpha));
    json_object_object_add(root, "seed", json_object_new_int64((int64_t)f->config.seed));
    json_object_object_add(root, "total_rows_seen", json_object_new_int64((int64_t)f->total_rows_seen));
    json_object_object_add(root, "decimals", json_object_new_int(dec));

    // Metadata & Column Configuration
    if (f->category[0] != '\0') {
        json_object_object_add(root, "category", json_object_new_string(f->category));
    }
    json_object_object_add(root, "total_input_cols", json_object_new_int((int)f->total_input_cols));
    json_object_object_add(root, "label_dims", json_object_new_string(f->label_dims_spec));
    json_object_object_add(root, "include_dims", json_object_new_string(f->include_dims_spec));
    json_object_object_add(root, "ignore_dims", json_object_new_string(f->ignore_dims_spec));
    json_object_object_add(root, "category_dims", json_object_new_string(f->category_dims_spec));

    // Globals object for CEIF format compatibility
    struct json_object *globals = json_object_new_object();
    json_object_object_add(globals, "labelDims", json_object_new_string(f->label_dims_spec));
    json_object_object_add(globals, "includeDims", json_object_new_string(f->include_dims_spec));
    json_object_object_add(globals, "ignoreDims", json_object_new_string(f->ignore_dims_spec));
    json_object_object_add(globals, "categoryDims", json_object_new_string(f->category_dims_spec));
    json_object_object_add(globals, "decimals", json_object_new_int(dec));
    json_object_object_add(root, "globals", globals);

    // Save sample pool with clean decimal precision
    struct json_object *j_pool = json_object_new_array();
    size_t total_pool_coords = f->pool_count * f->dimensions;
    for (size_t i = 0; i < total_pool_coords; i++) {
        json_object_array_add(j_pool, geif_clean_double_json(f->sample_pool[i], dec));
    }
    json_object_object_add(root, "pool_count", json_object_new_int((int)f->pool_count));
    json_object_object_add(root, "sample_pool", j_pool);

    return root;
}

/**
 * @brief Deserializes a json-c json_object into a populated geif_forest_t.
 *
 * Instantiates the forest, populates envelopes and sample pool, and automatically
 * builds and calibrates the tree structures in RAM.
 *
 * @param[out] forest_out Pointer receiving the newly allocated forest.
 * @param[in]  root       json-c root object representing the forest.
 * @return GEIF_OK on success, or GEIF_ERR_FORMAT_CORRUPT / GEIF_ERR_OUT_OF_MEMORY.
 */
geif_status_t geif_forest_from_json_object(geif_forest_t **forest_out, struct json_object *root)
{
    if (!forest_out || !root) return GEIF_ERR_INVALID_ARG;

    struct json_object *j_val;
    if (!json_object_object_get_ex(root, "dimensions", &j_val)) {
        return GEIF_ERR_FORMAT_CORRUPT;
    }
    uint32_t dimensions = (uint32_t)json_object_get_int(j_val);

    geif_config_t cfg = geif_config_default();
    if (json_object_object_get_ex(root, "algorithm", &j_val)) {
        cfg.algo = geif_algo_from_name(json_object_get_string(j_val));
    } else {
        if (json_object_object_get_ex(root, "trees", NULL) || json_object_object_get_ex(root, "globals", NULL)) {
            cfg.algo = GEIF_ALGO_CEIF;
        } else {
            cfg.algo = GEIF_ALGO_DEFAULT;
        }
    }
    if (json_object_object_get_ex(root, "seed", &j_val)) cfg.seed = (uint32_t)json_object_get_int64(j_val);
    if (json_object_object_get_ex(root, "tree_count", &j_val)) cfg.tree_count = (uint32_t)json_object_get_int(j_val);
    if (json_object_object_get_ex(root, "samples_per_tree", &j_val)) cfg.samples_per_tree = (uint32_t)json_object_get_int(j_val);
    if (json_object_object_get_ex(root, "max_depth", &j_val)) cfg.max_depth = (uint32_t)json_object_get_int(j_val);
    if (json_object_object_get_ex(root, "kappa", &j_val)) cfg.kappa = json_object_get_double(j_val);
    if (json_object_object_get_ex(root, "alpha", &j_val)) cfg.alpha = json_object_get_double(j_val);

    geif_forest_t *f = NULL;
    geif_status_t status = geif_forest_create(&f, dimensions, &cfg);
    if (status != GEIF_OK) {
        return status;
    }

    if (json_object_object_get_ex(root, "total_rows_seen", &j_val)) {
        f->total_rows_seen = (uint64_t)json_object_get_int64(j_val);
    }

    // Metadata & Column Configuration
    if (json_object_object_get_ex(root, "category", &j_val)) {
        strncpy(f->category, json_object_get_string(j_val), sizeof(f->category) - 1);
        f->category[sizeof(f->category) - 1] = '\0';
    }
    if (json_object_object_get_ex(root, "total_input_cols", &j_val)) {
        f->total_input_cols = (uint32_t)json_object_get_int(j_val);
    }
    if (json_object_object_get_ex(root, "label_dims", &j_val)) {
        strncpy(f->label_dims_spec, json_object_get_string(j_val), sizeof(f->label_dims_spec) - 1);
        f->label_dims_spec[sizeof(f->label_dims_spec) - 1] = '\0';
    }
    if (json_object_object_get_ex(root, "include_dims", &j_val)) {
        strncpy(f->include_dims_spec, json_object_get_string(j_val), sizeof(f->include_dims_spec) - 1);
        f->include_dims_spec[sizeof(f->include_dims_spec) - 1] = '\0';
    }
    if (json_object_object_get_ex(root, "ignore_dims", &j_val)) {
        strncpy(f->ignore_dims_spec, json_object_get_string(j_val), sizeof(f->ignore_dims_spec) - 1);
        f->ignore_dims_spec[sizeof(f->ignore_dims_spec) - 1] = '\0';
    }
    if (json_object_object_get_ex(root, "category_dims", &j_val)) {
        strncpy(f->category_dims_spec, json_object_get_string(j_val), sizeof(f->category_dims_spec) - 1);
        f->category_dims_spec[sizeof(f->category_dims_spec) - 1] = '\0';
    }

    // Globals fallback for CEIF models
    struct json_object *globals = NULL;
    if (json_object_object_get_ex(root, "globals", &globals)) {
        if (f->label_dims_spec[0] == '\0' && json_object_object_get_ex(globals, "labelDims", &j_val)) {
            strncpy(f->label_dims_spec, json_object_get_string(j_val), sizeof(f->label_dims_spec) - 1);
            f->label_dims_spec[sizeof(f->label_dims_spec) - 1] = '\0';
        }
        if (f->include_dims_spec[0] == '\0' && json_object_object_get_ex(globals, "includeDims", &j_val)) {
            strncpy(f->include_dims_spec, json_object_get_string(j_val), sizeof(f->include_dims_spec) - 1);
            f->include_dims_spec[sizeof(f->include_dims_spec) - 1] = '\0';
        }
        if (f->ignore_dims_spec[0] == '\0' && json_object_object_get_ex(globals, "ignoreDims", &j_val)) {
            strncpy(f->ignore_dims_spec, json_object_get_string(j_val), sizeof(f->ignore_dims_spec) - 1);
            f->ignore_dims_spec[sizeof(f->ignore_dims_spec) - 1] = '\0';
        }
        if (f->category_dims_spec[0] == '\0' && json_object_object_get_ex(globals, "categoryDims", &j_val)) {
            strncpy(f->category_dims_spec, json_object_get_string(j_val), sizeof(f->category_dims_spec) - 1);
            f->category_dims_spec[sizeof(f->category_dims_spec) - 1] = '\0';
        }
        if (json_object_object_get_ex(globals, "decimals", &j_val)) {
            f->decimals = json_object_get_int(j_val);
        }
    }

    if (json_object_object_get_ex(root, "decimals", &j_val)) {
        f->decimals = json_object_get_int(j_val);
    }
    f->scale_score = true;
    if (json_object_object_get_ex(root, "outlier_score", &j_val)) {
        const char *os = json_object_get_string(j_val);
        (void)os;
    }
    if (globals && json_object_object_get_ex(globals, "outlierScore", &j_val)) {
        const char *os = json_object_get_string(j_val);
        (void)os;
    }

    // Load sample pool
    struct json_object *j_arr = NULL;
    if (json_object_object_get_ex(root, "pool_count", &j_val)) {
        f->pool_count = (size_t)json_object_get_int(j_val);
    }
    if (json_object_object_get_ex(root, "sample_pool", &j_arr)) {
        size_t len = (size_t)json_object_array_length(j_arr);
        for (size_t i = 0; i < len && i < f->pool_capacity * dimensions; i++) {
            f->sample_pool[i] = json_object_get_double(json_object_array_get_idx(j_arr, i));
        }
    }
    f->is_trained = false;

    *forest_out = f;
    return GEIF_OK;
}

/**
 * @brief Serializes a single forest and writes it to a file or stdout in formatted JSON.
 *
 * @param[in] f    Forest instance to save.
 * @param[in] path File destination path (or "-" for stdout).
 * @return GEIF_OK on success, or GEIF_ERR_IO / GEIF_ERR_INVALID_ARG on failure.
 */
geif_status_t geif_forest_save_json(const geif_forest_t *f, const char *path)
{
    if (!f || !path) return GEIF_ERR_INVALID_ARG;

    struct json_object *root = geif_forest_to_json_object(f);
    if (!root) return GEIF_ERR_OUT_OF_MEMORY;

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
 * @brief Serializes an ensemble containing multiple category sub-forests to JSON.
 *
 * Emits global ensemble configuration, category routing dimensions, and an array
 * of sub-forest objects with their respective sample pools and metadata.
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
    json_object_object_add(root, "total_input_cols", json_object_new_int((int)ens->total_input_cols));
    json_object_object_add(root, "subforest_count", json_object_new_int((int)ens->count));
    json_object_object_add(root, "label_dims", json_object_new_string(ens->label_dims_spec));
    json_object_object_add(root, "include_dims", json_object_new_string(ens->include_dims_spec));
    json_object_object_add(root, "ignore_dims", json_object_new_string(ens->ignore_dims_spec));
    json_object_object_add(root, "category_dims", json_object_new_string(ens->category_dims_spec));
    const char *outlier_spec = (ens->outlier_score_spec[0] != '\0') ? ens->outlier_score_spec : "0.500000";
    json_object_object_add(root, "outlier_score", json_object_new_string(outlier_spec));
    int ens_dec = (ens->decimals >= 0) ? ens->decimals : 6;
    json_object_object_add(root, "decimals", json_object_new_int(ens_dec));

    // Globals object for CEIF compatibility
    struct json_object *globals = json_object_new_object();
    json_object_object_add(globals, "labelDims", json_object_new_string(ens->label_dims_spec));
    json_object_object_add(globals, "includeDims", json_object_new_string(ens->include_dims_spec));
    json_object_object_add(globals, "ignoreDims", json_object_new_string(ens->ignore_dims_spec));
    json_object_object_add(globals, "categoryDims", json_object_new_string(ens->category_dims_spec));
    json_object_object_add(globals, "outlierScore", json_object_new_string(outlier_spec));
    json_object_object_add(globals, "decimals", json_object_new_int(ens_dec));
    json_object_object_add(root, "globals", globals);

    // Save sub-forests array
    struct json_object *j_forests = json_object_new_array();
    for (size_t i = 0; i < ens->count; i++) {
        if (ens->entries[i].forest) {
            struct json_object *jf = geif_forest_to_json_object(ens->entries[i].forest);
            if (jf) {
                json_object_object_add(jf, "category", json_object_new_string(ens->entries[i].category));
                json_object_object_add(jf, "last_updated", json_object_new_int64((int64_t)ens->entries[i].last_updated));
                json_object_object_add(jf, "total_rows", json_object_new_int64((int64_t)ens->entries[i].total_rows));
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
    if (json_object_object_get_ex(root, "outlier_score", &j_val)) {
        strncpy(ens->outlier_score_spec, json_object_get_string(j_val), sizeof(ens->outlier_score_spec) - 1);
    }

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
        if (ens->outlier_score_spec[0] == '\0' && json_object_object_get_ex(globals, "outlierScore", &j_val)) {
            strncpy(ens->outlier_score_spec, json_object_get_string(j_val), sizeof(ens->outlier_score_spec) - 1);
        }
        if (json_object_object_get_ex(globals, "decimals", &j_val)) {
            ens->decimals = json_object_get_int(j_val);
        }
    }

    if (json_object_object_get_ex(root, "decimals", &j_val)) {
        ens->decimals = json_object_get_int(j_val);
    }

    if (has_forests) {
        int f_count = json_object_array_length(j_forests);
        for (int i = 0; i < f_count; i++) {
            struct json_object *jf = json_object_array_get_idx(j_forests, i);
            geif_forest_t *sub = NULL;
            struct json_object *j_samples = NULL;
            if (geif_forest_from_json_object(&sub, jf) == GEIF_OK && sub) {
                // Native GEIF format
                if (ens->decimals > 0 && sub->decimals <= 0) sub->decimals = ens->decimals;
            } else if (json_object_object_get_ex(jf, "samples", &j_samples)) {
                // CEIF format sub-forest with raw samples
                geif_status_t fstat = geif_forest_create(&sub, dimensions, &cfg);
                if (fstat == GEIF_OK && sub) {
                    if (ens->decimals > 0) sub->decimals = ens->decimals;
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
                }
                geif_ensemble_add_forest(ens, cat_str, sub, total_rows, last_updated);
            }
        }
    } else {
        // Single-forest fallback
        geif_forest_t *sub = NULL;
        if (geif_forest_from_json_object(&sub, root) == GEIF_OK && sub) {
            const char *cat_str = sub->category;
            geif_ensemble_add_forest(ens, cat_str, sub, sub->total_rows_seen, time(NULL));
        }
    }

    for (size_t i = 0; i < ens->count; i++) {
        if (ens->entries[i].forest) {
            ens->entries[i].forest->scale_score = ens->scale_score;
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
