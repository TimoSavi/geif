/**
 * @file json_io.c
 * @brief JSON serialization and deserialization for GEIF models.
 */

#include "geif/geif.h"
#include <json-c/json.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

geif_status_t geif_forest_save_json(const geif_forest_t *f, const char *path)
{
    if (!f || !path) return GEIF_ERR_INVALID_ARG;

    struct json_object *root = json_object_new_object();
    if (!root) return GEIF_ERR_OUT_OF_MEMORY;

    json_object_object_add(root, "format", json_object_new_string("GEIF-1.0"));
    json_object_object_add(root, "dimensions", json_object_new_int((int)f->dimensions));
    json_object_object_add(root, "tree_count", json_object_new_int((int)f->tree_count));
    json_object_object_add(root, "samples_per_tree", json_object_new_int((int)f->config.samples_per_tree));
    json_object_object_add(root, "max_depth", json_object_new_int((int)f->config.max_depth));
    json_object_object_add(root, "kappa", json_object_new_double(f->config.kappa));
    json_object_object_add(root, "alpha", json_object_new_double(f->config.alpha));
    json_object_object_add(root, "H_train_max", json_object_new_double(f->H_train_max));
    json_object_object_add(root, "H_max", json_object_new_double(f->H_max));
    json_object_object_add(root, "delta_nominal", json_object_new_double(f->delta_nominal));
    json_object_object_add(root, "total_rows_seen", json_object_new_int64((int64_t)f->total_rows_seen));

    // Save envelopes
    struct json_object *j_min = json_object_new_array();
    struct json_object *j_max = json_object_new_array();
    struct json_object *j_eff = json_object_new_array();
    struct json_object *j_act = json_object_new_array();

    for (uint32_t j = 0; j < f->dimensions; j++) {
        json_object_array_add(j_min, json_object_new_double(f->envelope_min[j]));
        json_object_array_add(j_max, json_object_new_double(f->envelope_max[j]));
        json_object_array_add(j_eff, json_object_new_double(f->effective_span[j]));
        json_object_array_add(j_act, json_object_new_int((int)f->dim_active[j]));
    }
    json_object_object_add(root, "envelope_min", j_min);
    json_object_object_add(root, "envelope_max", j_max);
    json_object_object_add(root, "effective_span", j_eff);
    json_object_object_add(root, "dim_active", j_act);

    // Save sample pool
    struct json_object *j_pool = json_object_new_array();
    size_t total_pool_coords = f->pool_count * f->dimensions;
    for (size_t i = 0; i < total_pool_coords; i++) {
        json_object_array_add(j_pool, json_object_new_double(f->sample_pool[i]));
    }
    json_object_object_add(root, "pool_count", json_object_new_int((int)f->pool_count));
    json_object_object_add(root, "sample_pool", j_pool);

    // Save trees
    struct json_object *j_trees = json_object_new_array();
    for (uint32_t t = 0; t < f->tree_count; t++) {
        const geif_tree_t *tree = &f->trees[t];
        struct json_object *j_tree = json_object_new_object();
        json_object_object_add(j_tree, "node_count", json_object_new_int((int)tree->node_count));

        struct json_object *j_nodes = json_object_new_array();
        for (size_t n = 0; n < tree->node_count; n++) {
            const geif_node_t *node = &tree->nodes[n];
            struct json_object *j_node = json_object_new_object();
            json_object_object_add(j_node, "left", json_object_new_int(node->left_child));
            json_object_object_add(j_node, "right", json_object_new_int(node->right_child));
            json_object_object_add(j_node, "offset", json_object_new_int((int)node->normal_offset));
            json_object_object_add(j_node, "pdotn", json_object_new_double(node->pdotn));
            json_object_object_add(j_node, "weight", json_object_new_double(node->step_weight));
            json_object_object_add(j_node, "delta", json_object_new_double(node->delta_AB));
            json_object_object_add(j_node, "samples", json_object_new_int(node->sample_count));
            json_object_object_add(j_node, "leaf_idx", json_object_new_int((int)node->leaf_point_idx));
            json_object_array_add(j_nodes, j_node);
        }
        json_object_object_add(j_tree, "nodes", j_nodes);

        struct json_object *j_normals = json_object_new_array();
        size_t total_normals = tree->node_count * f->dimensions;
        for (size_t k = 0; k < total_normals; k++) {
            json_object_array_add(j_normals, json_object_new_double(tree->normals_pool[k]));
        }
        json_object_object_add(j_tree, "normals", j_normals);

        json_object_array_add(j_trees, j_tree);
    }
    json_object_object_add(root, "trees", j_trees);

    int ret = json_object_to_file_ext(path, root, JSON_C_TO_STRING_PRETTY);
    json_object_put(root);

    return (ret == 0) ? GEIF_OK : GEIF_ERR_IO;
}

geif_status_t geif_forest_load_json(geif_forest_t **forest_out, const char *path)
{
    if (!forest_out || !path) return GEIF_ERR_INVALID_ARG;

    struct json_object *root = json_object_from_file(path);
    if (!root) return GEIF_ERR_IO;

    struct json_object *j_val;
    if (!json_object_object_get_ex(root, "dimensions", &j_val)) {
        json_object_put(root);
        return GEIF_ERR_FORMAT_CORRUPT;
    }
    uint32_t dimensions = (uint32_t)json_object_get_int(j_val);

    geif_config_t cfg = geif_config_default();
    if (json_object_object_get_ex(root, "tree_count", &j_val)) cfg.tree_count = (uint32_t)json_object_get_int(j_val);
    if (json_object_object_get_ex(root, "samples_per_tree", &j_val)) cfg.samples_per_tree = (uint32_t)json_object_get_int(j_val);
    if (json_object_object_get_ex(root, "max_depth", &j_val)) cfg.max_depth = (uint32_t)json_object_get_int(j_val);
    if (json_object_object_get_ex(root, "kappa", &j_val)) cfg.kappa = json_object_get_double(j_val);
    if (json_object_object_get_ex(root, "alpha", &j_val)) cfg.alpha = json_object_get_double(j_val);

    geif_forest_t *f = NULL;
    geif_status_t status = geif_forest_create(&f, dimensions, &cfg);
    if (status != GEIF_OK) {
        json_object_put(root);
        return status;
    }

    if (json_object_object_get_ex(root, "H_train_max", &j_val)) f->H_train_max = json_object_get_double(j_val);
    if (json_object_object_get_ex(root, "H_max", &j_val)) f->H_max = json_object_get_double(j_val);
    if (json_object_object_get_ex(root, "delta_nominal", &j_val)) f->delta_nominal = json_object_get_double(j_val);
    if (json_object_object_get_ex(root, "total_rows_seen", &j_val)) f->total_rows_seen = (uint64_t)json_object_get_int64(j_val);

    // Load envelopes
    struct json_object *j_arr;
    if (json_object_object_get_ex(root, "envelope_min", &j_arr)) {
        for (uint32_t j = 0; j < dimensions; j++) {
            f->envelope_min[j] = json_object_get_double(json_object_array_get_idx(j_arr, j));
        }
    }
    if (json_object_object_get_ex(root, "envelope_max", &j_arr)) {
        for (uint32_t j = 0; j < dimensions; j++) {
            f->envelope_max[j] = json_object_get_double(json_object_array_get_idx(j_arr, j));
            f->envelope_span[j] = f->envelope_max[j] - f->envelope_min[j];
        }
    }
    if (json_object_object_get_ex(root, "effective_span", &j_arr)) {
        for (uint32_t j = 0; j < dimensions; j++) {
            f->effective_span[j] = json_object_get_double(json_object_array_get_idx(j_arr, j));
        }
    }
    if (json_object_object_get_ex(root, "dim_active", &j_arr)) {
        for (uint32_t j = 0; j < dimensions; j++) {
            f->dim_active[j] = (uint8_t)json_object_get_int(json_object_array_get_idx(j_arr, j));
        }
    }

    // Load sample pool
    if (json_object_object_get_ex(root, "pool_count", &j_val)) f->pool_count = (size_t)json_object_get_int(j_val);
    if (json_object_object_get_ex(root, "sample_pool", &j_arr)) {
        size_t len = (size_t)json_object_array_length(j_arr);
        for (size_t i = 0; i < len && i < f->pool_capacity * dimensions; i++) {
            f->sample_pool[i] = json_object_get_double(json_object_array_get_idx(j_arr, i));
        }
    }

    // Load trees
    struct json_object *j_trees;
    if (json_object_object_get_ex(root, "trees", &j_trees)) {
        int t_len = json_object_array_length(j_trees);
        for (int t = 0; t < t_len && t < (int)f->tree_count; t++) {
            struct json_object *j_tree = json_object_array_get_idx(j_trees, t);
            geif_tree_t *tree = &f->trees[t];

            struct json_object *j_nodes;
            if (json_object_object_get_ex(j_tree, "nodes", &j_nodes)) {
                size_t n_len = (size_t)json_object_array_length(j_nodes);
                tree->node_count = n_len;
                tree->node_capacity = n_len;
                tree->nodes = (geif_node_t *)malloc(n_len * sizeof(geif_node_t));

                for (size_t n = 0; n < n_len; n++) {
                    struct json_object *j_node = json_object_array_get_idx(j_nodes, n);
                    struct json_object *jv;
                    geif_node_t *node = &tree->nodes[n];

                    json_object_object_get_ex(j_node, "left", &jv); node->left_child = json_object_get_int(jv);
                    json_object_object_get_ex(j_node, "right", &jv); node->right_child = json_object_get_int(jv);
                    json_object_object_get_ex(j_node, "offset", &jv); node->normal_offset = (uint32_t)json_object_get_int(jv);
                    json_object_object_get_ex(j_node, "pdotn", &jv); node->pdotn = json_object_get_double(jv);
                    json_object_object_get_ex(j_node, "weight", &jv); node->step_weight = json_object_get_double(jv);
                    json_object_object_get_ex(j_node, "delta", &jv); node->delta_AB = json_object_get_double(jv);
                    json_object_object_get_ex(j_node, "samples", &jv); node->sample_count = json_object_get_int(jv);
                    json_object_object_get_ex(j_node, "leaf_idx", &jv); node->leaf_point_idx = (uint32_t)json_object_get_int(jv);
                }
            }

            struct json_object *j_normals;
            if (json_object_object_get_ex(j_tree, "normals", &j_normals)) {
                size_t norm_len = (size_t)json_object_array_length(j_normals);
                tree->normals_capacity = norm_len;
                tree->normals_pool = (double *)malloc(norm_len * sizeof(double));
                for (size_t k = 0; k < norm_len; k++) {
                    tree->normals_pool[k] = json_object_get_double(json_object_array_get_idx(j_normals, k));
                }
            }
        }
    }

    json_object_put(root);
    *forest_out = f;
    return GEIF_OK;
}
