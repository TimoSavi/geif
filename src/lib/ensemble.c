/**
 * @file ensemble.c
 * @brief Multi-category sub-forest ensemble management, dynamic routing, and lifecycle.
 */

#include "geif/geif.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#define GEIF_ENS_INIT_CAPACITY 16
#define GEIF_ENS_INIT_HASH_SIZE 64

static uint32_t ensemble_hash(const char *str)
{
    if (!str) return 0;
    uint32_t h = 2166136261u;
    while (*str) {
        h ^= (uint8_t)*str++;
        h *= 16777619u;
    }
    return h;
}

static void ensemble_rehash(geif_ensemble_t *ens)
{
    size_t new_size = ens->hash_size * 2;
    geif_cat_hash_node_t **new_buckets = (geif_cat_hash_node_t **)calloc(new_size, sizeof(geif_cat_hash_node_t *));
    if (!new_buckets) return;

    for (size_t b = 0; b < ens->hash_size; b++) {
        geif_cat_hash_node_t *curr = ens->hash_buckets[b];
        while (curr) {
            geif_cat_hash_node_t *next = curr->next;
            uint32_t h = ensemble_hash(ens->entries[curr->entry_idx].category);
            size_t nb = (size_t)(h % new_size);
            curr->next = new_buckets[nb];
            new_buckets[nb] = curr;
            curr = next;
        }
    }

    free(ens->hash_buckets);
    ens->hash_buckets = new_buckets;
    ens->hash_size = new_size;
}

geif_status_t geif_ensemble_create(geif_ensemble_t **ensemble_out,
                                   uint32_t dimensions,
                                   const geif_config_t *config)
{
    if (!ensemble_out || dimensions == 0) {
        return GEIF_ERR_INVALID_ARG;
    }

    geif_ensemble_t *ens = (geif_ensemble_t *)calloc(1, sizeof(geif_ensemble_t));
    if (!ens) return GEIF_ERR_OUT_OF_MEMORY;

    ens->dimensions = dimensions;
    ens->config = config ? *config : geif_config_default();
    ens->decimals = 6;

    ens->capacity = GEIF_ENS_INIT_CAPACITY;
    ens->entries = (geif_category_entry_t *)calloc(ens->capacity, sizeof(geif_category_entry_t));
    if (!ens->entries) {
        free(ens);
        return GEIF_ERR_OUT_OF_MEMORY;
    }

    ens->hash_size = GEIF_ENS_INIT_HASH_SIZE;
    ens->hash_buckets = (geif_cat_hash_node_t **)calloc(ens->hash_size, sizeof(geif_cat_hash_node_t *));
    if (!ens->hash_buckets) {
        free(ens->entries);
        free(ens);
        return GEIF_ERR_OUT_OF_MEMORY;
    }

    *ensemble_out = ens;
    return GEIF_OK;
}

void geif_ensemble_destroy(geif_ensemble_t *ens)
{
    if (!ens) return;

    if (ens->entries) {
        for (size_t i = 0; i < ens->count; i++) {
            if (ens->entries[i].forest) {
                geif_forest_destroy(ens->entries[i].forest);
                ens->entries[i].forest = NULL;
            }
        }
        free(ens->entries);
        ens->entries = NULL;
    }

    if (ens->hash_buckets) {
        for (size_t b = 0; b < ens->hash_size; b++) {
            geif_cat_hash_node_t *curr = ens->hash_buckets[b];
            while (curr) {
                geif_cat_hash_node_t *next = curr->next;
                free(curr);
                curr = next;
            }
        }
        free(ens->hash_buckets);
        ens->hash_buckets = NULL;
    }

    free(ens);
}

geif_forest_t *geif_ensemble_find(const geif_ensemble_t *ens, const char *category)
{
    if (!ens || !ens->hash_buckets || ens->count == 0) return NULL;

    const char *cat_key = (category && category[0] != '\0') ? category : "";
    uint32_t h = ensemble_hash(cat_key);
    size_t b = (size_t)(h % ens->hash_size);

    geif_cat_hash_node_t *node = ens->hash_buckets[b];
    while (node) {
        uint32_t idx = node->entry_idx;
        if (strcmp(ens->entries[idx].category, cat_key) == 0) {
            return ens->entries[idx].forest;
        }
        node = node->next;
    }

    return NULL;
}

geif_forest_t *geif_ensemble_get_or_create(geif_ensemble_t *ens, const char *category)
{
    if (!ens) return NULL;

    const char *cat_key = (category && category[0] != '\0') ? category : "";
    geif_forest_t *existing = geif_ensemble_find(ens, cat_key);
    if (existing) return existing;

    // Expand entries array if needed
    if (ens->count >= ens->capacity) {
        size_t new_cap = ens->capacity * 2;
        geif_category_entry_t *new_entries = (geif_category_entry_t *)realloc(ens->entries, new_cap * sizeof(geif_category_entry_t));
        if (!new_entries) return NULL;
        memset(new_entries + ens->capacity, 0, (new_cap - ens->capacity) * sizeof(geif_category_entry_t));
        ens->entries = new_entries;
        ens->capacity = new_cap;
    }

    // Rehash table if load factor is high
    if (ens->count >= ens->hash_size * 3 / 4) {
        ensemble_rehash(ens);
    }

    // Allocate new sub-forest
    geif_forest_t *forest = NULL;
    geif_status_t status = geif_forest_create(&forest, ens->dimensions, &ens->config);
    if (status != GEIF_OK || !forest) return NULL;

    // Propagate metadata to sub-forest
    strncpy(forest->category, cat_key, sizeof(forest->category) - 1);
    forest->category[sizeof(forest->category) - 1] = '\0';
    forest->total_input_cols = ens->total_input_cols;
    strncpy(forest->label_dims_spec, ens->label_dims_spec, sizeof(forest->label_dims_spec) - 1);
    strncpy(forest->include_dims_spec, ens->include_dims_spec, sizeof(forest->include_dims_spec) - 1);
    strncpy(forest->ignore_dims_spec, ens->ignore_dims_spec, sizeof(forest->ignore_dims_spec) - 1);
    strncpy(forest->category_dims_spec, ens->category_dims_spec, sizeof(forest->category_dims_spec) - 1);
    forest->decimals = ens->decimals;

    uint32_t entry_idx = (uint32_t)ens->count++;
    geif_category_entry_t *entry = &ens->entries[entry_idx];
    strncpy(entry->category, cat_key, sizeof(entry->category) - 1);
    entry->category[sizeof(entry->category) - 1] = '\0';
    entry->forest = forest;
    entry->last_updated = time(NULL);
    entry->total_rows = 0;

    // Insert into hash bucket
    uint32_t h = ensemble_hash(cat_key);
    size_t b = (size_t)(h % ens->hash_size);
    geif_cat_hash_node_t *node = (geif_cat_hash_node_t *)malloc(sizeof(geif_cat_hash_node_t));
    if (!node) return forest; // Non-fatal for forest itself, but hash lookup won't find it
    node->entry_idx = entry_idx;
    node->next = ens->hash_buckets[b];
    ens->hash_buckets[b] = node;

    return forest;
}

geif_status_t geif_ensemble_feed(geif_ensemble_t *ens,
                                 const char *category,
                                 const double *point)
{
    if (!ens || !point) return GEIF_ERR_INVALID_ARG;

    const char *cat_key = (category && category[0] != '\0') ? category : "";
    geif_forest_t *forest = geif_ensemble_get_or_create(ens, cat_key);
    if (!forest) return GEIF_ERR_OUT_OF_MEMORY;

    // Update row counts
    uint32_t h = ensemble_hash(cat_key);
    size_t b = (size_t)(h % ens->hash_size);
    geif_cat_hash_node_t *node = ens->hash_buckets[b];
    while (node) {
        if (strcmp(ens->entries[node->entry_idx].category, cat_key) == 0) {
            ens->entries[node->entry_idx].total_rows++;
            ens->entries[node->entry_idx].last_updated = time(NULL);
            break;
        }
        node = node->next;
    }

    return geif_forest_feed(forest, point);
}

geif_status_t geif_ensemble_prune_categories(geif_ensemble_t *ens, uint64_t min_rows)
{
    if (!ens) return GEIF_ERR_INVALID_ARG;
    if (min_rows == 0 || ens->count == 0) return GEIF_OK;

    size_t new_count = 0;
    for (size_t i = 0; i < ens->count; i++) {
        if (ens->entries[i].total_rows < min_rows) {
            if (ens->entries[i].forest) {
                geif_forest_destroy(ens->entries[i].forest);
                ens->entries[i].forest = NULL;
            }
        } else {
            if (new_count != i) {
                ens->entries[new_count] = ens->entries[i];
            }
            new_count++;
        }
    }
    ens->count = new_count;

    // Clear and rebuild hash table
    for (size_t b = 0; b < ens->hash_size; b++) {
        geif_cat_hash_node_t *node = ens->hash_buckets[b];
        while (node) {
            geif_cat_hash_node_t *next = node->next;
            free(node);
            node = next;
        }
        ens->hash_buckets[b] = NULL;
    }

    for (size_t i = 0; i < ens->count; i++) {
        uint32_t h = ensemble_hash(ens->entries[i].category);
        size_t b = (size_t)(h % ens->hash_size);
        geif_cat_hash_node_t *node = (geif_cat_hash_node_t *)malloc(sizeof(geif_cat_hash_node_t));
        if (node) {
            node->entry_idx = (uint32_t)i;
            node->next = ens->hash_buckets[b];
            ens->hash_buckets[b] = node;
        }
    }

    return GEIF_OK;
}

geif_status_t geif_ensemble_prune_age(geif_ensemble_t *ens, time_t max_age_seconds, time_t now)
{
    if (!ens) return GEIF_ERR_INVALID_ARG;
    if (max_age_seconds == 0 || ens->count == 0) return GEIF_OK;

    if (now == 0) now = time(NULL);
    time_t cutoff = (now > max_age_seconds) ? (now - max_age_seconds) : 0;

    size_t new_count = 0;
    for (size_t i = 0; i < ens->count; i++) {
        if (ens->entries[i].last_updated < cutoff) {
            if (ens->entries[i].forest) {
                geif_forest_destroy(ens->entries[i].forest);
                ens->entries[i].forest = NULL;
            }
        } else {
            if (new_count != i) {
                ens->entries[new_count] = ens->entries[i];
            }
            new_count++;
        }
    }
    ens->count = new_count;

    // Clear and rebuild hash table
    for (size_t b = 0; b < ens->hash_size; b++) {
        geif_cat_hash_node_t *node = ens->hash_buckets[b];
        while (node) {
            geif_cat_hash_node_t *next = node->next;
            free(node);
            node = next;
        }
        ens->hash_buckets[b] = NULL;
    }

    for (size_t i = 0; i < ens->count; i++) {
        uint32_t h = ensemble_hash(ens->entries[i].category);
        size_t b = (size_t)(h % ens->hash_size);
        geif_cat_hash_node_t *node = (geif_cat_hash_node_t *)malloc(sizeof(geif_cat_hash_node_t));
        if (node) {
            node->entry_idx = (uint32_t)i;
            node->next = ens->hash_buckets[b];
            ens->hash_buckets[b] = node;
        }
    }

    return GEIF_OK;
}

geif_status_t geif_ensemble_train(geif_ensemble_t *ens)
{
    if (!ens || ens->count == 0) return GEIF_ERR_EMPTY_DATASET;

    for (size_t i = 0; i < ens->count; i++) {
        geif_forest_t *forest = ens->entries[i].forest;
        if (forest && forest->pool_count > 0) {
            geif_status_t status = geif_forest_train(forest);
            if (status != GEIF_OK) {
                return status;
            }
        }
    }

    return GEIF_OK;
}

geif_status_t geif_ensemble_score_detailed(const geif_ensemble_t *ens,
                                           const char *category,
                                           const double *point,
                                           double *score_out,
                                           double *metric_depth_out,
                                           double *d_out_out)
{
    if (!ens || !point || !score_out) return GEIF_ERR_INVALID_ARG;

    const char *cat_key = (category && category[0] != '\0') ? category : "";
    geif_forest_t *forest = NULL;

    uint32_t h = ensemble_hash(cat_key);
    size_t b = (size_t)(h % ens->hash_size);
    geif_cat_hash_node_t *node = ens->hash_buckets[b];
    while (node) {
        uint32_t idx = node->entry_idx;
        if (strcmp(ens->entries[idx].category, cat_key) == 0) {
            forest = ens->entries[idx].forest;
            ((geif_ensemble_t *)ens)->entries[idx].seen_in_analysis = true;
            break;
        }
        node = node->next;
    }

    if (!forest) {
        // Unknown category unseen during training: maximum outlier
        *score_out = 1.0;
        if (metric_depth_out) *metric_depth_out = 0.0;
        if (d_out_out) *d_out_out = 999.0;
        return GEIF_ERR_INVALID_ARG;
    }

    return geif_forest_score_detailed(forest, point, score_out, metric_depth_out, d_out_out);
}

void geif_ensemble_summary(const geif_ensemble_t *ens, char *buffer, size_t buffer_size)
{
    if (!ens || !buffer || buffer_size == 0) return;

    size_t written = 0;
    int n = snprintf(buffer + written, buffer_size - written,
                     "GEIF Ensemble Diagnostic Summary:\n"
                     "  Dimensions:          %u\n"
                     "  Sub-forests:         %zu\n"
                     "  Total Input Cols:    %u\n"
                     "  Category Columns (-C): %s\n"
                     "  Label Columns (-L):  %s\n"
                     "  Include Columns (-U): %s\n"
                     "  Ignore Columns (-I):  %s\n",
                     ens->dimensions,
                     ens->count,
                     ens->total_input_cols,
                     ens->category_dims_spec[0] ? ens->category_dims_spec : "(none - global forest)",
                     ens->label_dims_spec[0] ? ens->label_dims_spec : "(none)",
                     ens->include_dims_spec[0] ? ens->include_dims_spec : "(all)",
                     ens->ignore_dims_spec[0] ? ens->ignore_dims_spec : "(none)");
    if (n > 0) written += (size_t)n;

    for (size_t i = 0; i < ens->count && written + 80 < buffer_size; i++) {
        const geif_category_entry_t *entry = &ens->entries[i];
        const geif_forest_t *f = entry->forest;
        const char *cat_label = entry->category[0] ? entry->category : "(default)";

        if (f) {
            n = snprintf(buffer + written, buffer_size - written,
                         "  - Sub-forest '%s': rows=%llu, samples=%zu, trees=%u, H_max=%.4f, updated=%ld\n",
                         cat_label,
                         (unsigned long long)entry->total_rows,
                         f->pool_count,
                         f->tree_count,
                         f->H_max,
                         (long)entry->last_updated);
        } else {
            n = snprintf(buffer + written, buffer_size - written,
                         "  - Sub-forest '%s': (empty), updated=%ld\n", cat_label, (long)entry->last_updated);
        }
        if (n > 0) written += (size_t)n;
    }
}

geif_status_t geif_forest_remove_outliers(geif_forest_t *f, uint32_t k)
{
    if (!f) return GEIF_ERR_INVALID_ARG;
    if (k == 0) return GEIF_OK;
    if (f->pool_count <= 2) return GEIF_OK;

    uint32_t d = f->dimensions;
    uint32_t removed = 0;

    while (removed < k && f->pool_count > 2) {
        int64_t max_idx = -1;
        double max_score = -1.0;

        for (size_t i = 0; i < f->pool_count; i++) {
            const double *sample = &f->sample_pool[i * d];
            double s = 0.0;
            geif_forest_score(f, sample, &s);
            if (s > max_score) {
                max_score = s;
                max_idx = (int64_t)i;
            }
        }

        if (max_idx >= 0) {
            memmove(&f->sample_pool[max_idx * d],
                    &f->sample_pool[(max_idx + 1) * d],
                    (f->pool_count - (size_t)max_idx - 1) * d * sizeof(double));
            f->pool_count--;
            removed++;
        } else {
            break;
        }
    }

    if (removed > 0) {
        return geif_forest_train(f);
    }
    return GEIF_OK;
}

geif_status_t geif_ensemble_remove_outliers(geif_ensemble_t *ens, uint32_t k)
{
    if (!ens) return GEIF_ERR_INVALID_ARG;
    if (k == 0) return GEIF_OK;

    for (size_t i = 0; i < ens->count; i++) {
        if (ens->entries[i].forest) {
            geif_status_t st = geif_forest_remove_outliers(ens->entries[i].forest, k);
            if (st != GEIF_OK) return st;
        }
    }
    return GEIF_OK;
}
