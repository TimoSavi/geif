/**
 * @file tree_common.c
 * @brief Implementation of common tree memory allocation, traversal, and calibration.
 */

#include "tree_common.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void init_dimension_scales(geif_forest_t *f)
{
    uint32_t d = f->dimensions;
    uint32_t psi = f->config.samples_per_tree;
    if (psi == 0) psi = 256;

    if (f->pool_count > 0 && f->sample_pool) {
        for (uint32_t j = 0; j < d; j++) {
            f->envelope_min[j] = f->sample_pool[j];
            f->envelope_max[j] = f->sample_pool[j];
        }
        for (size_t i = 1; i < f->pool_count; i++) {
            const double *pt = &f->sample_pool[i * d];
            for (uint32_t j = 0; j < d; j++) {
                if (pt[j] < f->envelope_min[j]) f->envelope_min[j] = pt[j];
                if (pt[j] > f->envelope_max[j]) f->envelope_max[j] = pt[j];
            }
        }
    }

    double max_span = 0.0;
    int32_t best_dim = -1;

    for (uint32_t j = 0; j < d; j++) {
        f->envelope_span[j] = f->envelope_max[j] - f->envelope_min[j];
        if (f->envelope_span[j] >= 1e-9) {
            f->dim_active[j] = 1;
            f->effective_span[j] = f->envelope_span[j];
            if (f->envelope_span[j] > max_span) {
                max_span = f->envelope_span[j];
                best_dim = (int32_t)j;
            }
        } else {
            f->dim_active[j] = 0;
            f->effective_span[j] = 0.0;
        }
    }

    if (best_dim < 0) best_dim = 0;
    if (max_span <= 0.0) max_span = 1.0;
    f->scale_range_idx = best_dim;

    if (!f->scaled_pool && f->pool_capacity > 0) {
        f->scaled_pool = (double *)malloc(f->pool_capacity * d * sizeof(double));
    }

    if (f->scaled_pool && f->pool_count > 0) {
        double target_range = max_span;
        double scale_min = f->envelope_min[best_dim];
        for (size_t i = 0; i < f->pool_count; i++) {
            for (uint32_t j = 0; j < d; j++) {
                f->scaled_pool[i * d + j] = geif_scale_value(f->sample_pool[i * d + j],
                                                             target_range, scale_min,
                                                             f->envelope_min[j], f->envelope_max[j]);
            }
        }
    }

    uint32_t n_eff = (f->pool_count < psi) ? (uint32_t)f->pool_count : psi;
    if (n_eff < 2) n_eff = 2;
    f->avg_sample_dist = sqrt(GEIF_DIST_AVG((double)d)) * (max_span / pow((double)n_eff, 1.0 / (double)d));

    geif_init_c_cache();
    f->c_factor = geif_c((double)n_eff);
    f->delta_nominal = f->avg_sample_dist;
}

int32_t allocate_node(geif_tree_t *tree)
{
    if (tree->node_count >= tree->node_capacity) {
        size_t new_cap = (tree->node_capacity == 0) ? 64 : tree->node_capacity * 2;
        geif_node_t *new_nodes = (geif_node_t *)realloc(tree->nodes, new_cap * sizeof(geif_node_t));
        if (!new_nodes) return -1;
        tree->nodes = new_nodes;
        tree->node_capacity = new_cap;
    }
    int32_t idx = (int32_t)tree->node_count++;
    memset(&tree->nodes[idx], 0, sizeof(geif_node_t));
    tree->nodes[idx].left_child = -1;
    tree->nodes[idx].right_child = -1;
    return idx;
}

uint32_t append_normal(geif_tree_t *tree, const double *normal, uint32_t d)
{
    size_t needed = (tree->node_count + 1) * d;
    if (needed > tree->normals_capacity) {
        size_t new_cap = (tree->normals_capacity == 0) ? 64 * d : tree->normals_capacity * 2;
        if (new_cap < needed) new_cap = needed;
        double *new_pool = (double *)realloc(tree->normals_pool, new_cap * sizeof(double));
        if (!new_pool) return 0;
        tree->normals_pool = new_pool;
        tree->normals_capacity = new_cap;
    }
    uint32_t offset = (uint32_t)((tree->node_count - 1) * d);
    memcpy(&tree->normals_pool[offset], normal, d * sizeof(double));
    return offset;
}

uint32_t append_leaf_samples(geif_tree_t *tree, const uint32_t *samples, size_t count)
{
    if (count == 0) return 0;
    size_t needed = tree->leaf_samples_count + count;
    if (needed > tree->leaf_samples_capacity) {
        size_t new_cap = (tree->leaf_samples_capacity == 0) ? 128 : tree->leaf_samples_capacity * 2;
        if (new_cap < needed) new_cap = needed;
        uint32_t *new_arr = (uint32_t *)realloc(tree->leaf_samples, new_cap * sizeof(uint32_t));
        if (!new_arr) return 0;
        tree->leaf_samples = new_arr;
        tree->leaf_samples_capacity = new_cap;
    }
    uint32_t offset = (uint32_t)tree->leaf_samples_count;
    memcpy(&tree->leaf_samples[offset], samples, count * sizeof(uint32_t));
    tree->leaf_samples_count += count;
    return offset;
}

double evaluate_tree(const geif_forest_t *f,
                    const geif_tree_t *tree,
                    const double *scaled_point)
{
    if (tree->node_count == 0) {
        return 0.0;
    }

    int32_t curr = 0;
    double depth = 0.0;
    uint32_t d = f->dimensions;

    while (curr >= 0 && curr < (int32_t)tree->node_count) {
        const geif_node_t *node = &tree->nodes[curr];

        // Leaf node reached
        if (node->left_child == -1 && node->right_child == -1) {
            double leaf_c = 0.0;
            if (f->avg_sample_dist > 0.0 && node->sample_count > 0 && tree->leaf_samples &&
                (node->leaf_sample_offset + (size_t)node->sample_count <= tree->leaf_samples_count)) {
                double min_dist_sq = 1e300;
                const uint32_t *leaf_s = &tree->leaf_samples[node->leaf_sample_offset];
                for (int32_t i = 0; i < node->sample_count; i++) {
                    uint32_t s_idx = leaf_s[i];
                    if (s_idx < f->pool_count) {
                        const double *sample = (f->scaled_pool) ? &f->scaled_pool[s_idx * d]
                                                               : &f->sample_pool[s_idx * d];
                        double dist_sq = geif_dist_sq(scaled_point, sample, d);
                        if (dist_sq < min_dist_sq) {
                            min_dist_sq = dist_sq;
                        }
                    }
                }
                if (min_dist_sq < 1e299) {
                    double rel_dist = (sqrt(min_dist_sq) / f->avg_sample_dist) + MIN_REL_DIST;
                    double adjusted_n = (double)node->sample_count / rel_dist;
                    leaf_c = geif_c(adjusted_n);
                } else if (node->sample_count > 1) {
                    leaf_c = geif_c((double)node->sample_count);
                }
            } else if (node->sample_count > 1) {
                leaf_c = geif_c((double)node->sample_count);
            }
            return depth + leaf_c;
        }

        // Internal split node
        const double *normal = &tree->normals_pool[node->normal_offset];
        double dot_val = geif_dot(scaled_point, normal, d);

        if (dot_val < node->pdotn) {
            if (node->left_child == -1) return depth;
            curr = node->left_child;
        } else {
            if (node->right_child == -1) return depth;
            curr = node->right_child;
        }
        depth += 1.0;
    }

    return depth;
}

double geif_tree_evaluate_metric_depth(const geif_forest_t *f,
                                      const double *point,
                                      double *d_out_out)
{
    if (!f || !point || f->tree_count == 0) {
        if (d_out_out) *d_out_out = 0.0;
        return 0.0;
    }

    uint32_t d = f->dimensions;
    double stack_buf[64];
    double *scaled_point = (d <= 64) ? stack_buf : (double *)malloc(d * sizeof(double));
    if (!scaled_point) {
        if (d_out_out) *d_out_out = 0.0;
        return 0.0;
    }

    if (f->scale_range_idx >= 0 && f->envelope_span && f->envelope_min && f->envelope_max) {
        double target_range = f->envelope_span[f->scale_range_idx];
        double scale_min = f->envelope_min[f->scale_range_idx];
        for (uint32_t j = 0; j < d; j++) {
            scaled_point[j] = geif_scale_value(point[j], target_range, scale_min,
                                               f->envelope_min[j], f->envelope_max[j]);
        }
    } else {
        memcpy(scaled_point, point, d * sizeof(double));
    }

    double sum_H = 0.0;
    for (uint32_t t = 0; t < f->tree_count; t++) {
        sum_H += evaluate_tree(f, &f->trees[t], scaled_point);
    }
    double H_avg = sum_H / (double)f->tree_count;

    // Outer space distance in scaled space
    double dist_out_sq = 0.0;
    double target_range = (f->scale_range_idx >= 0 && f->envelope_span) ? f->envelope_span[f->scale_range_idx] : 1.0;
    if (f->envelope_min && f->envelope_max) {
        for (uint32_t j = 0; j < d; j++) {
            double d_low = f->envelope_min[j] - point[j];
            double d_high = point[j] - f->envelope_max[j];
            double delta = 0.0;
            if (d_low > 0.0) delta = d_low;
            else if (d_high > 0.0) delta = d_high;

            if (delta > 0.0 && f->envelope_span && f->envelope_span[j] > 1e-12) {
                double norm = delta * (target_range / f->envelope_span[j]);
                dist_out_sq += norm * norm;
            }
        }
    }

    double d_out_scaled = sqrt(dist_out_sq);
    if (scaled_point != stack_buf) free(scaled_point);
    if (d_out_out) *d_out_out = d_out_scaled;
    return H_avg;
}

geif_status_t geif_tree_score_point(const geif_forest_t *f,
                                   const double *point,
                                   double *score_out,
                                   double *metric_depth_out,
                                   double *d_out_out)
{
    if (!f || !point || !score_out) {
        return GEIF_ERR_INVALID_ARG;
    }

    double d_out = 0.0;
    double H_final = geif_tree_evaluate_metric_depth(f, point, &d_out);

    if (metric_depth_out) *metric_depth_out = H_final;
    if (d_out_out) *d_out_out = d_out;

    double psi = (f->config.samples_per_tree > 0) ? (double)f->config.samples_per_tree : 256.0;
    double c_psi = (f->c_factor > 0.0) ? f->c_factor : geif_c(psi);
    if (c_psi <= 0.0) c_psi = 1.0;

    // Standard Isolation Forest score s = 1.0 / 2^(H / c)
    double score = 1.0 / pow(2.0, H_final / c_psi);

    // Outer space exponential attenuation from CEIF
    if (d_out > 0.0) {
        double target_range = (f->scale_range_idx >= 0 && f->envelope_span) ? f->envelope_span[f->scale_range_idx] : 1.0;
        double d_norm = (target_range > 1e-12) ? (d_out / target_range) : d_out;
        score = 1.0 - (1.0 - score) * exp(-GEIF_OUTER_DECAY_RATE * d_norm);
    }
    if (score >= 1.0) score = 1.0 - 1e-6;

    if (f->scale_score) {
        double max_s = (f->max_score > 0.0) ? f->max_score : 1.0;
        double min_s = f->min_score;
        double scaled_score = score;
        if (max_s > min_s) {
            scaled_score = (score - min_s) / (max_s - min_s);
            if (scaled_score < 0.0) scaled_score = 0.0;
            if (scaled_score >= 1.0) scaled_score = 1.0 - 1e-6;
        }
        *score_out = scaled_score;
    } else {
        *score_out = score;
    }
    return GEIF_OK;
}

void geif_tree_find_max_height(const geif_forest_t *f,
                               const geif_tree_t *t,
                               int32_t node_idx,
                               double depth,
                               double *max_h)
{
    if (node_idx < 0 || node_idx >= (int32_t)t->node_count) return;

    const geif_node_t *node = &t->nodes[node_idx];
    if (node->left_child == -1 && node->right_child == -1) {
        double leaf_c = 0.0;
        if (f->avg_sample_dist > 0.0 && node->sample_count > 0 && t->leaf_samples) {
            double rel_dist = MIN_REL_DIST;
            double adjusted_n = (double)node->sample_count / rel_dist;
            leaf_c = geif_c(adjusted_n);
        } else if (node->sample_count > 1) {
            leaf_c = geif_c((double)node->sample_count);
        }
        double h = depth + leaf_c;
        if (h > *max_h) {
            *max_h = h;
        }
        return;
    }

    if (node->left_child != -1) {
        geif_tree_find_max_height(f, t, node->left_child, depth + 1.0, max_h);
    }
    if (node->right_child != -1) {
        geif_tree_find_max_height(f, t, node->right_child, depth + 1.0, max_h);
    }
}

void geif_tree_calibrate(geif_forest_t *f)
{
    f->min_score = 0.0;
    f->max_score = 1.0;

    double sum_max_h = 0.0;
    for (uint32_t t = 0; t < f->tree_count; t++) {
        double max_h = 0.0;
        geif_tree_find_max_height(f, &f->trees[t], 0, 0.0, &max_h);
        sum_max_h += max_h;
    }

    double avg_max_h = (f->tree_count > 0) ? (sum_max_h / (double)f->tree_count) : 0.0;
    if (avg_max_h < 1.0) avg_max_h = 1.0;

    double psi_d = (f->config.samples_per_tree > 0) ? (double)f->config.samples_per_tree : 256.0;
    double c_psi = (f->c_factor > 0.0) ? f->c_factor : geif_c(psi_d);
    if (c_psi <= 0.0) c_psi = 1.0;

    f->H_train_max = avg_max_h;
    f->H_max = avg_max_h;
    f->min_score = pow(2.0, -avg_max_h / c_psi);
    if (f->min_score < 0.0) f->min_score = 0.0;
    if (f->min_score > 1.0) f->min_score = 1.0;
    f->max_score = 1.0;

    double sum_score = 0.0;
    for (size_t i = 0; i < f->pool_count; i++) {
        const double *pt = &f->sample_pool[i * f->dimensions];
        double s = 0.0;
        geif_tree_score_point(f, pt, &s, NULL, NULL);
        sum_score += s;
    }
    f->average_score = (f->pool_count > 0) ? (sum_score / (double)f->pool_count) : 0.5;

    if (f->averages && f->pool_count > 0 && f->sample_pool) {
        for (uint32_t j = 0; j < f->dimensions; j++) f->averages[j] = 0.0;
        for (size_t i = 0; i < f->pool_count; i++) {
            const double *pt = &f->sample_pool[i * f->dimensions];
            for (uint32_t j = 0; j < f->dimensions; j++) {
                f->averages[j] += pt[j];
            }
        }
        for (uint32_t j = 0; j < f->dimensions; j++) {
            f->averages[j] /= (double)f->pool_count;
        }
    }
}
