/**
 * @file test_grid.c
 * @brief Population drift visualization and test grid generator for GEIF.
 */

#include "test_grid.h"
#include "xmalloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_GRID_RESERVOIR_SAMPLES 10240

static bool cat_matches(const cat_filter_t *filter, const char *cat_name)
{
    if (!filter || !filter->active) return true;
    int reg_res = regexec(&filter->regex, cat_name ? cat_name : "", 0, NULL, 0);
    if (!filter->invert) {
        return (reg_res == 0);
    } else {
        return (reg_res != 0);
    }
}

void geif_generate_test_grid(const geif_ensemble_t *ens,
                             double test_extension_factor,
                             int test_sample_interval,
                             const cat_filter_t *filter,
                             double threshold,
                             bool threshold_is_average,
                             bool threshold_is_percentage,
                             const char *point_tmpl,
                             int decimals,
                             char list_sep,
                             uint32_t low_rgb,
                             uint32_t high_rgb,
                             const char *printf_format,
                             const char *print_dimension,
                             FILE *out_fp)
{
    if (!ens || !out_fp) return;

    for (size_t c = 0; c < ens->count; c++) {
        const char *cat_name = ens->entries[c].category;
        if (!cat_matches(filter, cat_name)) continue;

        geif_forest_t *f = ens->entries[c].forest;
        if (!f || f->dimensions == 0) continue;

        uint32_t dims = f->dimensions;
        double *len = xmalloc(dims * sizeof(double));
        int *sidx = xcalloc(dims, sizeof(int));
        double *test_dimension = xmalloc(dims * sizeof(double));
        double *prev_dimension = xmalloc(dims * sizeof(double));

        double zero_len_divisor = 0.0;
        for (uint32_t i = 0; i < dims; i++) {
            prev_dimension[i] = -999999.0;
            len[i] = f->envelope_max[i] - f->envelope_min[i];
            if (len[i] > 0.0 && 2.0 / len[i] > zero_len_divisor) {
                zero_len_divisor = 2.0 / len[i];
            }
        }
        if (zero_len_divisor < 2.0) zero_len_divisor = 2.0;

        int intervals = test_sample_interval > 0 ? test_sample_interval : 256;
        if (dims > 2 && test_sample_interval <= 0) {
            intervals = (int)pow(65536.0, 1.0 / (double)dims);
            if (intervals < 2) intervals = 2;
        }

        double cutoff = threshold;
        if (threshold_is_average) {
            cutoff = f->average_score;
        } else if (threshold_is_percentage) {
            cutoff = (f->percentage_score > 0.0) ? f->percentage_score : 0.5;
        }

        // Odometer traversal across all dimension combinations
        while (sidx[0] <= intervals) {
            for (uint32_t i = 0; i < dims; i++) {
                if (len[i] == 0.0) {
                    test_dimension[i] = (1.0 + test_extension_factor) *
                                        (2.0 * (double)sidx[i] / (double)intervals / zero_len_divisor) +
                                        (f->envelope_min[i] - ((1.0 + test_extension_factor) / zero_len_divisor));
                } else {
                    test_dimension[i] = (1.0 + test_extension_factor) *
                                        ((double)sidx[i] / (double)intervals) * len[i] +
                                        (f->envelope_min[i] - (test_extension_factor * len[i]) / 2.0);
                }
            }

            double score = 0.0;
            double H_metric = 0.0;
            double d_out = 0.0;
            geif_forest_score_detailed(f, test_dimension, &score, &H_metric, &d_out);

            if (score > cutoff) {
                bool is_outlier = (score >= cutoff);
                geif_template_context_t ctx = {
                    .score           = score,
                    .metric_depth    = H_metric,
                    .H_max           = f->H_max,
                    .d_out           = d_out,
                    .is_outlier      = is_outlier ? 1 : 0,
                    .vector          = test_dimension,
                    .vector_dim      = dims,
                    .averages        = f->averages,
                    .attr_scores     = NULL,
                    .orig_line       = NULL,
                    .category        = cat_name,
                    .label           = NULL,
                    .total_rows      = ens->entries[c].total_rows,
                    .timestamp       = ens->entries[c].last_updated,
                    .decimals        = decimals,
                    .list_separator  = list_sep,
                    .low_rgb         = low_rgb,
                    .high_rgb        = high_rgb,
                    .printf_format   = printf_format,
                    .print_dimension = print_dimension
                };

                if (point_tmpl && point_tmpl[0] != '\0') {
                    char out_buf[8192];
                    geif_format_template(out_buf, sizeof(out_buf), point_tmpl, &ctx);
                    fprintf(out_fp, "%s\n", out_buf);
                } else {
                    for (uint32_t i = 0; i < dims; i++) {
                        fprintf(out_fp, "%.*f%c", decimals, test_dimension[i], list_sep);
                    }
                    fprintf(out_fp, "%.*f%c%d%c%.*f%c%.*f\n",
                            decimals, score, list_sep,
                            is_outlier ? 1 : 0, list_sep,
                            decimals, H_metric, list_sep,
                            decimals, d_out);
                }
            }

            // Advance odometer
            for (int i = (int)dims - 1; i >= 0; i--) {
                if (i > 0 && sidx[i] == intervals) {
                    sidx[i] = 0;
                    sidx[i - 1]++;
                } else {
                    if (i == (int)dims - 1) sidx[i]++;
                    break;
                }
            }
        }

        // Output actual data samples from reservoir with score 0.0
        uint32_t sample_count = f->pool_count < MAX_GRID_RESERVOIR_SAMPLES ? f->pool_count : MAX_GRID_RESERVOIR_SAMPLES;
        for (uint32_t s = 0; s < sample_count; s++) {
            const double *sample = &f->sample_pool[s * dims];
            geif_template_context_t ctx = {
                .score           = 0.0,
                .metric_depth    = f->H_max,
                .H_max           = f->H_max,
                .d_out           = 0.0,
                .is_outlier      = 0,
                .vector          = sample,
                .vector_dim      = dims,
                .averages        = f->averages,
                .attr_scores     = NULL,
                .orig_line       = NULL,
                .category        = cat_name,
                .label           = NULL,
                .total_rows      = ens->entries[c].total_rows,
                .timestamp       = ens->entries[c].last_updated,
                .decimals        = decimals,
                .list_separator  = list_sep,
                .low_rgb         = low_rgb,
                .high_rgb        = high_rgb,
                .printf_format   = printf_format,
                .print_dimension = print_dimension
            };

            if (point_tmpl && point_tmpl[0] != '\0') {
                char out_buf[8192];
                geif_format_template(out_buf, sizeof(out_buf), point_tmpl, &ctx);
                fprintf(out_fp, "%s\n", out_buf);
            } else {
                for (uint32_t i = 0; i < dims; i++) {
                    fprintf(out_fp, "%.*f%c", decimals, sample[i], list_sep);
                }
                fprintf(out_fp, "%.*f%c0%c%.*f%c0.0\n",
                        decimals, 0.0, list_sep, list_sep,
                        decimals, f->H_max, list_sep);
            }
        }

        free(len);
        free(sidx);
        free(test_dimension);
        free(prev_dimension);
    }
}
