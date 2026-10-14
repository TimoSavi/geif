/**
 * @file main.c
 * @brief GEIF CLI frontend for training, inference, and model management.
 */

#include "geif/geif.h"
#include "xmalloc.h"
#include "columns.h"
#include "template.h"
#include "rcfile.h"
#include "test_grid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <math.h>
#include <regex.h>
#include <time.h>

#define GEIF_VERSION "1.1.0"

/**
 * @brief Displays command-line syntax and comprehensive option reference for GEIF CLI.
 *
 * @param[in] prog Executable program name (argv[0]).
 */
static void print_usage(const char *prog)
{
    printf("GEIF - Geometric Extended Isolation Forest (v%s)\n\n", GEIF_VERSION);
    printf("Usage:\n");
    printf("  Train a model:     %s -l <train.csv> -w <model.json> [-t trees] [-s samples] [-f sep] [-H] [-I range] [-U range] [-L range] [-C range] [-R min_rows]\n", prog);
    printf("  Score streaming:   %s -r <model.json> -a <test.csv> [-o <out.csv>] [-T threshold] [-F regex] [-N tmpl] [-M tmpl] [-p tmpl] [-S]\n", prog);
    printf("  Categorize:        %s -r <model.json> -c <test.csv> [-o <out.csv>] [-O threshold] [-F regex] [-p tmpl]\n", prog);
    printf("  Inspect model:     %s -r <model.json> -q\n\n", prog);
    printf("Options:\n");
    printf("  -l <file>      Train forest from input CSV file (use '-' for stdin)\n");
    printf("  -a <file>      Analyze / score samples from input CSV (use '-' for stdin)\n");
    printf("  -c <file>      Categorize / classify samples against all categories and assign best match\n");
    printf("  -w <file>      Save trained model to JSON file (use '-' for stdout)\n");
    printf("  -r <file>      Load trained model from JSON file (use '-' for stdin)\n");
    printf("  -z <file>      In-place model update (load from file, update with new data, save back)\n");
    printf("  -o <file>      Output file for scores (default: stdout, '-' for stdout)\n");
    printf("  -O <thresh>    Outlier threshold: float [0..1] (scaled by default), 'average', or percentage (e.g. '80%%')\n");
    printf("  -B, --algo <s> Algorithm engine: bubble (default), ceif, exemplar, voronoi\n");
    printf("  -T [margin]    Generate synthetic test grid for population drift visualization (margin: e.g. 0.1)\n");
    printf("  -k             Prune most extreme outlier from model reservoir and recalibrate (repeatable)\n");
    printf("  -g <file>      Configuration / RC file (overrides ~/.geifrc and ~/.ceifrc)\n");
    printf("  -t <int>       Number of trees in forest (default: 100)\n");
    printf("  -i <int>       Test grid sample intervals (default: 256) or tree count alias\n");
    printf("  -s <int>       Number of samples per tree / exemplar count (default: 256)\n");
    printf("  -m <fmt|int>   Dimension format string (e.g. \"%%'.0f\") or max depth cap\n");
    printf("  -f <char>      Input field delimiter (default: ',')\n");
    printf("  -e <char>      List separator for output / field delimiter fallback\n");
    printf("  -H, --header   Skip header line in input CSV\n");
    printf("      --no-header Do not skip header line (override model/rc setting)\n");
    printf("  -I <range>     Ignore column indices/ranges from features (e.g. '12' or '1,3,5')\n");
    printf("  -U <range>     Use/include only specified column indices/ranges (e.g. '2-10')\n");
    printf("  -L <range>     Label column indices/ranges (excluded from features, kept as label)\n");
    printf("  -C <range>     Category column indices/ranges (e.g. '12' or '2-4')\n");
    printf("  -F <filter>    Filter out categories matching regex (prefix '-v ' to invert / keep only matching)\n");
    printf("  -R <int>       Minimum row count required to retain a category\n");
    printf("  -D <interval>  Drop / prune categories older than interval (e.g. '30d', '7d', '24h', '3600s')\n");
    printf("  -N [tmpl]      Output template for NEW / unseen categories during learning/analysis (default: '%%v')\n");
    printf("  -M [tmpl]      Output template for MISSED categories (default: '%%C')\n");
    printf("  -p <tmpl>      Output template for scored rows (%%s=score, %%l=label, %%c=cat, %%m=metric/dims, %%d=vector, %%e=attrib, %%a=avg, %%t=time, %%u=epoch)\n");
    printf("  -v [tmpl]      Output template for inlier/average rows, or verbose flag\n");
    printf("  -d <int>       Floating point decimal precision (default: 6)\n");
    printf("  -j <tmpl>      Per-dimension expansion template for %%m (%%d=val, %%a=avg, %%e=attrib, %%i=index)\n");
    printf("  -W             Enable metric auto-scaling (accepted for backward compatibility)\n");
    printf("  -A, --aggregate Aggregate sample values by category rather than storing/scoring raw rows\n");
    printf("  -S             Silent / outliers only (suppress inliers from output)\n");
    printf("  -q             Print model summary / diagnostics and exit\n");
    printf("  -h             Show this help message and exit\n");
}

/**
 * @brief Tokenizes a delimited text line with quote awareness and whitespace trimming.
 *
 * Strips carriage returns and newlines, parses quoted strings without splitting on interior
 * delimiters, and trims surrounding whitespace. Modifies the input line buffer in-place.
 *
 * @param[in,out] line       Input character buffer to tokenize.
 * @param[in]     delim      Delimiter character (e.g. ',', '\t', ';').
 * @param[out]    tokens     Array of string pointers to receive token starts.
 * @param[in]     max_tokens Maximum token capacity of tokens array.
 * @return Number of tokens parsed.
 */
static uint32_t tokenize_line(char *line, char delim, char **tokens, uint32_t max_tokens)
{
    uint32_t count = 0;
    char *p = line;

    // Strip trailing \r\n
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\r' || line[len - 1] == '\n')) {
        line[--len] = '\0';
    }

    while (*p != '\0' && count < max_tokens) {
        if (delim != ' ' && delim != '\t') {
            while (*p == ' ' || *p == '\t') p++;
        }

        char *token_start = p;
        if (*p == '"') {
            token_start = ++p;
            while (*p != '\0' && *p != '"') p++;
            if (*p == '"') {
                *p++ = '\0';
                while (*p != '\0' && *p != delim) p++;
                if (*p == delim) p++;
            }
        } else {
            while (*p != '\0' && *p != delim) p++;
            if (*p == delim) {
                *p++ = '\0';
            }
        }

        if (delim != ' ' && delim != '\t') {
            char *end = token_start + strlen(token_start) - 1;
            while (end >= token_start && (*end == ' ' || *end == '\t')) {
                *end-- = '\0';
            }
        }

        tokens[count++] = token_start;
    }

    return count;
}

/**
 * @brief Parses human-readable duration strings (e.g. "30d", "24h", "60m") into seconds.
 *
 * Supports units: 'y' (years), 'm'/'M' (months or minutes depending on context),
 * 'w' (weeks), 'd' (days), 'h' (hours), 's' (seconds). Defaults to days if unit is omitted.
 *
 * @param[in] s Duration string (e.g. "7d").
 * @return Duration in seconds, or 0 on error.
 */
static time_t parse_delete_interval(const char *s)
{
    if (!s || s[0] == '\0') return 0;
    size_t len = strlen(s);
    time_t value = (time_t)atol(s);
    if (value <= 0) return 0;

    char unit = s[len - 1];
    switch (unit) {
    case 'Y':
    case 'y':
        value *= 31556926; // 365.2422 days
        break;
    case 'M':
        value *= 2629743;  // 30.4368 days
        break;
    case 'W':
    case 'w':
        value *= 604800;   // 7 days
        break;
    case 'D':
    case 'd':
        value *= 86400;    // 1 day
        break;
    case 'h':
    case 'H':
        value *= 3600;     // 1 hour
        break;
    case 'm':
        value *= 60;       // 1 minute
        break;
    case 's':
    case 'S':
        break;
    default:
        if (unit >= '0' && unit <= '9') {
            value *= 86400; // default to days (as in CEIF)
        } else {
            fprintf(stderr, "geif: error: invalid time format in -D '%s'\n", s);
            return 0;
        }
        break;
    }
    return value;
}

/**
 * @brief Evaluates an incoming CSV record against an ensemble during streaming analysis (-a).
 *
 * Extracts category and label, verifies category filters, routes to the corresponding
 * sub-forest (or flags as an unseen category), computes anomaly score and metric depth,
 * determines outlier status against threshold, and formats output via templates.
 *
 * @param[in,out] ensemble                 Category ensemble model.
 * @param[in]     col_cfg                  Column mapping configuration.
 * @param[in]     cat_filter               Active category filter rules (-F).
 * @param[in]     tokens                   Parsed text fields of current row.
 * @param[in]     n_tok                    Token count.
 * @param[in]     orig_line                Original unparsed text line.
 * @param[in]     line_num                 1-based row counter.
 * @param[in]     list_sep                 Output list separator character (-e).
 * @param[in]     cat_sep                  Category delimiter character (-C).
 * @param[in]     threshold                Outlier decision boundary [0, 1].
 * @param[in]     threshold_is_average     True if threshold is calibrated to category mean score.
 * @param[in]     threshold_is_percentage  True if threshold is percentile-based.
 * @param[in]     silent_outliers          True to suppress nominal inlier rows (-S).
 * @param[in]     point_tmpl               Output format template for scored points (-p).
 * @param[in]     average_tmpl             Output format template for nominal inliers (-v).
 * @param[in]     new_cat_tmpl             Output format template for unseen categories (-N).
 * @param[in]     decimals                 Floating point precision (-d).
 * @param[in]     printf_format            Dimension format override string (-m).
 * @param[in]     print_dimension          Per-dimension template expansion (-j).
 * @param[in]     low_rgb                  Low-score RGB color hex.
 * @param[in]     high_rgb                 High-score RGB color hex.
 * @param[in,out] vec                      Scratch buffer for feature vector.
 * @param[in]     dims                     Number of feature dimensions.
 * @param[in,out] out_fp                   Output file stream.
 * @param[in,out] analyzed                 Cumulative analyzed rows counter.
 * @param[in,out] total_outliers           Cumulative outliers detected counter.
 */
static void process_scoring_row(geif_ensemble_t *ensemble,
                                const geif_column_config_t *col_cfg,
                                const cat_filter_t *cat_filter,
                                char **tokens,
                                uint32_t n_tok,
                                const char *orig_line,
                                uint64_t line_num,
                                char list_sep,
                                char cat_sep,
                                char label_sep,
                                double threshold,
                                bool threshold_is_average,
                                bool threshold_is_percentage,
                                bool silent_outliers,
                                const char *point_tmpl,
                                const char *average_tmpl,
                                const char *new_cat_tmpl,
                                int decimals,
                                const char *printf_format,
                                const char *print_dimension,
                                uint32_t low_rgb,
                                uint32_t high_rgb,
                                double *vec,
                                uint32_t dims,
                                FILE *out_fp,
                                uint64_t *analyzed,
                                uint64_t *total_outliers)
{
    (void)average_tmpl;
    char cat_buf[256];
    char label_buf[256];
    geif_extract_category(col_cfg, tokens, n_tok, cat_sep, cat_buf, sizeof(cat_buf));
    geif_extract_label(col_cfg, tokens, n_tok, label_sep, label_buf, sizeof(label_buf));

    // Category filter check (-F)
    if (!geif_cat_filter_allows(cat_filter, cat_buf)) {
        return; // Filtered out
    }

    geif_forest_t *sub_forest = geif_ensemble_find(ensemble, cat_buf);
    if (!sub_forest) {
        // Unseen / NEW category!
        (*analyzed)++;
        (*total_outliers)++;

        if (new_cat_tmpl) {
            bool has_vec = geif_extract_features(col_cfg, tokens, n_tok, vec);
            geif_template_context_t ctx = {
                .orig_line        = orig_line,
                .label            = label_buf,
                .category         = cat_buf,
                .input_category   = cat_buf,
                .score            = 1.0,
                .test_average_score = 1.0,
                .metric_depth     = 0.0,
                .d_out            = 999.0,
                .H_max            = 0.0,
                .is_outlier       = 1,
                .timestamp        = time(NULL),
                .total_rows       = 0,
                .analyzed_rows    = *analyzed,
                .high_analyzed_rows = 1,
                .row_idx          = line_num,
                .vector           = has_vec ? vec : NULL,
                .averages         = NULL,
                .attr_scores      = NULL,
                .vector_dim       = has_vec ? dims : 0,
                .raw_values       = tokens,
                .raw_value_count  = n_tok,
                .list_separator   = list_sep,
                .category_sep     = cat_sep,
                .label_sep        = label_sep,
                .decimals         = decimals,
                .printf_format    = printf_format,
                .print_dimension  = print_dimension,
                .low_rgb          = low_rgb,
                .high_rgb         = high_rgb,
                .is_summary       = false
            };
            char out_buf[8192];
            geif_format_template(out_buf, sizeof(out_buf), new_cat_tmpl, &ctx);
            fprintf(out_fp, "%s\n", out_buf);
        } else if (!point_tmpl && !silent_outliers) {
            fprintf(out_fp, "%s%c%.*f%c%d%c%.*f%c%.*f\n",
                    orig_line, list_sep, decimals, 1.0, list_sep,
                    1, list_sep, decimals, 0.0, list_sep, decimals, 999.0);
        }
        return;
    }

    // Known category: extract features and score (or accumulate if aggregate mode -A)
    if (geif_extract_features(col_cfg, tokens, n_tok, vec)) {
        if (ensemble->aggregate) {
            geif_category_entry_t *matched_entry = geif_ensemble_get_entry(ensemble, cat_buf);
            if (matched_entry) {
                matched_entry->seen_in_analysis = true;
                if (!matched_entry->summary) {
                    matched_entry->summary = (double *)xcalloc(dims, sizeof(double));
                }
                for (uint32_t j = 0; j < dims; j++) {
                    matched_entry->summary[j] += vec[j];
                }
            }
            return;
        }

        double score = 0.0, H_metric = 0.0, d_out = 0.0;
        geif_ensemble_score_detailed(ensemble, cat_buf, vec, &score, &H_metric, &d_out);

        double eff_threshold = threshold;
        if (threshold_is_average) {
            eff_threshold = (sub_forest->average_score > 0.0) ? sub_forest->average_score : 0.5;
        } else if (threshold_is_percentage) {
            eff_threshold = (sub_forest->percentage_score > 0.0) ? sub_forest->percentage_score : 0.5;
        }

        (*analyzed)++;
        bool is_outlier = (score > eff_threshold);
        if (is_outlier) (*total_outliers)++;

        geif_category_entry_t *matched_entry = geif_ensemble_get_entry(ensemble, cat_buf);
        time_t cat_ts = (matched_entry && matched_entry->last_updated > 0) ? matched_entry->last_updated : time(NULL);

        if (matched_entry) {
            matched_entry->analyzed_rows++;
            matched_entry->sum_score += score;
            if (is_outlier) {
                matched_entry->high_analyzed_rows++;
            }
        }

        double *averages = NULL;
        double *attr_scores = NULL;
        if (dims > 0) {
            averages = (double *)malloc(dims * sizeof(double));
            attr_scores = (double *)malloc(dims * sizeof(double));
            if (averages) geif_forest_get_averages(sub_forest, averages);
            if (attr_scores) geif_forest_dimension_attribution(sub_forest, vec, attr_scores);
        }

        geif_template_context_t ctx = {
            .orig_line        = orig_line,
            .label            = label_buf,
            .category         = cat_buf,
            .input_category   = cat_buf,
            .score            = score,
            .test_average_score = (matched_entry && matched_entry->analyzed_rows > 0)
                                 ? (matched_entry->sum_score / (double)matched_entry->analyzed_rows)
                                 : score,
            .metric_depth     = H_metric,
            .d_out            = d_out,
            .H_max            = sub_forest ? sub_forest->H_max : 0.0,
            .is_outlier       = is_outlier ? 1 : 0,
            .timestamp        = cat_ts,
            .total_rows       = sub_forest ? sub_forest->total_rows_seen : 0,
            .analyzed_rows    = matched_entry ? matched_entry->analyzed_rows : *analyzed,
            .high_analyzed_rows = matched_entry ? matched_entry->high_analyzed_rows : (is_outlier ? 1 : 0),
            .row_idx          = line_num,
            .vector           = vec,
            .averages         = averages,
            .attr_scores      = attr_scores,
            .vector_dim       = dims,
            .raw_values       = tokens,
            .raw_value_count  = n_tok,
            .list_separator   = list_sep,
            .category_sep     = cat_sep,
            .label_sep        = label_sep,
            .decimals         = decimals,
            .printf_format    = printf_format,
            .print_dimension  = print_dimension,
            .low_rgb          = low_rgb,
            .high_rgb         = high_rgb,
            .is_summary       = false
        };

        if (point_tmpl) {
            if (is_outlier) {
                char out_buf[8192];
                geif_format_template(out_buf, sizeof(out_buf), point_tmpl, &ctx);
                fprintf(out_fp, "%s\n", out_buf);
            }
        } else if (!silent_outliers || is_outlier) {
            fprintf(out_fp, "%s%c%.6f%c%d%c%.6f%c%.6f\n",
                    orig_line, list_sep, score, list_sep,
                    is_outlier ? 1 : 0, list_sep, H_metric, list_sep, d_out);
        }

        if (averages) free(averages);
        if (attr_scores) free(attr_scores);
    }
}

typedef struct {
    char category[256];
    char label[256];
    double *summary;
} agg_cat_bucket_t;

static void score_and_print_categorize_vec(geif_ensemble_t *ensemble,
                                           const cat_filter_t *cat_filter,
                                           const char *in_cat,
                                           const char *in_label,
                                           const char *orig_line,
                                           char **tokens,
                                           uint32_t n_tok,
                                           uint64_t line_num,
                                           char list_sep,
                                           char cat_sep,
                                           char label_sep,
                                           double threshold,
                                           bool threshold_is_average,
                                           bool threshold_is_percentage,
                                           bool score_limit_given,
                                           const char *point_tmpl,
                                           int decimals,
                                           const char *printf_format,
                                           const char *print_dimension,
                                           uint32_t low_rgb,
                                           uint32_t high_rgb,
                                           const double *vec,
                                           uint32_t dims,
                                           FILE *out_fp,
                                           uint64_t *analyzed)
{
    double min_score = 999.0;
    int best_idx = -1;
    double best_metric_depth = 0.0;
    double best_d_out = 0.0;

    for (size_t i = 0; i < ensemble->count; i++) {
        const char *cat_name = ensemble->entries[i].category;
        if (!geif_cat_filter_allows(cat_filter, cat_name)) {
            continue;
        }
        geif_forest_t *sub_forest = ensemble->entries[i].forest;
        if (!sub_forest) continue;

        double s = 0.0, md = 0.0, dout = 0.0;
        geif_status_t st = geif_forest_score_detailed(sub_forest, vec, &s, &md, &dout);
        if (st != GEIF_OK) continue;

        if (best_idx == -1 || s < min_score) {
            min_score = s;
            best_idx = (int)i;
            best_metric_depth = md;
            best_d_out = dout;
        }
    }

    if (best_idx < 0) return;

    geif_forest_t *best_forest = ensemble->entries[best_idx].forest;
    const char *best_cat = ensemble->entries[best_idx].category;

    double eff_thresh = 1.0;
    if (score_limit_given) {
        if (threshold_is_percentage) {
            eff_thresh = (best_forest->percentage_score > 0.0) ? best_forest->percentage_score : 0.5;
        } else if (threshold_is_average) {
            eff_thresh = (best_forest->average_score > 0.0) ? best_forest->average_score : 0.5;
        } else {
            eff_thresh = threshold;
        }

        if (min_score > eff_thresh) {
            return; // Outlier suppression under -O
        }
    }

    (*analyzed)++;
    ensemble->entries[best_idx].total_rows++;

    double *averages = NULL;
    double *attr_scores = NULL;
    if (dims > 0) {
        averages = (double *)malloc(dims * sizeof(double));
        attr_scores = (double *)malloc(dims * sizeof(double));
        if (averages) geif_forest_get_averages(best_forest, averages);
        if (attr_scores) geif_forest_dimension_attribution(best_forest, vec, attr_scores);
    }

    geif_template_context_t ctx = {
        .orig_line        = orig_line,
        .label            = in_label,
        .category         = best_cat,
        .input_category   = in_cat,
        .score            = min_score,
        .test_average_score = min_score,
        .metric_depth     = best_metric_depth,
        .d_out            = best_d_out,
        .H_max            = best_forest ? best_forest->H_max : 0.0,
        .is_outlier       = (score_limit_given && min_score > eff_thresh) ? 1 : 0,
        .timestamp        = time(NULL),
        .total_rows       = best_forest ? best_forest->total_rows_seen : 0,
        .analyzed_rows    = *analyzed,
        .high_analyzed_rows = 0,
        .row_idx          = line_num,
        .vector           = vec,
        .averages         = averages,
        .attr_scores      = attr_scores,
        .vector_dim       = dims,
        .raw_values       = tokens,
        .raw_value_count  = n_tok,
        .list_separator   = list_sep,
        .category_sep     = cat_sep,
        .label_sep        = label_sep,
        .decimals         = decimals,
        .printf_format    = printf_format,
        .print_dimension  = print_dimension,
        .low_rgb          = low_rgb,
        .high_rgb         = high_rgb,
        .is_summary       = false
    };

    const char *tmpl_to_use = point_tmpl ? point_tmpl : "%s %v";
    char out_buf[8192];
    geif_format_template(out_buf, sizeof(out_buf), tmpl_to_use, &ctx);
    fprintf(out_fp, "%s\n", out_buf);

    if (averages) free(averages);
    if (attr_scores) free(attr_scores);
}

/**
 * @brief Classifies an unassigned input sample against all ensemble categories (-c).
 */
static void process_categorize_row(geif_ensemble_t *ensemble,
                                   const geif_column_config_t *col_cfg,
                                   bool direct_feature_mapping,
                                   const cat_filter_t *cat_filter,
                                   char **tokens,
                                   uint32_t n_tok,
                                   const char *orig_line,
                                   uint64_t line_num,
                                   char list_sep,
                                   char cat_sep,
                                   char label_sep,
                                   double threshold,
                                   bool threshold_is_average,
                                   bool threshold_is_percentage,
                                   bool score_limit_given,
                                   const char *point_tmpl,
                                   int decimals,
                                   const char *printf_format,
                                   const char *print_dimension,
                                   uint32_t low_rgb,
                                   uint32_t high_rgb,
                                   double *vec,
                                   uint32_t dims,
                                   FILE *out_fp,
                                   uint64_t *analyzed,
                                   agg_cat_bucket_t **agg_buckets,
                                   size_t *agg_count,
                                   size_t *agg_cap)
{
    if (direct_feature_mapping) {
        for (uint32_t i = 0; i < dims && i < n_tok; i++) {
            char *endptr;
            double val = strtod(tokens[i], &endptr);
            vec[i] = (isnan(val) || isinf(val) || endptr == tokens[i]) ? 0.0 : val;
        }
    } else {
        if (!geif_extract_features(col_cfg, tokens, n_tok, vec)) {
            return;
        }
    }

    char in_label_buf[256] = {0};
    char in_cat_buf[256] = {0};
    if (!direct_feature_mapping && col_cfg) {
        geif_extract_label(col_cfg, tokens, n_tok, label_sep, in_label_buf, sizeof(in_label_buf));
        geif_extract_category(col_cfg, tokens, n_tok, cat_sep, in_cat_buf, sizeof(in_cat_buf));
    }

    if (ensemble->aggregate && agg_buckets && agg_count && agg_cap) {
        agg_cat_bucket_t *found = NULL;
        for (size_t i = 0; i < *agg_count; i++) {
            if (strcmp((*agg_buckets)[i].category, in_cat_buf) == 0) {
                found = &(*agg_buckets)[i];
                break;
            }
        }
        if (!found) {
            if (*agg_count >= *agg_cap) {
                size_t new_cap = (*agg_cap == 0) ? 16 : (*agg_cap * 2);
                *agg_buckets = (agg_cat_bucket_t *)xrealloc(*agg_buckets, new_cap * sizeof(agg_cat_bucket_t));
                *agg_cap = new_cap;
            }
            found = &(*agg_buckets)[(*agg_count)++];
            memset(found, 0, sizeof(*found));
            memcpy(found->category, in_cat_buf, sizeof(found->category));
            memcpy(found->label, in_label_buf, sizeof(found->label));
            found->summary = (double *)xcalloc(dims, sizeof(double));
        }
        for (uint32_t j = 0; j < dims; j++) {
            found->summary[j] += vec[j];
        }
        return;
    }

    score_and_print_categorize_vec(ensemble, cat_filter, in_cat_buf, in_label_buf,
                                   orig_line, tokens, n_tok, line_num,
                                   list_sep, cat_sep, label_sep, threshold,
                                   threshold_is_average, threshold_is_percentage,
                                   score_limit_given, point_tmpl, decimals,
                                   printf_format, print_dimension, low_rgb, high_rgb,
                                   vec, dims, out_fp, analyzed);
}

/**
 * @brief Recalculates empirical percentile threshold scores for all sub-forests in an ensemble.
 *
 * Evaluates reservoir samples to locate the empirical score boundary corresponding
 * to the given cumulative percentile (e.g. 80% or 95%).
 *
 * @param[in,out] ens     Category ensemble instance.
 * @param[in]     pct     Percentile value (0.0 to 100.0).
 * @param[in]     verbose True to print per-category percentile diagnostics.
 */
static void update_ensemble_percentage_scores(geif_ensemble_t *ens, double pct, const cat_filter_t *filter, bool verbose)
{
    if (!ens) return;
    for (size_t c = 0; c < ens->count; c++) {
        const char *cat_name = ens->entries[c].category;
        if (filter && !geif_cat_filter_allows(filter, cat_name)) {
            continue;
        }
        geif_forest_t *sf = ens->entries[c].forest;
        if (sf) {
            sf->percentage_score = geif_forest_calculate_percentile_score(sf, pct);
            if (verbose) {
                printf("Percentage score for '%s': %.6f (%.2f%% of samples have lower score)\n",
                       cat_name[0] ? cat_name : "(default)",
                       sf->percentage_score, pct);
            }
        }
    }
}

/**
 * @brief Main entry point for the GEIF CLI application.
 *
 * Handles CLI command-line arguments, RC configuration files, model loading/saving,
 * training on CSV datasets, reservoir streaming, outlier pruning (-k),
 * synthetic test grid generation (-T), and real-time anomaly inference (-a/-c).
 *
 * @param[in] argc Argument count.
 * @param[in] argv Argument vector.
 * @return 0 on success, non-zero exit code on failure.
 */
int main(int argc, char *argv[])
{
    const char *learn_file      = NULL;
    const char *analyze_file    = NULL;
    const char *categorize_file = NULL;
    const char *save_file       = NULL;
    const char *load_file       = NULL;
    const char *output_file     = NULL;
    const char *ignore_spec   = NULL;
    const char *include_spec  = NULL;
    const char *label_spec    = NULL;
    const char *category_spec = NULL;
    uint64_t    min_cat_rows  = 0;
    const char *new_cat_tmpl  = NULL;
    const char *missed_cat_tmpl = NULL;
    const char *point_tmpl    = NULL;
    bool        silent_outliers = false;
    time_t      delete_interval = 0;
    cat_filter_t cat_filter   = {0};
    char delimiter           = ',';
    char list_separator      = ',';
    bool field_sep_explicit  = false;
    bool list_sep_explicit   = false;
    bool skip_header         = false;
    bool header_explicit     = false;
    bool aggregate           = false;
    bool aggregate_explicit  = false;
    double threshold         = 0.5;
    bool threshold_is_average = false;
    bool threshold_is_percentage = false;
    double outlier_percentage = 0.0;
    bool query_mode          = false;
    bool verbose             = false;
    int  decimals            = 6;
    const char *printf_format = NULL;
    const char *print_dimension = NULL;
    const char *average_tmpl  = NULL;
    uint32_t low_rgb         = 0x20FF20;
    uint32_t high_rgb        = 0xFF0000;

    char category_sep        = ';';
    char label_sep           = '-';

    bool cli_outlier_score_given = false;
    char cli_outlier_score_spec[64] = {0};
    int kill_outliers_count = 0;

    bool   run_test_grid         = false;
    double test_extension_factor = 0.0;
    int    test_range_interval   = 256;

    geif_rc_config_t rc_cfg;
    geif_rc_config_init(&rc_cfg);
    geif_rc_load_default(&rc_cfg);

    bool cli_trees_given = false;
    bool cli_samples_given = false;
    bool cli_decimals_given = false;
    bool cli_print_dim_given = false;

    geif_config_t config = geif_config_default();

    static const struct option long_options[] = {
        {"algo",         required_argument, 0, 'B'},
        {"algorithm",    required_argument, 0, 'B'},
        {"method",       required_argument, 0, 'B'},
        {"categorize",   required_argument, 0, 'c'},
        {"trees",        required_argument, 0, 't'},
        {"samples",      required_argument, 0, 's'},
        {"output",       required_argument, 0, 'o'},
        {"inplace",      required_argument, 0, 'z'},
        {"inplace-forest", required_argument, 0, 'z'},
        {"missing",      optional_argument, 0, 'M'},
        {"new",          optional_argument, 0, 'N'},
        {"header",       no_argument,       0, 'H'},
        {"no-header",    no_argument,       0, 1001},
        {"aggregate",    no_argument,       0, 'A'},
        {"help",         no_argument,       0, 'h'},
        {"version",      no_argument,       0, 'v'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "l:a:c:w:r:z:o:T::O:t:i:s:m:f:e:HqvhI:U:L:C:F:R:N::M::p:SD:d:j:v::WAkg:B:", long_options, NULL)) != -1) {
        switch (opt) {
        case 'B': config.algo = geif_algo_from_name(optarg); break;
        case 'l': learn_file = optarg; break;
        case 'a': analyze_file = optarg; break;
        case 'c': categorize_file = optarg; break;
        case 'w': save_file = optarg; break;
        case 'r': load_file = optarg; break;
        case 'z':
            load_file = optarg;
            save_file = optarg;
            break;
        case 'o': output_file = optarg; break;
        case 'k': kill_outliers_count++; break;
        case 'T':
            run_test_grid = true;
            if (optarg != NULL) {
                test_extension_factor = atof(optarg);
            } else if (optind < argc && argv[optind][0] != '-') {
                test_extension_factor = atof(argv[optind++]);
            }
            break;
        case 'O':
            cli_outlier_score_given = true;
            strncpy(cli_outlier_score_spec, optarg, sizeof(cli_outlier_score_spec) - 1);
            if (strcmp(optarg, "average") == 0) {
                threshold_is_average = true;
                threshold_is_percentage = false;
            } else {
                threshold_is_average = false;
                size_t olen = strlen(optarg);
                if (olen > 0 && optarg[olen - 1] == '%') {
                    threshold_is_percentage = true;
                    outlier_percentage = atof(optarg);
                } else {
                    threshold_is_percentage = false;
                    threshold = atof(optarg);
                }
            }
            break;
        case 't':
            config.tree_count = (uint32_t)atoi(optarg);
            cli_trees_given = true;
            break;
        case 'i':
            config.tree_count = (uint32_t)atoi(optarg);
            test_range_interval = atoi(optarg);
            cli_trees_given = true;
            break;
        case 's':
            config.samples_per_tree = (uint32_t)atoi(optarg);
            cli_samples_given = true;
            break;
        case 'g':
            if (!geif_rc_parse_file(&rc_cfg, optarg)) {
                fprintf(stderr, "geif: error: cannot read config file '%s'\n", optarg);
                return 1;
            }
            break;
        case 'm':
            if (strchr(optarg, '%') != NULL) {
                printf_format = optarg;
            } else {
                config.max_depth = (uint32_t)atoi(optarg);
            }
            break;
        case 'd':
            decimals = atoi(optarg);
            cli_decimals_given = true;
            break;
        case 'j':
            print_dimension = optarg;
            cli_print_dim_given = true;
            break;
        case 'f':
            delimiter = optarg[0];
            field_sep_explicit = true;
            break;
        case 'e':
            list_separator = optarg[0];
            list_sep_explicit = true;
            if (!field_sep_explicit) {
                delimiter = optarg[0];
            }
            break;
        case 'H':
            skip_header = true;
            header_explicit = true;
            break;
        case 1001:
            skip_header = false;
            header_explicit = true;
            break;
        case 'I': ignore_spec = optarg; break;
        case 'U': include_spec = optarg; break;
        case 'L': label_spec = optarg; break;
        case 'C': category_spec = optarg; break;
        case 'F':
            if (!geif_cat_filter_add(&cat_filter, optarg)) {
                return 1;
            }
            break;
        case 'R':
            min_cat_rows = (uint64_t)strtoull(optarg, NULL, 10);
            break;
        case 'D':
            delete_interval = parse_delete_interval(optarg);
            if (delete_interval == 0) {
                fprintf(stderr, "geif: error: invalid time interval for -D: '%s'\n", optarg);
                return 1;
            }
            break;
        case 'N':
            if (optarg != NULL) {
                new_cat_tmpl = optarg;
            } else if (optind < argc && argv[optind][0] != '-') {
                new_cat_tmpl = argv[optind++];
            } else {
                new_cat_tmpl = "%v";
            }
            break;
        case 'M':
            if (optarg != NULL) {
                missed_cat_tmpl = optarg;
            } else if (optind < argc && argv[optind][0] != '-') {
                missed_cat_tmpl = argv[optind++];
            } else {
                missed_cat_tmpl = "%C";
            }
            break;
        case 'p':
            point_tmpl = optarg;
            break;
        case 'S':
            silent_outliers = true;
            break;
        case 'W':
            // Geometric auto-scaling (enabled by default in GEIF, accepted for CEIF compatibility)
            break;
        case 'A':
            aggregate = true;
            aggregate_explicit = true;
            break;
        case 'q': query_mode = true; break;
        case 'v':
            if (optarg != NULL) {
                average_tmpl = optarg;
            } else if (optind < argc && argv[optind][0] != '-') {
                average_tmpl = argv[optind++];
            } else {
                verbose = true;
            }
            break;
        case 'h': print_usage(argv[0]); return 0;
        default:  print_usage(argv[0]); return 1;
        }
    }

    // Apply configuration file settings if not explicitly specified on CLI
    if (!cli_trees_given && rc_cfg.tree_count_set) {
        config.tree_count = rc_cfg.tree_count;
    }
    if (!cli_samples_given && rc_cfg.samples_per_tree_set) {
        config.samples_per_tree = rc_cfg.samples_per_tree;
    }
    if (!cli_decimals_given && rc_cfg.decimals_set) {
        decimals = rc_cfg.decimals;
    }
    if (!cli_print_dim_given && rc_cfg.print_dimension_set) {
        print_dimension = rc_cfg.print_dimension;
    }
    if (rc_cfg.low_rgb_set) {
        low_rgb = rc_cfg.low_rgb;
    }
    if (rc_cfg.high_rgb_set) {
        high_rgb = rc_cfg.high_rgb;
    }
    if (rc_cfg.category_sep_set) {
        category_sep = rc_cfg.category_sep;
    }
    if (rc_cfg.label_sep_set) {
        label_sep = rc_cfg.label_sep;
    }
    if (!header_explicit && rc_cfg.header_set) {
        skip_header = rc_cfg.header;
    }
    if (!cli_outlier_score_given && rc_cfg.outlier_score_set) {
        cli_outlier_score_given = true;
        memcpy(cli_outlier_score_spec, rc_cfg.outlier_score_spec, sizeof(cli_outlier_score_spec));
        if (strcmp(rc_cfg.outlier_score_spec, "average") == 0) {
            threshold_is_average = true;
            threshold_is_percentage = false;
        } else {
            threshold_is_average = false;
            size_t olen = strlen(rc_cfg.outlier_score_spec);
            if (olen > 0 && rc_cfg.outlier_score_spec[olen - 1] == '%') {
                threshold_is_percentage = true;
                outlier_percentage = atof(rc_cfg.outlier_score_spec);
            } else {
                threshold_is_percentage = false;
                threshold = atof(rc_cfg.outlier_score_spec);
            }
        }
    }

    // If analysis file is provided, -T was used as threshold in legacy calls
    if (analyze_file && run_test_grid && !cli_outlier_score_given) {
        threshold = test_extension_factor;
        threshold_is_average = false;
        threshold_is_percentage = false;
        cli_outlier_score_given = true;
        snprintf(cli_outlier_score_spec, sizeof(cli_outlier_score_spec), "%g", test_extension_factor);
        run_test_grid = false;
    }

    geif_ensemble_t *ensemble = NULL;

    // Mode: Load existing model
    if (load_file) {
        geif_status_t status = geif_ensemble_load_json(&ensemble, load_file);
        if (status != GEIF_OK) {
            fprintf(stderr, "Error loading model from '%s': %s\n", load_file, geif_status_str(status));
            return 1;
        }
        if (verbose) {
            printf("Loaded model from '%s' (%zu sub-forests, %u dimensions)\n",
                   load_file, ensemble->count, ensemble->dimensions);
        }
        ensemble->scale_score = true;
        for (size_t i = 0; i < ensemble->count; i++) {
            if (ensemble->entries[i].forest) ensemble->entries[i].forest->scale_score = true;
        }
        if (cli_outlier_score_given) {
            memcpy(ensemble->outlier_score_spec, cli_outlier_score_spec, sizeof(ensemble->outlier_score_spec));
        } else if (ensemble->outlier_score_spec[0] != '\0') {
            if (strcmp(ensemble->outlier_score_spec, "average") == 0) {
                threshold_is_average = true;
                threshold_is_percentage = false;
            } else {
                threshold_is_average = false;
                size_t olen = strlen(ensemble->outlier_score_spec);
                if (olen > 0 && ensemble->outlier_score_spec[olen - 1] == '%') {
                    threshold_is_percentage = true;
                    outlier_percentage = atof(ensemble->outlier_score_spec);
                } else {
                    threshold_is_percentage = false;
                    threshold = atof(ensemble->outlier_score_spec);
                }
            }
        }

        if (!learn_file) {
            if (delete_interval > 0) {
                size_t before = ensemble->count;
                geif_ensemble_prune_age(ensemble, delete_interval, time(NULL));
                if (verbose && before != ensemble->count) {
                    printf("Age pruning (-D): dropped %zu stale sub-forests (%zu remaining).\n",
                           before - ensemble->count, ensemble->count);
                }
            }
            if (min_cat_rows > 0) {
                size_t before = ensemble->count;
                geif_ensemble_prune_categories(ensemble, min_cat_rows);
                if (verbose && before != ensemble->count) {
                    printf("Pruned categories with fewer than %llu rows (%zu -> %zu categories retained).\n",
                           (unsigned long long)min_cat_rows, before, ensemble->count);
                }
            }
        }

        if (!learn_file && !query_mode) {
            for (size_t i = 0; i < ensemble->count; i++) {
                const char *cat_name = ensemble->entries[i].category;
                if (!geif_cat_filter_allows(&cat_filter, cat_name)) {
                    continue;
                }
                geif_forest_t *sf = ensemble->entries[i].forest;
                if (sf && sf->pool_count > 0 && !sf->is_trained) {
                    geif_forest_train(sf);
                }
            }
        }
        if (threshold_is_percentage) {
            update_ensemble_percentage_scores(ensemble, outlier_percentage, &cat_filter, verbose);
        }
        if (cli_decimals_given || rc_cfg.decimals_set) {
            ensemble->decimals = decimals;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) ensemble->entries[i].forest->decimals = decimals;
            }
        } else if (ensemble->decimals >= 0) {
            decimals = ensemble->decimals;
        }

        if (!field_sep_explicit && ensemble->input_delimiter != '\0') {
            delimiter = ensemble->input_delimiter;
        } else if (field_sep_explicit) {
            ensemble->input_delimiter = delimiter;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) ensemble->entries[i].forest->input_delimiter = delimiter;
            }
        }
        if (!list_sep_explicit && ensemble->list_separator != '\0') {
            list_separator = ensemble->list_separator;
        } else if (list_sep_explicit) {
            ensemble->list_separator = list_separator;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) ensemble->entries[i].forest->list_separator = list_separator;
            }
        }
        if (!header_explicit) {
            skip_header = ensemble->has_header;
        } else {
            ensemble->has_header = skip_header;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) ensemble->entries[i].forest->has_header = skip_header;
            }
        }
        if (!aggregate_explicit) {
            aggregate = ensemble->aggregate;
        } else {
            ensemble->aggregate = aggregate;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) ensemble->entries[i].forest->aggregate = aggregate;
            }
        }
    }

    // Mode: Model info query (-q)
    if (query_mode) {
        if (!ensemble) {
            fprintf(stderr, "Error: -q requires a loaded model via -r <model.json>\n");
            return 1;
        }
        char summary[4096];
        geif_ensemble_summary(ensemble, summary, sizeof(summary));
        printf("%s", summary);
        geif_ensemble_destroy(ensemble);
        return 0;
    }

    // Standalone model maintenance (-D pruning, -R row threshold pruning, and/or -k outlier recalibration): -r <model> [-D <interval>] [-R <min_rows>] [-k] -w <model>
    if (!learn_file && !analyze_file && !categorize_file && (delete_interval > 0 || kill_outliers_count > 0 || min_cat_rows > 0)) {
        if (!ensemble) {
            fprintf(stderr, "Error: maintenance operations (-D, -R, -k) require a loaded model via -r <model.json>\n");
            return 1;
        }
        if (delete_interval > 0) {
            size_t before = ensemble->count;
            geif_ensemble_prune_age(ensemble, delete_interval, time(NULL));
            if (verbose && before != ensemble->count) {
                printf("Age pruning (-D): dropped %zu stale sub-forests (%zu remaining).\n",
                       before - ensemble->count, ensemble->count);
            }
        }
        if (min_cat_rows > 0) {
            size_t before = ensemble->count;
            geif_ensemble_prune_categories(ensemble, min_cat_rows);
            if (verbose && before != ensemble->count) {
                printf("Pruned categories with fewer than %llu rows (%zu -> %zu categories retained).\n",
                       (unsigned long long)min_cat_rows, before, ensemble->count);
            }
        }
        if (kill_outliers_count > 0) {
            if (verbose) {
                printf("Pruning %d outlier(s) per category and recalibrating ensemble...\n", kill_outliers_count);
            }
            geif_status_t status = geif_ensemble_remove_outliers(ensemble, (uint32_t)kill_outliers_count);
            if (status != GEIF_OK) {
                fprintf(stderr, "Error during outlier pruning (-k): %s\n", geif_status_str(status));
                geif_ensemble_destroy(ensemble);
                return 1;
            }
            if (threshold_is_percentage) {
                update_ensemble_percentage_scores(ensemble, outlier_percentage, &cat_filter, verbose);
            }
        }
        if (save_file) {
            ensemble->decimals = decimals;
            ensemble->input_delimiter = delimiter;
            ensemble->list_separator = list_separator;
            ensemble->has_header = skip_header;
            ensemble->aggregate = aggregate;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) {
                    ensemble->entries[i].forest->decimals = decimals;
                    ensemble->entries[i].forest->input_delimiter = delimiter;
                    ensemble->entries[i].forest->list_separator = list_separator;
                    ensemble->entries[i].forest->has_header = skip_header;
                    ensemble->entries[i].forest->aggregate = aggregate;
                }
            }
            geif_status_t status = geif_ensemble_save_json(ensemble, save_file);
            if (status != GEIF_OK) {
                fprintf(stderr, "Error saving ensemble model to '%s': %s\n",
                        save_file, geif_status_str(status));
                geif_ensemble_destroy(ensemble);
                return 1;
            }
        }
        geif_ensemble_destroy(ensemble);
        return 0;
    }

    // Mode: Test Grid Generation (-T [margin]) for population drift visualization
    if (run_test_grid && !analyze_file) {
        if (!ensemble) {
            fprintf(stderr, "Error: -T requires a loaded model via -r <model.json>\n");
            return 1;
        }

        FILE *out_fp = stdout;
        if (output_file && strcmp(output_file, "-") != 0) {
            out_fp = xfopen(output_file, "w");
            if (!out_fp) {
                geif_ensemble_destroy(ensemble);
                return 1;
            }
        }

        geif_generate_test_grid(ensemble,
                                test_extension_factor,
                                test_range_interval,
                                &cat_filter,
                                threshold,
                                threshold_is_average,
                                threshold_is_percentage,
                                point_tmpl,
                                decimals,
                                list_separator,
                                low_rgb,
                                high_rgb,
                                printf_format,
                                print_dimension,
                                out_fp);

        if (out_fp != stdout) fclose(out_fp);
        geif_ensemble_destroy(ensemble);
        return 0;
    }

    // Mode: Train new model (-l)
    if (learn_file) {
        FILE *fp = xfopen(learn_file, "r");
        if (!fp) {
            if (ensemble) geif_ensemble_destroy(ensemble);
            return 1;
        }

        bool is_existing_model = (ensemble != NULL);
        char line[8192];
        char line_copy[8192];
        char *tokens[1024];
        uint32_t total_cols = 0;
        bool first_line_is_data = false;
        uint64_t line_num = 0;

        // Read first non-empty line to detect columns and handle header
        while (fgets(line, sizeof(line), fp)) {
            line_num++;
            if (line[0] == '#' || line[0] == '\r' || line[0] == '\n') continue;
            strncpy(line_copy, line, sizeof(line_copy) - 1);
            line_copy[sizeof(line_copy) - 1] = '\0';
            total_cols = tokenize_line(line_copy, delimiter, tokens, 1024);
            if (total_cols > 0) break;
        }

        if (total_cols == 0) {
            fprintf(stderr, "Error: could not detect columns in '%s'\n", learn_file);
            xfclose(fp);
            if (ensemble) geif_ensemble_destroy(ensemble);
            return 1;
        }

        const char *active_ignore = ignore_spec ? ignore_spec : (ensemble ? (ensemble->ignore_dims_spec[0] ? ensemble->ignore_dims_spec : NULL) : NULL);
        const char *active_include = include_spec ? include_spec : (ensemble ? (ensemble->include_dims_spec[0] ? ensemble->include_dims_spec : NULL) : NULL);
        const char *active_label = label_spec ? label_spec : (ensemble ? (ensemble->label_dims_spec[0] ? ensemble->label_dims_spec : NULL) : NULL);
        const char *active_category = category_spec ? category_spec : (ensemble ? (ensemble->category_dims_spec[0] ? ensemble->category_dims_spec : NULL) : NULL);

        geif_column_config_t col_cfg;
        geif_column_config_init(&col_cfg, active_ignore, active_include, active_label, active_category);
        if (!geif_column_config_resolve(&col_cfg, total_cols)) {
            fprintf(stderr, "geif: error: no active feature dimensions remaining after applying column masks (-I, -U, -L, -C)\n");
            xfclose(fp);
            geif_column_config_free(&col_cfg);
            if (ensemble) geif_ensemble_destroy(ensemble);
            return 1;
        }

        uint32_t dims = col_cfg.feature_dim_count;

        if (!ensemble) {
            geif_status_t status = geif_ensemble_create(&ensemble, dims, &config);
            if (status != GEIF_OK) {
                fprintf(stderr, "Error initializing ensemble: %s\n", geif_status_str(status));
                xfclose(fp);
                geif_column_config_free(&col_cfg);
                return 1;
            }
        } else if (dims != ensemble->dimensions) {
            fprintf(stderr, "geif: error: training file '%s' has %u feature dimensions, but model expects %u\n",
                    learn_file, dims, ensemble->dimensions);
            xfclose(fp);
            geif_column_config_free(&col_cfg);
            return 1;
        }

        // Save column metadata and aggregate flag into ensemble
        ensemble->total_input_cols = total_cols;
        ensemble->decimals = decimals;
        ensemble->scale_score = true;
        ensemble->input_delimiter = delimiter;
        ensemble->list_separator = list_separator;
        ensemble->has_header = skip_header;
        ensemble->aggregate = aggregate;
        for (size_t i = 0; i < ensemble->count; i++) {
            if (ensemble->entries[i].forest) {
                ensemble->entries[i].forest->aggregate = aggregate;
                ensemble->entries[i].forest->summary_slot = -1;
            }
        }
        if (cli_outlier_score_given) {
            memcpy(ensemble->outlier_score_spec, cli_outlier_score_spec, sizeof(ensemble->outlier_score_spec));
        }
        if (active_ignore && active_ignore != ensemble->ignore_dims_spec) {
            snprintf(ensemble->ignore_dims_spec, sizeof(ensemble->ignore_dims_spec), "%s", active_ignore);
        }
        if (active_include && active_include != ensemble->include_dims_spec) {
            snprintf(ensemble->include_dims_spec, sizeof(ensemble->include_dims_spec), "%s", active_include);
        }
        if (active_label && active_label != ensemble->label_dims_spec) {
            snprintf(ensemble->label_dims_spec, sizeof(ensemble->label_dims_spec), "%s", active_label);
        }
        if (active_category && active_category != ensemble->category_dims_spec) {
            snprintf(ensemble->category_dims_spec, sizeof(ensemble->category_dims_spec), "%s", active_category);
        }

        FILE *learn_out_fp = stdout;
        bool close_learn_out_fp = false;
        if (new_cat_tmpl && output_file && !analyze_file && strcmp(output_file, "-") != 0) {
            learn_out_fp = xfopen(output_file, "w");
            if (!learn_out_fp) {
                xfclose(fp);
                geif_column_config_free(&col_cfg);
                if (ensemble) geif_ensemble_destroy(ensemble);
                return 1;
            }
            close_learn_out_fp = true;
        }

        if (!skip_header) {
            first_line_is_data = true;
        }

        double *vec = (double *)xmalloc(dims * sizeof(double));
        char cat_buf[256];
        char label_buf[256];
        uint64_t rows = 0;

        if (first_line_is_data) {
            char orig_line[8192];
            strncpy(orig_line, line, sizeof(orig_line) - 1);
            orig_line[sizeof(orig_line) - 1] = '\0';
            orig_line[strcspn(orig_line, "\r\n")] = 0;

            geif_extract_category(&col_cfg, tokens, total_cols, category_sep, cat_buf, sizeof(cat_buf));
            geif_extract_label(&col_cfg, tokens, total_cols, label_sep, label_buf, sizeof(label_buf));

            geif_category_entry_t *entry = geif_ensemble_get_entry(ensemble, cat_buf);
            bool is_new_cat_sample = false;
            if (is_existing_model) {
                if (!entry || entry->is_new_category) {
                    is_new_cat_sample = true;
                }
            } else {
                if (!entry) {
                    is_new_cat_sample = true;
                }
            }

            if (geif_extract_features(&col_cfg, tokens, total_cols, vec)) {
                geif_ensemble_feed(ensemble, cat_buf, vec);
                rows++;
            }

            if (is_new_cat_sample && new_cat_tmpl && geif_cat_filter_allows(&cat_filter, cat_buf)) {
                geif_template_context_t ctx = {
                    .orig_line        = orig_line,
                    .label            = label_buf,
                    .category         = cat_buf,
                    .input_category   = cat_buf,
                    .score            = 1.0,
                    .test_average_score = 1.0,
                    .metric_depth     = 0.0,
                    .d_out            = 999.0,
                    .H_max            = 0.0,
                    .is_outlier       = 1,
                    .timestamp        = time(NULL),
                    .total_rows       = 0,
                    .analyzed_rows    = rows,
                    .high_analyzed_rows = 1,
                    .row_idx          = line_num,
                    .vector           = vec,
                    .averages         = NULL,
                    .attr_scores      = NULL,
                    .vector_dim       = dims,
                    .raw_values       = tokens,
                    .raw_value_count  = total_cols,
                    .list_separator   = list_separator,
                    .category_sep     = category_sep,
                    .label_sep        = label_sep,
                    .decimals         = decimals,
                    .printf_format    = printf_format,
                    .print_dimension  = print_dimension,
                    .low_rgb          = low_rgb,
                    .high_rgb         = high_rgb,
                    .is_summary       = false
                };
                char out_buf[8192];
                geif_format_template(out_buf, sizeof(out_buf), new_cat_tmpl, &ctx);
                fprintf(learn_out_fp, "%s\n", out_buf);
            }
        }

        while (fgets(line, sizeof(line), fp)) {
            line_num++;
            if (line[0] == '#' || line[0] == '\r' || line[0] == '\n') continue;

            char orig_line[8192];
            strncpy(orig_line, line, sizeof(orig_line) - 1);
            orig_line[sizeof(orig_line) - 1] = '\0';
            orig_line[strcspn(orig_line, "\r\n")] = 0;

            strncpy(line_copy, line, sizeof(line_copy) - 1);
            line_copy[sizeof(line_copy) - 1] = '\0';
            uint32_t n_tok = tokenize_line(line_copy, delimiter, tokens, 1024);
            if (n_tok >= total_cols) {
                geif_extract_category(&col_cfg, tokens, n_tok, category_sep, cat_buf, sizeof(cat_buf));
                geif_extract_label(&col_cfg, tokens, n_tok, label_sep, label_buf, sizeof(label_buf));

                geif_category_entry_t *entry = geif_ensemble_get_entry(ensemble, cat_buf);
                bool is_new_cat_sample = false;
                if (is_existing_model) {
                    if (!entry || entry->is_new_category) {
                        is_new_cat_sample = true;
                    }
                } else {
                    if (!entry) {
                        is_new_cat_sample = true;
                    }
                }

                if (geif_extract_features(&col_cfg, tokens, n_tok, vec)) {
                    geif_ensemble_feed(ensemble, cat_buf, vec);
                    rows++;
                }

                if (is_new_cat_sample && new_cat_tmpl && geif_cat_filter_allows(&cat_filter, cat_buf)) {
                    geif_template_context_t ctx = {
                        .orig_line        = orig_line,
                        .label            = label_buf,
                        .category         = cat_buf,
                        .input_category   = cat_buf,
                        .score            = 1.0,
                        .test_average_score = 1.0,
                        .metric_depth     = 0.0,
                        .d_out            = 999.0,
                        .H_max            = 0.0,
                        .is_outlier       = 1,
                        .timestamp        = time(NULL),
                        .total_rows       = 0,
                        .analyzed_rows    = rows,
                        .high_analyzed_rows = 1,
                        .row_idx          = line_num,
                        .vector           = vec,
                        .averages         = NULL,
                        .attr_scores      = NULL,
                        .vector_dim       = dims,
                        .raw_values       = tokens,
                        .raw_value_count  = n_tok,
                        .list_separator   = list_separator,
                        .category_sep     = category_sep,
                        .label_sep        = label_sep,
                        .decimals         = decimals,
                        .printf_format    = printf_format,
                        .print_dimension  = print_dimension,
                        .low_rgb          = low_rgb,
                        .high_rgb         = high_rgb,
                        .is_summary       = false
                    };
                    char out_buf[8192];
                    geif_format_template(out_buf, sizeof(out_buf), new_cat_tmpl, &ctx);
                    fprintf(learn_out_fp, "%s\n", out_buf);
                }
            }
        }
        xfree(vec);
        xfclose(fp);
        geif_column_config_free(&col_cfg);
        if (close_learn_out_fp && learn_out_fp) {
            xfclose(learn_out_fp);
            learn_out_fp = NULL;
        }

        if (delete_interval > 0) {
            size_t before = ensemble->count;
            geif_ensemble_prune_age(ensemble, delete_interval, time(NULL));
            if (verbose && before != ensemble->count) {
                printf("Age pruning (-D): dropped %zu stale sub-forests (%zu remaining).\n",
                       before - ensemble->count, ensemble->count);
            }
        }

        if (min_cat_rows > 0) {
            size_t before = ensemble->count;
            geif_ensemble_prune_categories(ensemble, min_cat_rows);
            if (verbose && before != ensemble->count) {
                printf("Pruned categories with fewer than %llu rows (%zu -> %zu categories retained).\n",
                       (unsigned long long)min_cat_rows, before, ensemble->count);
            }
        }

        if (verbose) {
            printf("Ingested %llu rows from '%s' (%u feature dimensions, %zu categories). Training ensemble...\n",
                   (unsigned long long)rows, learn_file, dims, ensemble->count);
        }

        geif_status_t status = geif_ensemble_train(ensemble);
        if (status != GEIF_OK) {
            fprintf(stderr, "Error training ensemble: %s\n", geif_status_str(status));
            geif_ensemble_destroy(ensemble);
            return 1;
        }

        if (cli_outlier_score_given) {
            memcpy(ensemble->outlier_score_spec, cli_outlier_score_spec, sizeof(ensemble->outlier_score_spec));
            ensemble->outlier_score_spec[sizeof(ensemble->outlier_score_spec) - 1] = '\0';
        }

        if (verbose) {
            printf("Training complete across %zu sub-forests.\n", ensemble->count);
        }

        if (kill_outliers_count > 0) {
            if (verbose) {
                printf("Pruning %d outlier(s) per category and recalibrating ensemble...\n", kill_outliers_count);
            }
            status = geif_ensemble_remove_outliers(ensemble, (uint32_t)kill_outliers_count);
            if (status != GEIF_OK) {
                fprintf(stderr, "Error during outlier pruning (-k): %s\n", geif_status_str(status));
                geif_ensemble_destroy(ensemble);
                return 1;
            }
        }

        if (threshold_is_percentage) {
            update_ensemble_percentage_scores(ensemble, outlier_percentage, &cat_filter, verbose);
        }

        // Save trained model if requested
        if (save_file) {
            ensemble->decimals = decimals;
            ensemble->input_delimiter = delimiter;
            ensemble->list_separator = list_separator;
            ensemble->has_header = skip_header;
            ensemble->aggregate = aggregate;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) {
                    ensemble->entries[i].forest->decimals = decimals;
                    ensemble->entries[i].forest->input_delimiter = delimiter;
                    ensemble->entries[i].forest->list_separator = list_separator;
                    ensemble->entries[i].forest->has_header = skip_header;
                    ensemble->entries[i].forest->aggregate = aggregate;
                }
            }
            status = geif_ensemble_save_json(ensemble, save_file);
            if (status != GEIF_OK) {
                fprintf(stderr, "Error saving model to '%s': %s\n", save_file, geif_status_str(status));
            } else if (verbose && strcmp(save_file, "-") != 0) {
                printf("Saved model to '%s'\n", save_file);
            }
        }
    }

    uint64_t total_outliers = 0;

    // Mode: Analyze / Score (-a)
    if (analyze_file) {
        if (!ensemble) {
            fprintf(stderr, "Error: Analysis requires a model. Train with -l or load with -r.\n");
            return 1;
        }

        if (delete_interval > 0) {
            size_t before = ensemble->count;
            geif_ensemble_prune_age(ensemble, delete_interval, time(NULL));
            if (verbose && before != ensemble->count) {
                printf("Age pruning (-D): dropped %zu stale sub-forests (%zu remaining).\n",
                       before - ensemble->count, ensemble->count);
            }
        }

        if (min_cat_rows > 0) {
            size_t before = ensemble->count;
            geif_ensemble_prune_categories(ensemble, min_cat_rows);
            if (verbose && before != ensemble->count) {
                printf("Pruned categories with fewer than %llu rows (%zu -> %zu categories retained).\n",
                       (unsigned long long)min_cat_rows, before, ensemble->count);
            }
        }

        if (kill_outliers_count > 0) {
            if (verbose) {
                printf("Pruning %d outlier(s) per category and recalibrating ensemble...\n", kill_outliers_count);
            }
            geif_status_t status = geif_ensemble_remove_outliers(ensemble, (uint32_t)kill_outliers_count);
            if (status != GEIF_OK) {
                fprintf(stderr, "Error during outlier pruning (-k): %s\n", geif_status_str(status));
                geif_ensemble_destroy(ensemble);
                return 1;
            }
            if (threshold_is_percentage) {
                update_ensemble_percentage_scores(ensemble, outlier_percentage, &cat_filter, verbose);
            }
        }

        FILE *in_fp = xfopen(analyze_file, "r");
        if (!in_fp) {
            geif_ensemble_destroy(ensemble);
            return 1;
        }

        const char *out_target = output_file ? output_file : "-";
        FILE *out_fp = xfopen(out_target, "w");
        if (!out_fp) {
            xfclose(in_fp);
            geif_ensemble_destroy(ensemble);
            return 1;
        }

        // Column configuration: CLI flags override loaded model metadata
        const char *active_ignore = ignore_spec ? ignore_spec : (ensemble->ignore_dims_spec[0] ? ensemble->ignore_dims_spec : NULL);
        const char *active_include = include_spec ? include_spec : (ensemble->include_dims_spec[0] ? ensemble->include_dims_spec : NULL);
        const char *active_label = label_spec ? label_spec : (ensemble->label_dims_spec[0] ? ensemble->label_dims_spec : NULL);
        const char *active_category = category_spec ? category_spec : (ensemble->category_dims_spec[0] ? ensemble->category_dims_spec : NULL);

        char line[8192];
        char line_copy[8192];
        char *tokens[1024];
        uint32_t total_cols = 0;
        bool first_line_is_data = false;

        // Read first line to detect total input columns
        while (fgets(line, sizeof(line), in_fp)) {
            if (line[0] == '#' || line[0] == '\r' || line[0] == '\n') continue;
            strncpy(line_copy, line, sizeof(line_copy) - 1);
            line_copy[sizeof(line_copy) - 1] = '\0';
            total_cols = tokenize_line(line_copy, delimiter, tokens, 1024);
            if (total_cols > 0) break;
        }

        if (total_cols == 0) {
            xfclose(in_fp);
            xfclose(out_fp);
            geif_ensemble_destroy(ensemble);
            return 0;
        }

        geif_column_config_t col_cfg;
        geif_column_config_init(&col_cfg, active_ignore, active_include, active_label, active_category);
        if (!geif_column_config_resolve(&col_cfg, total_cols)) {
            fprintf(stderr, "geif: error: no active feature dimensions in analysis input '%s'\n", analyze_file);
            xfclose(in_fp);
            xfclose(out_fp);
            geif_column_config_free(&col_cfg);
            geif_ensemble_destroy(ensemble);
            return 1;
        }

        if (col_cfg.feature_dim_count != ensemble->dimensions) {
            fprintf(stderr, "geif: error: input '%s' has %u active feature dimensions, but model expects %u\n",
                    analyze_file, col_cfg.feature_dim_count, ensemble->dimensions);
            xfclose(in_fp);
            xfclose(out_fp);
            geif_column_config_free(&col_cfg);
            geif_ensemble_destroy(ensemble);
            return 1;
        }

        if (!skip_header) {
            first_line_is_data = true;
        }

        uint32_t dims = ensemble->dimensions;
        double *vec = (double *)xmalloc(dims * sizeof(double));
        uint64_t analyzed = 0;
        uint64_t line_num = 0;

        if (first_line_is_data) {
            line_num = 1;
            char orig_line[8192];
            strncpy(orig_line, line, sizeof(orig_line) - 1);
            orig_line[sizeof(orig_line) - 1] = '\0';
            orig_line[strcspn(orig_line, "\r\n")] = 0;

            process_scoring_row(ensemble, &col_cfg, &cat_filter, tokens, total_cols,
                                orig_line, line_num, list_separator, category_sep, label_sep, threshold,
                                threshold_is_average, threshold_is_percentage, silent_outliers, point_tmpl, average_tmpl,
                                new_cat_tmpl, decimals, printf_format, print_dimension,
                                low_rgb, high_rgb, vec, dims,
                                out_fp, &analyzed, &total_outliers);
        }

        while (fgets(line, sizeof(line), in_fp)) {
            line_num++;
            if (line[0] == '#' || line[0] == '\r' || line[0] == '\n') continue;

            char orig_line[8192];
            strncpy(orig_line, line, sizeof(orig_line) - 1);
            orig_line[sizeof(orig_line) - 1] = '\0';
            orig_line[strcspn(orig_line, "\r\n")] = 0;

            strncpy(line_copy, line, sizeof(line_copy) - 1);
            line_copy[sizeof(line_copy) - 1] = '\0';

            uint32_t n_tok = tokenize_line(line_copy, delimiter, tokens, 1024);
            if (n_tok >= total_cols) {
                process_scoring_row(ensemble, &col_cfg, &cat_filter, tokens, n_tok,
                                    orig_line, line_num, list_separator, category_sep, label_sep, threshold,
                                    threshold_is_average, threshold_is_percentage, silent_outliers, point_tmpl, average_tmpl,
                                    new_cat_tmpl, decimals, printf_format, print_dimension,
                                    low_rgb, high_rgb, vec, dims,
                                    out_fp, &analyzed, &total_outliers);
            }
        }

        // In aggregate mode (-A), evaluate accumulated category summaries at EOF
        if (ensemble->aggregate) {
            for (size_t i = 0; i < ensemble->count; i++) {
                geif_category_entry_t *matched_entry = &ensemble->entries[i];
                if (!matched_entry->summary || !matched_entry->forest) continue;
                const char *cat_name = matched_entry->category;
                if (!geif_cat_filter_allows(&cat_filter, cat_name)) {
                    xfree(matched_entry->summary);
                    matched_entry->summary = NULL;
                    continue;
                }

                geif_forest_t *sub_forest = matched_entry->forest;
                double effective_threshold = threshold;
                if (threshold_is_percentage) {
                    effective_threshold = (sub_forest->percentage_score > 0.0) ? sub_forest->percentage_score : 0.5;
                } else if (threshold_is_average) {
                    effective_threshold = (sub_forest->average_score > 0.0) ? sub_forest->average_score : 0.5;
                }

                double score = 0.0, h_M = 0.0, d_out = 0.0;
                geif_forest_score_detailed(sub_forest, matched_entry->summary, &score, &h_M, &d_out);
                double *attr = (double *)xcalloc(dims, sizeof(double));
                if (dims > 0) geif_forest_dimension_attribution(sub_forest, matched_entry->summary, attr);

                analyzed++;
                bool is_out = (score >= effective_threshold);
                if (is_out) total_outliers++;

                matched_entry->analyzed_rows++;
                matched_entry->sum_score += score;
                if (is_out) matched_entry->high_analyzed_rows++;

                time_t cat_ts = matched_entry->last_updated;
                if (cat_ts == 0) cat_ts = time(NULL);

                double test_avg = matched_entry->sum_score / (double)matched_entry->analyzed_rows;

                geif_template_context_t ctx = {
                    .orig_line        = "",
                    .label            = "",
                    .category         = cat_name,
                    .input_category   = cat_name,
                    .score            = score,
                    .test_average_score = test_avg,
                    .metric_depth     = h_M,
                    .d_out            = d_out,
                    .H_max            = sub_forest->H_max,
                    .is_outlier       = is_out ? 1 : 0,
                    .timestamp        = cat_ts,
                    .total_rows       = matched_entry->total_rows,
                    .analyzed_rows    = matched_entry->analyzed_rows,
                    .high_analyzed_rows = matched_entry->high_analyzed_rows,
                    .row_idx          = analyzed,
                    .vector           = matched_entry->summary,
                    .averages         = sub_forest->averages,
                    .attr_scores      = attr,
                    .vector_dim       = dims,
                    .raw_values       = NULL,
                    .raw_value_count  = 0,
                    .list_separator   = list_separator,
                    .category_sep     = category_sep,
                    .label_sep        = label_sep,
                    .decimals         = decimals,
                    .printf_format    = printf_format,
                    .print_dimension  = print_dimension,
                    .low_rgb          = low_rgb,
                    .high_rgb         = high_rgb,
                    .is_summary       = false
                };

                char out_buf[8192];
                if (point_tmpl) {
                    if (is_out) {
                        geif_format_template(out_buf, sizeof(out_buf), point_tmpl, &ctx);
                        fprintf(out_fp, "%s\n", out_buf);
                    }
                } else if (!average_tmpl) {
                    if (!silent_outliers || is_out) {
                        geif_format_template(out_buf, sizeof(out_buf), "%s %v", &ctx);
                        fprintf(out_fp, "%s\n", out_buf);
                    }
                }

                xfree(attr);
                xfree(matched_entry->summary);
                matched_entry->summary = NULL;
            }
        }

        // Emit Average summary (-v <format>) if specified
        if (average_tmpl) {
            for (size_t i = 0; i < ensemble->count; i++) {
                const char *cat_name = ensemble->entries[i].category;
                if (!geif_cat_filter_allows(&cat_filter, cat_name)) {
                    continue;
                }
                if (ensemble->entries[i].analyzed_rows > 0) {
                    geif_forest_t *sub_f = ensemble->entries[i].forest;
                    time_t cat_ts = ensemble->entries[i].last_updated;
                    if (cat_ts == 0) cat_ts = time(NULL);

                    double eff_thresh = 1.0;
                    if (threshold_is_percentage) {
                        eff_thresh = (sub_f && sub_f->percentage_score > 0.0) ? sub_f->percentage_score : 0.5;
                    } else if (threshold_is_average) {
                        eff_thresh = (sub_f && sub_f->average_score > 0.0) ? sub_f->average_score : 0.5;
                    } else {
                        eff_thresh = threshold;
                    }

                    double test_avg = ensemble->entries[i].sum_score / (double)ensemble->entries[i].analyzed_rows;

                    geif_template_context_t ctx = {
                        .orig_line        = "",
                        .label            = "",
                        .category         = cat_name,
                        .input_category   = cat_name,
                        .score            = eff_thresh,
                        .test_average_score = test_avg,
                        .metric_depth     = 0.0,
                        .d_out            = 0.0,
                        .H_max            = sub_f ? sub_f->H_max : 0.0,
                        .is_outlier       = 0,
                        .timestamp        = cat_ts,
                        .total_rows       = ensemble->entries[i].total_rows,
                        .analyzed_rows    = ensemble->entries[i].analyzed_rows,
                        .high_analyzed_rows = ensemble->entries[i].high_analyzed_rows,
                        .row_idx          = analyzed,
                        .vector           = NULL,
                        .averages         = sub_f ? sub_f->averages : NULL,
                        .attr_scores      = NULL,
                        .vector_dim       = sub_f ? sub_f->dimensions : 0,
                        .raw_values       = NULL,
                        .raw_value_count  = 0,
                        .list_separator   = list_separator,
                        .category_sep     = category_sep,
                        .label_sep        = label_sep,
                        .decimals         = decimals,
                        .printf_format    = printf_format,
                        .print_dimension  = print_dimension,
                        .low_rgb          = low_rgb,
                        .high_rgb         = high_rgb,
                        .is_summary       = true
                    };
                    char out_buf[8192];
                    geif_format_template(out_buf, sizeof(out_buf), average_tmpl, &ctx);
                    fprintf(out_fp, "%s\n", out_buf);
                }
            }
        }

        // Emit Missed categories (-M) if specified
        if (missed_cat_tmpl) {
            for (size_t i = 0; i < ensemble->count; i++) {
                const char *cat_name = ensemble->entries[i].category;
                if (!geif_cat_filter_allows(&cat_filter, cat_name)) {
                    continue;
                }
                if (!ensemble->entries[i].seen_in_analysis) {
                    total_outliers++;
                    geif_forest_t *sub_f = ensemble->entries[i].forest;
                    time_t cat_ts = ensemble->entries[i].last_updated;
                    if (cat_ts == 0) cat_ts = time(NULL);

                    geif_template_context_t ctx = {
                        .orig_line        = cat_name,
                        .label            = "",
                        .category         = cat_name,
                        .input_category   = cat_name,
                        .score            = 1.0,
                        .test_average_score = 0.0,
                        .metric_depth     = 0.0,
                        .d_out            = 0.0,
                        .H_max            = sub_f ? sub_f->H_max : 0.0,
                        .is_outlier       = 1,
                        .timestamp        = cat_ts,
                        .total_rows       = ensemble->entries[i].total_rows,
                        .analyzed_rows    = analyzed,
                        .high_analyzed_rows = 0,
                        .row_idx          = 0,
                        .vector           = NULL,
                        .averages         = sub_f ? sub_f->averages : NULL,
                        .attr_scores      = NULL,
                        .vector_dim       = sub_f ? sub_f->dimensions : 0,
                        .raw_values       = NULL,
                        .raw_value_count  = 0,
                        .list_separator   = list_separator,
                        .category_sep     = category_sep,
                        .label_sep        = label_sep,
                        .decimals         = decimals,
                        .printf_format    = printf_format,
                        .print_dimension  = print_dimension,
                        .low_rgb          = low_rgb,
                        .high_rgb         = high_rgb,
                        .is_summary       = false
                    };
                    char out_buf[8192];
                    geif_format_template(out_buf, sizeof(out_buf), missed_cat_tmpl, &ctx);
                    fprintf(out_fp, "%s\n", out_buf);
                }
            }
        }

        xfree(vec);
        xfclose(in_fp);
        xfclose(out_fp);
        geif_column_config_free(&col_cfg);

        if (verbose) {
            fprintf(stderr, "Scored %llu rows. Outliers (>= %.2f): %llu (%.2f%%)\n",
                    (unsigned long long)analyzed, threshold,
                    (unsigned long long)total_outliers,
                    analyzed > 0 ? (100.0 * (double)total_outliers / (double)analyzed) : 0.0);
        }

        if (!learn_file && save_file) {
            ensemble->decimals = decimals;
            ensemble->input_delimiter = delimiter;
            ensemble->list_separator = list_separator;
            ensemble->has_header = skip_header;
            ensemble->aggregate = aggregate;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) {
                    ensemble->entries[i].forest->decimals = decimals;
                    ensemble->entries[i].forest->input_delimiter = delimiter;
                    ensemble->entries[i].forest->list_separator = list_separator;
                    ensemble->entries[i].forest->has_header = skip_header;
                    ensemble->entries[i].forest->aggregate = aggregate;
                }
            }
            geif_status_t status = geif_ensemble_save_json(ensemble, save_file);
            if (status != GEIF_OK) {
                fprintf(stderr, "Error saving model to '%s': %s\n", save_file, geif_status_str(status));
            } else if (verbose && strcmp(save_file, "-") != 0) {
                printf("Saved model to '%s'\n", save_file);
            }
        }
    }

    // Mode: Categorize (-c)
    if (categorize_file) {
        if (!ensemble) {
            fprintf(stderr, "Error: Categorization requires a model. Train with -l or load with -r.\n");
            geif_cat_filter_free(&cat_filter);
            return 1;
        }

        if (delete_interval > 0) {
            size_t before = ensemble->count;
            geif_ensemble_prune_age(ensemble, delete_interval, time(NULL));
            if (verbose && before != ensemble->count) {
                printf("Age pruning (-D): dropped %zu stale sub-forests (%zu remaining).\n",
                       before - ensemble->count, ensemble->count);
            }
        }

        if (min_cat_rows > 0) {
            size_t before = ensemble->count;
            geif_ensemble_prune_categories(ensemble, min_cat_rows);
            if (verbose && before != ensemble->count) {
                printf("Pruned categories with fewer than %llu rows (%zu -> %zu categories retained).\n",
                       (unsigned long long)min_cat_rows, before, ensemble->count);
            }
        }

        if (ensemble->count == 0) {
            fprintf(stderr, "Error: Categorization requires at least one category in the model.\n");
            geif_ensemble_destroy(ensemble);
            geif_cat_filter_free(&cat_filter);
            return 1;
        }

        if (kill_outliers_count > 0) {
            if (verbose) {
                printf("Pruning %d outlier(s) per category and recalibrating ensemble...\n", kill_outliers_count);
            }
            geif_status_t status = geif_ensemble_remove_outliers(ensemble, (uint32_t)kill_outliers_count);
            if (status != GEIF_OK) {
                fprintf(stderr, "Error during outlier pruning (-k): %s\n", geif_status_str(status));
                geif_ensemble_destroy(ensemble);
                geif_cat_filter_free(&cat_filter);
                return 1;
            }
            if (threshold_is_percentage) {
                update_ensemble_percentage_scores(ensemble, outlier_percentage, &cat_filter, verbose);
            }
        }

        FILE *in_fp = xfopen(categorize_file, "r");
        if (!in_fp) {
            geif_ensemble_destroy(ensemble);
            geif_cat_filter_free(&cat_filter);
            return 1;
        }

        const char *out_target = output_file ? output_file : "-";
        FILE *out_fp = xfopen(out_target, "w");
        if (!out_fp) {
            xfclose(in_fp);
            geif_ensemble_destroy(ensemble);
            geif_cat_filter_free(&cat_filter);
            return 1;
        }

        const char *active_ignore = ignore_spec ? ignore_spec : (ensemble->ignore_dims_spec[0] ? ensemble->ignore_dims_spec : NULL);
        const char *active_include = include_spec ? include_spec : (ensemble->include_dims_spec[0] ? ensemble->include_dims_spec : NULL);
        const char *active_label = label_spec ? label_spec : (ensemble->label_dims_spec[0] ? ensemble->label_dims_spec : NULL);
        const char *active_category = category_spec ? category_spec : (ensemble->category_dims_spec[0] ? ensemble->category_dims_spec : NULL);

        char line[8192];
        char line_copy[8192];
        char *tokens[1024];
        uint32_t total_cols = 0;
        bool first_line_is_data = false;
        uint64_t line_num = 0;

        while (fgets(line, sizeof(line), in_fp)) {
            line_num++;
            if (line[0] == '#' || line[0] == '\r' || line[0] == '\n') continue;
            strncpy(line_copy, line, sizeof(line_copy) - 1);
            line_copy[sizeof(line_copy) - 1] = '\0';
            total_cols = tokenize_line(line_copy, delimiter, tokens, 1024);
            if (total_cols > 0) break;
        }

        if (total_cols == 0) {
            xfclose(in_fp);
            xfclose(out_fp);
            geif_ensemble_destroy(ensemble);
            geif_cat_filter_free(&cat_filter);
            return 0;
        }

        geif_column_config_t col_cfg;
        geif_column_config_init(&col_cfg, active_ignore, active_include, active_label, active_category);
        bool resolved = geif_column_config_resolve(&col_cfg, total_cols);
        bool direct_feature_mapping = false;

        if (!resolved || col_cfg.feature_dim_count != ensemble->dimensions) {
            if (total_cols == ensemble->dimensions) {
                direct_feature_mapping = true;
            } else {
                fprintf(stderr, "geif: error: input '%s' has %u columns, cannot map to model's %u feature dimensions\n",
                        categorize_file, total_cols, ensemble->dimensions);
                xfclose(in_fp);
                xfclose(out_fp);
                geif_column_config_free(&col_cfg);
                geif_ensemble_destroy(ensemble);
                geif_cat_filter_free(&cat_filter);
                return 1;
            }
        }

        if (!skip_header) {
            first_line_is_data = true;
        }

        uint32_t dims = ensemble->dimensions;
        double *vec = (double *)xmalloc(dims * sizeof(double));
        uint64_t analyzed = 0;
        agg_cat_bucket_t *agg_buckets = NULL;
        size_t agg_count = 0, agg_cap = 0;

        if (first_line_is_data) {
            char orig_line[8192];
            strncpy(orig_line, line, sizeof(orig_line) - 1);
            orig_line[sizeof(orig_line) - 1] = '\0';
            orig_line[strcspn(orig_line, "\r\n")] = 0;

            process_categorize_row(ensemble, &col_cfg, direct_feature_mapping, &cat_filter,
                                   tokens, total_cols, orig_line, line_num,
                                   list_separator, category_sep, label_sep, threshold,
                                   threshold_is_average, threshold_is_percentage, cli_outlier_score_given,
                                   point_tmpl, decimals, printf_format, print_dimension,
                                   low_rgb, high_rgb, vec, dims,
                                   out_fp, &analyzed, &agg_buckets, &agg_count, &agg_cap);
        }

        while (fgets(line, sizeof(line), in_fp)) {
            line_num++;
            if (line[0] == '#' || line[0] == '\r' || line[0] == '\n') continue;

            char orig_line[8192];
            strncpy(orig_line, line, sizeof(orig_line) - 1);
            orig_line[sizeof(orig_line) - 1] = '\0';
            orig_line[strcspn(orig_line, "\r\n")] = 0;

            strncpy(line_copy, line, sizeof(line_copy) - 1);
            line_copy[sizeof(line_copy) - 1] = '\0';

            uint32_t n_tok = tokenize_line(line_copy, delimiter, tokens, 1024);
            if (n_tok >= (direct_feature_mapping ? dims : total_cols)) {
                process_categorize_row(ensemble, &col_cfg, direct_feature_mapping, &cat_filter,
                                       tokens, n_tok, orig_line, line_num,
                                       list_separator, category_sep, label_sep, threshold,
                                       threshold_is_average, threshold_is_percentage, cli_outlier_score_given,
                                       point_tmpl, decimals, printf_format, print_dimension,
                                       low_rgb, high_rgb, vec, dims,
                                       out_fp, &analyzed, &agg_buckets, &agg_count, &agg_cap);
            }
        }

        if (ensemble->aggregate) {
            for (size_t b = 0; b < agg_count; b++) {
                score_and_print_categorize_vec(ensemble, &cat_filter, agg_buckets[b].category, agg_buckets[b].label,
                                               "", NULL, 0, b + 1,
                                               list_separator, category_sep, label_sep, threshold,
                                               threshold_is_average, threshold_is_percentage, cli_outlier_score_given,
                                               point_tmpl, decimals, printf_format, print_dimension,
                                               low_rgb, high_rgb, agg_buckets[b].summary, dims, out_fp, &analyzed);
                xfree(agg_buckets[b].summary);
            }
            xfree(agg_buckets);
        }

        xfree(vec);
        geif_column_config_free(&col_cfg);
        xfclose(in_fp);
        xfclose(out_fp);

        if (!learn_file && save_file) {
            ensemble->decimals = decimals;
            ensemble->input_delimiter = delimiter;
            ensemble->list_separator = list_separator;
            ensemble->has_header = skip_header;
            ensemble->aggregate = aggregate;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) {
                    ensemble->entries[i].forest->decimals = decimals;
                    ensemble->entries[i].forest->input_delimiter = delimiter;
                    ensemble->entries[i].forest->list_separator = list_separator;
                    ensemble->entries[i].forest->has_header = skip_header;
                    ensemble->entries[i].forest->aggregate = aggregate;
                }
            }
            geif_status_t status = geif_ensemble_save_json(ensemble, save_file);
            if (status != GEIF_OK) {
                fprintf(stderr, "Error saving model to '%s': %s\n", save_file, geif_status_str(status));
            } else if (verbose && strcmp(save_file, "-") != 0) {
                printf("Saved model to '%s'\n", save_file);
            }
        }
    }

    geif_cat_filter_free(&cat_filter);

    if (ensemble) geif_ensemble_destroy(ensemble);

    // Return 2 if outliers detected during analysis, 0 if clean success
    return (total_outliers > 0) ? 2 : 0;
}
