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

#define GEIF_VERSION "1.0.0"

static bool init_cat_filter(cat_filter_t *cf, const char *arg)
{
    if (!cf || !arg) return false;
    memset(cf, 0, sizeof(*cf));
    cf->active = true;

    const char *p = arg;
    while (*p == ' ' || *p == '\t') p++;

    if (strncmp(p, "-v", 2) == 0) {
        cf->invert = true;
        p += 2;
        while (*p == ' ' || *p == '\t') p++;
    }

    int rc = regcomp(&cf->regex, p, REG_EXTENDED | REG_NOSUB);
    if (rc != 0) {
        char errbuf[256];
        regerror(rc, &cf->regex, errbuf, sizeof(errbuf));
        fprintf(stderr, "geif: error: invalid category regex '%s': %s\n", p, errbuf);
        return false;
    }
    return true;
}

static bool match_cat_filter(const cat_filter_t *cf, const char *category)
{
    if (!cf || !cf->active) return true;
    const char *cat = category ? category : "";
    int rc = regexec(&cf->regex, cat, 0, NULL, 0);
    bool matches = (rc == 0);
    return cf->invert ? !matches : matches;
}

static void print_usage(const char *prog)
{
    printf("GEIF - Geometric Extended Isolation Forest (v%s)\n\n", GEIF_VERSION);
    printf("Usage:\n");
    printf("  Train a model:     %s -l <train.csv> -w <model.json> [-t trees] [-s samples] [-f sep] [-H] [-I range] [-U range] [-L range] [-C range] [-R min_rows]\n", prog);
    printf("  Score streaming:   %s -r <model.json> -a <test.csv> [-o <out.csv>] [-T threshold] [-F regex] [-N tmpl] [-M tmpl] [-p tmpl] [-S]\n", prog);
    printf("  Inspect model:     %s -r <model.json> -q\n\n", prog);
    printf("Options:\n");
    printf("  -l <file>      Train forest from input CSV file (use '-' for stdin)\n");
    printf("  -a <file>      Analyze / score samples from input CSV (use '-' for stdin)\n");
    printf("  -w <file>      Save trained model to JSON file (use '-' for stdout)\n");
    printf("  -r <file>      Load trained model from JSON file (use '-' for stdin)\n");
    printf("  -o <file>      Output file for scores (default: stdout, '-' for stdout)\n");
    printf("  -O <thresh>    Outlier threshold: float [0..1], 'average', percentage (e.g. '80%%'), or scaled (e.g. '0.65s')\n");
    printf("  -B, --algo <s> Algorithm engine: ceif (default), bubble, exemplar, voronoi\n");
    printf("  -T [margin]    Generate synthetic test grid for population drift visualization (margin: e.g. 0.1)\n");
    printf("  -k             Prune most extreme outlier from model reservoir and recalibrate (repeatable)\n");
    printf("  -g <file>      Configuration / RC file (overrides ~/.geifrc and ~/.ceifrc)\n");
    printf("  -t <int>       Number of trees in forest (default: 100)\n");
    printf("  -i <int>       Test grid sample intervals (default: 256) or tree count alias\n");
    printf("  -s <int>       Number of samples per tree (default: 256)\n");
    printf("  -m <fmt|int>   Dimension format string (e.g. \"%%'.0f\") or max depth cap\n");
    printf("  -f <char>      Input field delimiter (default: ',')\n");
    printf("  -e <char>      List separator for output / field delimiter fallback\n");
    printf("  -H             Skip header line in input CSV\n");
    printf("  -I <range>     Ignore column indices/ranges from features (e.g. '12' or '1,3,5')\n");
    printf("  -U <range>     Use/include only specified column indices/ranges (e.g. '2-10')\n");
    printf("  -L <range>     Label column indices/ranges (excluded from features, kept as label)\n");
    printf("  -C <range>     Category column indices/ranges (e.g. '12' or '2-4')\n");
    printf("  -F <filter>    Category filter regex during scoring (e.g. '-v ^5' or '^(5|6)$')\n");
    printf("  -R <int>       Minimum row count per category required to train sub-forest\n");
    printf("  -D <interval>  Drop / prune categories older than interval (e.g. '30d', '7d', '24h', '3600s')\n");
    printf("  -N <tmpl>      Output template for NEW / unseen categories during analysis\n");
    printf("  -M <tmpl>      Output template for MISSED categories (trained but absent in test data)\n");
    printf("  -p <tmpl>      Output template for scored rows (%%s=score, %%l=label, %%c=cat, %%m=metric/dims, %%d=vector, %%e=attrib, %%a=avg)\n");
    printf("  -v [tmpl]      Output template for inlier/average rows, or verbose flag\n");
    printf("  -d <int>       Floating point decimal precision (default: 6)\n");
    printf("  -j <tmpl>      Per-dimension expansion template for %%m (%%d=val, %%a=avg, %%e=attrib, %%i=index)\n");
    printf("  -W             Enable metric auto-scaling (accepted for backward compatibility)\n");
    printf("  -A             Aggregate ensemble processing (accepted for backward compatibility)\n");
    printf("  -S             Silent / outliers only (suppress inliers from output)\n");
    printf("  -q             Print model summary / diagnostics and exit\n");
    printf("  -h             Show this help message and exit\n");
}

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

static void process_scoring_row(geif_ensemble_t *ensemble,
                                const geif_column_config_t *col_cfg,
                                const cat_filter_t *cat_filter,
                                char **tokens,
                                uint32_t n_tok,
                                const char *orig_line,
                                char list_sep,
                                char cat_sep,
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
    char cat_buf[256];
    char label_buf[256];
    geif_extract_category(col_cfg, tokens, n_tok, cat_sep, cat_buf, sizeof(cat_buf));
    geif_extract_label(col_cfg, tokens, n_tok, cat_sep, label_buf, sizeof(label_buf));

    // Category filter check (-F)
    if (!match_cat_filter(cat_filter, cat_buf)) {
        return; // Filtered out
    }

    geif_forest_t *sub_forest = geif_ensemble_find(ensemble, cat_buf);
    if (!sub_forest) {
        // Unseen / NEW category!
        (*analyzed)++;
        (*total_outliers)++;

        if (new_cat_tmpl) {
            geif_template_context_t ctx = {
                .orig_line        = orig_line,
                .label            = label_buf,
                .category         = cat_buf,
                .score            = 1.0,
                .metric_depth     = 0.0,
                .d_out            = 999.0,
                .H_max            = 0.0,
                .is_outlier       = 1,
                .timestamp        = time(NULL),
                .total_rows       = 0,
                .analyzed_rows    = *analyzed,
                .vector           = NULL,
                .averages         = NULL,
                .attr_scores      = NULL,
                .vector_dim       = 0,
                .raw_values       = tokens,
                .raw_value_count  = n_tok,
                .list_separator   = list_sep,
                .decimals         = decimals,
                .printf_format    = printf_format,
                .print_dimension  = print_dimension,
                .low_rgb          = low_rgb,
                .high_rgb         = high_rgb
            };
            char out_buf[8192];
            geif_format_template(out_buf, sizeof(out_buf), new_cat_tmpl, &ctx);
            fprintf(out_fp, "%s\n", out_buf);
        } else {
            fprintf(out_fp, "%s%c%.*f%c%d%c%.*f%c%.*f\n",
                    orig_line, list_sep, decimals, 1.0, list_sep,
                    1, list_sep, decimals, 0.0, list_sep, decimals, 999.0);
        }
        return;
    }

    // Known category: extract features and score
    if (geif_extract_features(col_cfg, tokens, n_tok, vec)) {
        double score = 0.0, H_metric = 0.0, d_out = 0.0;
        geif_ensemble_score_detailed(ensemble, cat_buf, vec, &score, &H_metric, &d_out);

        double eff_threshold = threshold;
        if (threshold_is_average) {
            eff_threshold = (sub_forest->average_score > 0.0) ? sub_forest->average_score : 0.5;
        } else if (threshold_is_percentage) {
            eff_threshold = (sub_forest->percentage_score > 0.0) ? sub_forest->percentage_score : 0.5;
        }

        (*analyzed)++;
        bool is_outlier = (score >= eff_threshold);
        if (is_outlier) (*total_outliers)++;

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
            .score            = score,
            .metric_depth     = H_metric,
            .d_out            = d_out,
            .H_max            = sub_forest ? sub_forest->H_max : 0.0,
            .is_outlier       = is_outlier ? 1 : 0,
            .timestamp        = time(NULL),
            .total_rows       = sub_forest ? sub_forest->total_rows_seen : 0,
            .analyzed_rows    = *analyzed,
            .vector           = vec,
            .averages         = averages,
            .attr_scores      = attr_scores,
            .vector_dim       = dims,
            .raw_values       = tokens,
            .raw_value_count  = n_tok,
            .list_separator   = list_sep,
            .decimals         = decimals,
            .printf_format    = printf_format,
            .print_dimension  = print_dimension,
            .low_rgb          = low_rgb,
            .high_rgb         = high_rgb
        };

        const char *tmpl_to_use = NULL;
        if (is_outlier) {
            tmpl_to_use = point_tmpl;
        } else {
            if (average_tmpl) {
                tmpl_to_use = average_tmpl;
            } else if (!silent_outliers) {
                tmpl_to_use = point_tmpl;
            }
        }

        if (tmpl_to_use) {
            char out_buf[8192];
            geif_format_template(out_buf, sizeof(out_buf), tmpl_to_use, &ctx);
            fprintf(out_fp, "%s\n", out_buf);
        } else if (!silent_outliers || is_outlier) {
            fprintf(out_fp, "%s%c%.*f%c%d%c%.*f%c%.*f\n",
                    orig_line, list_sep, decimals, score, list_sep,
                    is_outlier ? 1 : 0, list_sep, decimals, H_metric, list_sep, decimals, d_out);
        }

        if (averages) free(averages);
        if (attr_scores) free(attr_scores);
    }
}

static void update_ensemble_percentage_scores(geif_ensemble_t *ens, double pct, bool verbose)
{
    if (!ens) return;
    for (size_t c = 0; c < ens->count; c++) {
        geif_forest_t *sf = ens->entries[c].forest;
        if (sf) {
            sf->percentage_score = geif_forest_calculate_percentile_score(sf, pct);
            if (verbose) {
                printf("Percentage score for '%s': %.6f (%.2f%% of samples have lower score)\n",
                       ens->entries[c].category[0] ? ens->entries[c].category : "(default)",
                       sf->percentage_score, pct);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    const char *learn_file   = NULL;
    const char *analyze_file = NULL;
    const char *save_file    = NULL;
    const char *load_file    = NULL;
    const char *output_file  = NULL;
    const char *ignore_spec   = NULL;
    const char *include_spec  = NULL;
    const char *label_spec    = NULL;
    const char *category_spec = NULL;
    const char *filter_spec   = NULL;
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
    bool skip_header         = false;
    double threshold         = 0.5;
    bool threshold_is_average = false;
    bool threshold_is_percentage = false;
    bool threshold_is_scaled = false;
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
        {"trees",        required_argument, 0, 't'},
        {"samples",      required_argument, 0, 's'},
        {"output",       required_argument, 0, 'o'},
        {"help",         no_argument,       0, 'h'},
        {"version",      no_argument,       0, 'v'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "l:a:w:r:o:T::O:t:i:s:m:f:e:HqvhI:U:L:C:F:R:N:M:p:SD:d:j:v::WAkg:B:", long_options, NULL)) != -1) {
        switch (opt) {
        case 'B': config.algo = geif_algo_from_name(optarg); break;
        case 'l': learn_file = optarg; break;
        case 'a': analyze_file = optarg; break;
        case 'w': save_file = optarg; break;
        case 'r': load_file = optarg; break;
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
                threshold_is_scaled = false;
            } else {
                threshold_is_average = false;
                size_t olen = strlen(optarg);
                if (olen > 0 && optarg[olen - 1] == '%') {
                    threshold_is_percentage = true;
                    threshold_is_scaled = false;
                    outlier_percentage = atof(optarg);
                } else if (olen > 0 && optarg[olen - 1] == 's') {
                    threshold_is_percentage = false;
                    threshold_is_scaled = true;
                    threshold = atof(optarg);
                } else {
                    threshold_is_percentage = false;
                    threshold_is_scaled = false;
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
            if (!field_sep_explicit) {
                delimiter = optarg[0];
            }
            break;
        case 'H': skip_header = true; break;
        case 'I': ignore_spec = optarg; break;
        case 'U': include_spec = optarg; break;
        case 'L': label_spec = optarg; break;
        case 'C': category_spec = optarg; break;
        case 'F':
            filter_spec = optarg;
            if (!init_cat_filter(&cat_filter, filter_spec)) {
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
            new_cat_tmpl = optarg;
            break;
        case 'M':
            missed_cat_tmpl = optarg;
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
            // Ensemble aggregate processing (accepted for CEIF compatibility)
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
    if (!cli_outlier_score_given && rc_cfg.outlier_score_set) {
        cli_outlier_score_given = true;
        strncpy(cli_outlier_score_spec, rc_cfg.outlier_score_spec, sizeof(cli_outlier_score_spec) - 1);
        if (strcmp(rc_cfg.outlier_score_spec, "average") == 0) {
            threshold_is_average = true;
            threshold_is_percentage = false;
            threshold_is_scaled = false;
        } else {
            threshold_is_average = false;
            size_t olen = strlen(rc_cfg.outlier_score_spec);
            if (olen > 0 && rc_cfg.outlier_score_spec[olen - 1] == '%') {
                threshold_is_percentage = true;
                threshold_is_scaled = false;
                outlier_percentage = atof(rc_cfg.outlier_score_spec);
            } else if (olen > 0 && rc_cfg.outlier_score_spec[olen - 1] == 's') {
                threshold_is_percentage = false;
                threshold_is_scaled = true;
                threshold = atof(rc_cfg.outlier_score_spec);
            } else {
                threshold_is_percentage = false;
                threshold_is_scaled = false;
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
        if (cli_outlier_score_given) {
            strncpy(ensemble->outlier_score_spec, cli_outlier_score_spec, sizeof(ensemble->outlier_score_spec) - 1);
            ensemble->scale_score = threshold_is_scaled;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) ensemble->entries[i].forest->scale_score = threshold_is_scaled;
            }
        } else if (ensemble->outlier_score_spec[0] != '\0') {
            if (strcmp(ensemble->outlier_score_spec, "average") == 0) {
                threshold_is_average = true;
                threshold_is_percentage = false;
                threshold_is_scaled = false;
            } else {
                threshold_is_average = false;
                size_t olen = strlen(ensemble->outlier_score_spec);
                if (olen > 0 && ensemble->outlier_score_spec[olen - 1] == '%') {
                    threshold_is_percentage = true;
                    threshold_is_scaled = false;
                    outlier_percentage = atof(ensemble->outlier_score_spec);
                } else if (olen > 0 && ensemble->outlier_score_spec[olen - 1] == 's') {
                    threshold_is_percentage = false;
                    threshold_is_scaled = true;
                    threshold = atof(ensemble->outlier_score_spec);
                } else {
                    threshold_is_percentage = false;
                    threshold_is_scaled = false;
                    threshold = atof(ensemble->outlier_score_spec);
                }
            }
            ensemble->scale_score = threshold_is_scaled;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) ensemble->entries[i].forest->scale_score = threshold_is_scaled;
            }
        }
        if (threshold_is_percentage) {
            update_ensemble_percentage_scores(ensemble, outlier_percentage, verbose);
        }
        if (cli_decimals_given || rc_cfg.decimals_set) {
            ensemble->decimals = decimals;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) ensemble->entries[i].forest->decimals = decimals;
            }
        } else if (ensemble->decimals > 0) {
            decimals = ensemble->decimals;
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

    // Standalone model maintenance (-D pruning and/or -k outlier recalibration): -r <model> [-D <interval>] [-k] -w <model>
    if (!learn_file && !analyze_file && (delete_interval > 0 || kill_outliers_count > 0)) {
        if (!ensemble) {
            fprintf(stderr, "Error: maintenance operations (-D, -k) require a loaded model via -r <model.json>\n");
            return 1;
        }
        if (delete_interval > 0) {
            size_t before = ensemble->count;
            geif_ensemble_prune_age(ensemble, delete_interval, time(NULL));
            if (verbose) {
                printf("Age pruning (-D): dropped %zu stale sub-forests (%zu remaining).\n",
                       before - ensemble->count, ensemble->count);
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
                update_ensemble_percentage_scores(ensemble, outlier_percentage, verbose);
            }
        }
        if (save_file) {
            ensemble->decimals = decimals;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) ensemble->entries[i].forest->decimals = decimals;
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

        char line[8192];
        char line_copy[8192];
        char *tokens[1024];
        uint32_t total_cols = 0;
        bool first_line_is_data = false;

        // Read first non-empty line to detect columns and handle header
        while (fgets(line, sizeof(line), fp)) {
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
        }

        // Save column metadata into ensemble
        ensemble->total_input_cols = total_cols;
        ensemble->decimals = decimals;
        ensemble->scale_score = threshold_is_scaled;
        if (cli_outlier_score_given) {
            strncpy(ensemble->outlier_score_spec, cli_outlier_score_spec, sizeof(ensemble->outlier_score_spec) - 1);
        }
        if (active_ignore) strncpy(ensemble->ignore_dims_spec, active_ignore, sizeof(ensemble->ignore_dims_spec) - 1);
        if (active_include) strncpy(ensemble->include_dims_spec, active_include, sizeof(ensemble->include_dims_spec) - 1);
        if (active_label) strncpy(ensemble->label_dims_spec, active_label, sizeof(ensemble->label_dims_spec) - 1);
        if (active_category) strncpy(ensemble->category_dims_spec, active_category, sizeof(ensemble->category_dims_spec) - 1);

        if (!skip_header) {
            first_line_is_data = true;
        }

        double *vec = (double *)xmalloc(dims * sizeof(double));
        char cat_buf[256];
        uint64_t rows = 0;

        if (first_line_is_data) {
            geif_extract_category(&col_cfg, tokens, total_cols, category_sep, cat_buf, sizeof(cat_buf));
            if (geif_extract_features(&col_cfg, tokens, total_cols, vec)) {
                geif_ensemble_feed(ensemble, cat_buf, vec);
                rows++;
            }
        }

        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '#' || line[0] == '\r' || line[0] == '\n') continue;
            strncpy(line_copy, line, sizeof(line_copy) - 1);
            line_copy[sizeof(line_copy) - 1] = '\0';
            uint32_t n_tok = tokenize_line(line_copy, delimiter, tokens, 1024);
            if (n_tok >= total_cols) {
                geif_extract_category(&col_cfg, tokens, n_tok, category_sep, cat_buf, sizeof(cat_buf));
                if (geif_extract_features(&col_cfg, tokens, n_tok, vec)) {
                    geif_ensemble_feed(ensemble, cat_buf, vec);
                    rows++;
                }
            }
        }
        xfree(vec);
        xfclose(fp);
        geif_column_config_free(&col_cfg);

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
            strncpy(ensemble->outlier_score_spec, cli_outlier_score_spec, sizeof(ensemble->outlier_score_spec) - 1);
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
            update_ensemble_percentage_scores(ensemble, outlier_percentage, verbose);
        }

        // Save trained model if requested
        if (save_file) {
            ensemble->decimals = decimals;
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) ensemble->entries[i].forest->decimals = decimals;
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
                update_ensemble_percentage_scores(ensemble, outlier_percentage, verbose);
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

        if (first_line_is_data) {
            char orig_line[8192];
            strncpy(orig_line, line, sizeof(orig_line) - 1);
            orig_line[sizeof(orig_line) - 1] = '\0';
            orig_line[strcspn(orig_line, "\r\n")] = 0;

            process_scoring_row(ensemble, &col_cfg, &cat_filter, tokens, total_cols,
                                orig_line, list_separator, category_sep, threshold,
                                threshold_is_average, threshold_is_percentage, silent_outliers, point_tmpl, average_tmpl,
                                new_cat_tmpl, decimals, printf_format, print_dimension,
                                low_rgb, high_rgb, vec, dims,
                                out_fp, &analyzed, &total_outliers);
        }

        while (fgets(line, sizeof(line), in_fp)) {
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
                                    orig_line, list_separator, category_sep, threshold,
                                    threshold_is_average, threshold_is_percentage, silent_outliers, point_tmpl, average_tmpl,
                                    new_cat_tmpl, decimals, printf_format, print_dimension,
                                    low_rgb, high_rgb, vec, dims,
                                    out_fp, &analyzed, &total_outliers);
            }
        }

        // Emit Missed categories (-M) if specified
        if (missed_cat_tmpl) {
            for (size_t i = 0; i < ensemble->count; i++) {
                const char *cat_name = ensemble->entries[i].category;
                if (!match_cat_filter(&cat_filter, cat_name)) {
                    continue;
                }
                if (!ensemble->entries[i].seen_in_analysis) {
                    total_outliers++;
                    geif_template_context_t ctx = {
                        .orig_line        = cat_name,
                        .label            = "",
                        .category         = cat_name,
                        .score            = 1.0,
                        .metric_depth     = 0.0,
                        .d_out            = 0.0,
                        .H_max            = ensemble->entries[i].forest ? ensemble->entries[i].forest->H_max : 0.0,
                        .is_outlier       = 1,
                        .timestamp        = time(NULL),
                        .total_rows       = ensemble->entries[i].total_rows,
                        .analyzed_rows    = analyzed,
                        .vector           = NULL,
                        .averages         = NULL,
                        .attr_scores      = NULL,
                        .vector_dim       = 0,
                        .raw_values       = NULL,
                        .raw_value_count  = 0,
                        .list_separator   = list_separator,
                        .decimals         = decimals,
                        .printf_format    = printf_format,
                        .print_dimension  = print_dimension,
                        .low_rgb          = low_rgb,
                        .high_rgb         = high_rgb
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
            for (size_t i = 0; i < ensemble->count; i++) {
                if (ensemble->entries[i].forest) ensemble->entries[i].forest->decimals = decimals;
            }
            geif_status_t status = geif_ensemble_save_json(ensemble, save_file);
            if (status != GEIF_OK) {
                fprintf(stderr, "Error saving model to '%s': %s\n", save_file, geif_status_str(status));
            } else if (verbose && strcmp(save_file, "-") != 0) {
                printf("Saved model to '%s'\n", save_file);
            }
        }
    }

    if (cat_filter.active) {
        regfree(&cat_filter.regex);
    }

    if (ensemble) geif_ensemble_destroy(ensemble);

    // Return 2 if outliers detected during analysis, 0 if clean success
    return (total_outliers > 0) ? 2 : 0;
}
