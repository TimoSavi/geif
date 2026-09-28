/**
 * @file main.c
 * @brief GEIF CLI frontend for training, inference, and model management.
 */

#include "geif/geif.h"
#include "xmalloc.h"
#include "columns.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <math.h>

#define GEIF_VERSION "1.0.0"

static void print_usage(const char *prog)
{
    printf("GEIF - Geometric Extended Isolation Forest (v%s)\n\n", GEIF_VERSION);
    printf("Usage:\n");
    printf("  Train a model:     %s -l <train.csv> -w <model.json> [-t trees] [-s samples] [-f sep] [-H] [-I range] [-U range] [-L range]\n", prog);
    printf("  Score streaming:   %s -r <model.json> -a <test.csv> [-o <out.csv>] [-T threshold] [-f sep] [-H]\n", prog);
    printf("  Inspect model:     %s -r <model.json> -q\n\n", prog);
    printf("Options:\n");
    printf("  -l <file>      Train forest from input CSV file (use '-' for stdin)\n");
    printf("  -a <file>      Analyze / score samples from input CSV (use '-' for stdin)\n");
    printf("  -w <file>      Save trained model to JSON file (use '-' for stdout)\n");
    printf("  -r <file>      Load trained model from JSON file (use '-' for stdin)\n");
    printf("  -o <file>      Output file for scores (default: stdout, '-' for stdout)\n");
    printf("  -T <float>     Outlier decision threshold in [0.0, 1.0] (default: 0.5)\n");
    printf("  -t, -i <int>   Number of trees in forest (default: 100)\n");
    printf("  -s <int>       Number of samples per tree (default: 256)\n");
    printf("  -m <int>       Maximum tree depth cap (default: 16)\n");
    printf("  -f <char>      Input field delimiter (default: ',')\n");
    printf("  -e <char>      List separator for output / field delimiter fallback\n");
    printf("  -H             Skip header line in input CSV\n");
    printf("  -I <range>     Ignore column indices/ranges from features (e.g. '12' or '1,3,5')\n");
    printf("  -U <range>     Use/include only specified column indices/ranges (e.g. '2-10')\n");
    printf("  -L <range>     Label column indices/ranges (excluded from features, kept as label)\n");
    printf("  -C <range>     Category column indices/ranges\n");
    printf("  -q             Print model summary / diagnostics and exit\n");
    printf("  -v             Verbose output\n");
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
    char delimiter           = ',';
    char list_separator      = ',';
    bool field_sep_explicit  = false;
    bool skip_header         = false;
    double threshold         = 0.5;
    bool query_mode          = false;
    bool verbose             = false;

    geif_config_t config = geif_config_default();

    int opt;
    while ((opt = getopt(argc, argv, "l:a:w:r:o:T:t:i:s:m:f:e:HqvhI:U:L:C:")) != -1) {
        switch (opt) {
        case 'l': learn_file = optarg; break;
        case 'a': analyze_file = optarg; break;
        case 'w': save_file = optarg; break;
        case 'r': load_file = optarg; break;
        case 'o': output_file = optarg; break;
        case 'T': threshold = atof(optarg); break;
        case 't':
        case 'i': config.tree_count = (uint32_t)atoi(optarg); break;
        case 's': config.samples_per_tree = (uint32_t)atoi(optarg); break;
        case 'm': config.max_depth = (uint32_t)atoi(optarg); break;
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
        case 'q': query_mode = true; break;
        case 'v': verbose = true; break;
        case 'h': print_usage(argv[0]); return 0;
        default:  print_usage(argv[0]); return 1;
        }
    }

    geif_forest_t *forest = NULL;

    // Mode: Load existing model
    if (load_file) {
        geif_status_t status = geif_forest_load_json(&forest, load_file);
        if (status != GEIF_OK) {
            fprintf(stderr, "Error loading model from '%s': %s\n", load_file, geif_status_str(status));
            return 1;
        }
        if (verbose) printf("Loaded model from '%s' (%u trees, %u dimensions)\n",
                            load_file, forest->tree_count, forest->dimensions);
    }

    // Mode: Model info query (-q)
    if (query_mode) {
        if (!forest) {
            fprintf(stderr, "Error: -q requires a loaded model via -r <model.json>\n");
            return 1;
        }
        char summary[1024];
        geif_forest_summary(forest, summary, sizeof(summary));
        printf("%s", summary);
        if (forest->total_input_cols > 0) {
            printf("  Input Columns:       %u\n", forest->total_input_cols);
        }
        if (forest->label_dims_spec[0] != '\0') {
            printf("  Label Columns (-L):  %s\n", forest->label_dims_spec);
        }
        if (forest->include_dims_spec[0] != '\0') {
            printf("  Include Columns (-U): %s\n", forest->include_dims_spec);
        }
        if (forest->ignore_dims_spec[0] != '\0') {
            printf("  Ignore Columns (-I):  %s\n", forest->ignore_dims_spec);
        }
        geif_forest_destroy(forest);
        return 0;
    }

    // Mode: Train new model (-l)
    if (learn_file) {
        FILE *fp = xfopen(learn_file, "r");
        if (!fp) {
            if (forest) geif_forest_destroy(forest);
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
            if (forest) geif_forest_destroy(forest);
            return 1;
        }

        geif_column_config_t col_cfg;
        geif_column_config_init(&col_cfg, ignore_spec, include_spec, label_spec, category_spec);
        if (!geif_column_config_resolve(&col_cfg, total_cols)) {
            fprintf(stderr, "geif: error: no active feature dimensions remaining after applying column masks (-I, -U, -L)\n");
            xfclose(fp);
            geif_column_config_free(&col_cfg);
            if (forest) geif_forest_destroy(forest);
            return 1;
        }

        uint32_t dims = col_cfg.feature_dim_count;

        if (!forest) {
            geif_status_t status = geif_forest_create(&forest, dims, &config);
            if (status != GEIF_OK) {
                fprintf(stderr, "Error initializing forest: %s\n", geif_status_str(status));
                xfclose(fp);
                geif_column_config_free(&col_cfg);
                return 1;
            }
        }

        // Save column metadata into forest
        forest->total_input_cols = total_cols;
        if (ignore_spec) strncpy(forest->ignore_dims_spec, ignore_spec, sizeof(forest->ignore_dims_spec) - 1);
        if (include_spec) strncpy(forest->include_dims_spec, include_spec, sizeof(forest->include_dims_spec) - 1);
        if (label_spec) strncpy(forest->label_dims_spec, label_spec, sizeof(forest->label_dims_spec) - 1);
        if (category_spec) strncpy(forest->category, category_spec, sizeof(forest->category) - 1);

        if (!skip_header) {
            first_line_is_data = true;
        }

        double *vec = (double *)xmalloc(dims * sizeof(double));
        uint64_t rows = 0;

        if (first_line_is_data) {
            if (geif_extract_features(&col_cfg, tokens, total_cols, vec)) {
                geif_forest_feed(forest, vec);
                rows++;
            }
        }

        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '#' || line[0] == '\r' || line[0] == '\n') continue;
            strncpy(line_copy, line, sizeof(line_copy) - 1);
            line_copy[sizeof(line_copy) - 1] = '\0';
            uint32_t n_tok = tokenize_line(line_copy, delimiter, tokens, 1024);
            if (n_tok >= total_cols && geif_extract_features(&col_cfg, tokens, n_tok, vec)) {
                geif_forest_feed(forest, vec);
                rows++;
            }
        }
        xfree(vec);
        xfclose(fp);
        geif_column_config_free(&col_cfg);

        if (verbose) printf("Ingested %llu rows from '%s' (%u feature dimensions). Training forest...\n",
                            (unsigned long long)rows, learn_file, dims);

        geif_status_t status = geif_forest_train(forest);
        if (status != GEIF_OK) {
            fprintf(stderr, "Error training forest: %s\n", geif_status_str(status));
            geif_forest_destroy(forest);
            return 1;
        }

        if (verbose) {
            printf("Training complete. Universal scale H_max = %.6f, Nominal Spacing = %.6f\n",
                   forest->H_max, forest->delta_nominal);
        }

        // Save trained model if requested
        if (save_file) {
            status = geif_forest_save_json(forest, save_file);
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
        if (!forest) {
            fprintf(stderr, "Error: Analysis requires a model. Train with -l or load with -r.\n");
            return 1;
        }

        FILE *in_fp = xfopen(analyze_file, "r");
        if (!in_fp) {
            geif_forest_destroy(forest);
            return 1;
        }

        const char *out_target = output_file ? output_file : "-";
        FILE *out_fp = xfopen(out_target, "w");
        if (!out_fp) {
            xfclose(in_fp);
            geif_forest_destroy(forest);
            return 1;
        }

        // Column configuration: CLI flags override loaded model metadata
        const char *active_ignore = ignore_spec ? ignore_spec : (forest->ignore_dims_spec[0] ? forest->ignore_dims_spec : NULL);
        const char *active_include = include_spec ? include_spec : (forest->include_dims_spec[0] ? forest->include_dims_spec : NULL);
        const char *active_label = label_spec ? label_spec : (forest->label_dims_spec[0] ? forest->label_dims_spec : NULL);
        const char *active_category = category_spec ? category_spec : (forest->category[0] ? forest->category : NULL);

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
            geif_forest_destroy(forest);
            return 0;
        }

        geif_column_config_t col_cfg;
        geif_column_config_init(&col_cfg, active_ignore, active_include, active_label, active_category);
        if (!geif_column_config_resolve(&col_cfg, total_cols)) {
            fprintf(stderr, "geif: error: no active feature dimensions in analysis input '%s'\n", analyze_file);
            xfclose(in_fp);
            xfclose(out_fp);
            geif_column_config_free(&col_cfg);
            geif_forest_destroy(forest);
            return 1;
        }

        if (col_cfg.feature_dim_count != forest->dimensions) {
            fprintf(stderr, "geif: error: input '%s' has %u active feature dimensions, but model expects %u\n",
                    analyze_file, col_cfg.feature_dim_count, forest->dimensions);
            xfclose(in_fp);
            xfclose(out_fp);
            geif_column_config_free(&col_cfg);
            geif_forest_destroy(forest);
            return 1;
        }

        if (!skip_header) {
            first_line_is_data = true;
        }

        uint32_t dims = forest->dimensions;
        double *vec = (double *)xmalloc(dims * sizeof(double));
        uint64_t analyzed = 0;

        if (first_line_is_data) {
            char orig_line[8192];
            strncpy(orig_line, line, sizeof(orig_line) - 1);
            orig_line[sizeof(orig_line) - 1] = '\0';
            orig_line[strcspn(orig_line, "\r\n")] = 0;

            if (geif_extract_features(&col_cfg, tokens, total_cols, vec)) {
                double score = 0.0, H_metric = 0.0, d_out = 0.0;
                geif_forest_score_detailed(forest, vec, &score, &H_metric, &d_out);

                analyzed++;
                bool is_outlier = (score >= threshold);
                if (is_outlier) total_outliers++;

                fprintf(out_fp, "%s%c%.6f%c%d%c%.6f%c%.6f\n",
                        orig_line, list_separator, score, list_separator,
                        is_outlier ? 1 : 0, list_separator, H_metric, list_separator, d_out);
            }
        }

        while (fgets(line, sizeof(line), in_fp)) {
            if (line[0] == '#' || line[0] == '\r' || line[0] == '\n') continue;

            // Keep original line without newline for reporting
            char orig_line[8192];
            strncpy(orig_line, line, sizeof(orig_line) - 1);
            orig_line[sizeof(orig_line) - 1] = '\0';
            orig_line[strcspn(orig_line, "\r\n")] = 0;

            strncpy(line_copy, line, sizeof(line_copy) - 1);
            line_copy[sizeof(line_copy) - 1] = '\0';

            uint32_t n_tok = tokenize_line(line_copy, delimiter, tokens, 1024);
            if (n_tok >= total_cols && geif_extract_features(&col_cfg, tokens, n_tok, vec)) {
                double score = 0.0, H_metric = 0.0, d_out = 0.0;
                geif_forest_score_detailed(forest, vec, &score, &H_metric, &d_out);

                analyzed++;
                bool is_outlier = (score >= threshold);
                if (is_outlier) total_outliers++;

                fprintf(out_fp, "%s%c%.6f%c%d%c%.6f%c%.6f\n",
                        orig_line, list_separator, score, list_separator,
                        is_outlier ? 1 : 0, list_separator, H_metric, list_separator, d_out);
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
    }

    if (forest) geif_forest_destroy(forest);

    // Return 2 if outliers detected during analysis, 0 if clean success
    return (total_outliers > 0) ? 2 : 0;
}
