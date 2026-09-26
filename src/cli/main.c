/**
 * @file main.c
 * @brief GEIF CLI frontend for training, inference, and model management.
 */

#include "geif/geif.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

#define GEIF_VERSION "1.0.0"

static void print_usage(const char *prog)
{
    printf("GEIF - Geometric Extended Isolation Forest (v%s)\n\n", GEIF_VERSION);
    printf("Usage:\n");
    printf("  Train a model:     %s -l <train.csv> -w <model.json> [-i trees] [-s samples]\n", prog);
    printf("  Score streaming:   %s -r <model.json> -a <test.csv> [-o <out.csv>] [-T threshold]\n", prog);
    printf("  Inspect model:     %s -r <model.json> -q\n\n", prog);
    printf("Options:\n");
    printf("  -l <file>      Train forest from input CSV file\n");
    printf("  -a <file>      Analyze / score samples from input CSV file (use '-' for stdin)\n");
    printf("  -w <file>      Save trained model to JSON file\n");
    printf("  -r <file>      Load trained model from JSON file\n");
    printf("  -o <file>      Output file for scores (default: stdout)\n");
    printf("  -T <float>     Outlier decision threshold in [0.0, 1.0] (default: 0.5)\n");
    printf("  -i <int>       Number of trees in forest (default: 100)\n");
    printf("  -s <int>       Number of samples per tree (default: 256)\n");
    printf("  -m <int>       Maximum tree depth cap (default: 16)\n");
    printf("  -q             Print model summary / diagnostics and exit\n");
    printf("  -v             Verbose output\n");
    printf("  -h             Show this help message and exit\n");
}

static uint32_t detect_dimensions(FILE *fp)
{
    char line[4096];
    long pos = ftell(fp);
    if (!fgets(line, sizeof(line), fp)) {
        return 0;
    }
    fseek(fp, pos, SEEK_SET);

    uint32_t count = 0;
    char *token = strtok(line, ",\t \r\n");
    while (token) {
        count++;
        token = strtok(NULL, ",\t \r\n");
    }
    return count;
}

static int parse_line(char *line, double *vec, uint32_t expected_dims)
{
    uint32_t idx = 0;
    char *token = strtok(line, ",\t \r\n");
    while (token && idx < expected_dims) {
        char *endptr;
        vec[idx] = strtod(token, &endptr);
        if (endptr == token) return 0;
        idx++;
        token = strtok(NULL, ",\t \r\n");
    }
    return (idx == expected_dims);
}

int main(int argc, char *argv[])
{
    const char *learn_file   = NULL;
    const char *analyze_file = NULL;
    const char *save_file    = NULL;
    const char *load_file    = NULL;
    const char *output_file  = NULL;
    double threshold         = 0.5;
    bool query_mode          = false;
    bool verbose             = false;

    geif_config_t config = geif_config_default();

    int opt;
    while ((opt = getopt(argc, argv, "l:a:w:r:o:T:i:s:m:qvh")) != -1) {
        switch (opt) {
        case 'l': learn_file = optarg; break;
        case 'a': analyze_file = optarg; break;
        case 'w': save_file = optarg; break;
        case 'r': load_file = optarg; break;
        case 'o': output_file = optarg; break;
        case 'T': threshold = atof(optarg); break;
        case 'i': config.tree_count = (uint32_t)atoi(optarg); break;
        case 's': config.samples_per_tree = (uint32_t)atoi(optarg); break;
        case 'm': config.max_depth = (uint32_t)atoi(optarg); break;
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
        geif_forest_destroy(forest);
        return 0;
    }

    // Mode: Train new model (-l)
    if (learn_file) {
        FILE *fp = fopen(learn_file, "r");
        if (!fp) {
            fprintf(stderr, "Error opening training file '%s'\n", learn_file);
            if (forest) geif_forest_destroy(forest);
            return 1;
        }

        uint32_t dims = detect_dimensions(fp);
        if (dims == 0) {
            fprintf(stderr, "Error: could not detect dimensions in '%s'\n", learn_file);
            fclose(fp);
            if (forest) geif_forest_destroy(forest);
            return 1;
        }

        if (!forest) {
            geif_status_t status = geif_forest_create(&forest, dims, &config);
            if (status != GEIF_OK) {
                fprintf(stderr, "Error initializing forest: %s\n", geif_status_str(status));
                fclose(fp);
                return 1;
            }
        }

        double *vec = (double *)malloc(dims * sizeof(double));
        char line[4096];
        uint64_t rows = 0;

        while (fgets(line, sizeof(line), fp)) {
            if (parse_line(line, vec, dims)) {
                geif_forest_feed(forest, vec);
                rows++;
            }
        }
        free(vec);
        fclose(fp);

        if (verbose) printf("Ingested %llu rows from '%s'. Training forest...\n",
                            (unsigned long long)rows, learn_file);

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
            } else if (verbose) {
                printf("Saved model to '%s'\n", save_file);
            }
        }
    }

    // Mode: Analyze / Score (-a)
    if (analyze_file) {
        if (!forest) {
            fprintf(stderr, "Error: Analysis requires a model. Train with -l or load with -r.\n");
            return 1;
        }

        FILE *in_fp = (strcmp(analyze_file, "-") == 0) ? stdin : fopen(analyze_file, "r");
        if (!in_fp) {
            fprintf(stderr, "Error opening analysis input '%s'\n", analyze_file);
            geif_forest_destroy(forest);
            return 1;
        }

        FILE *out_fp = output_file ? fopen(output_file, "w") : stdout;
        if (!out_fp) {
            fprintf(stderr, "Error opening output file '%s'\n", output_file);
            if (in_fp != stdin) fclose(in_fp);
            geif_forest_destroy(forest);
            return 1;
        }

        uint32_t dims = forest->dimensions;
        double *vec = (double *)malloc(dims * sizeof(double));
        char line[4096];
        uint64_t analyzed = 0;
        uint64_t outliers = 0;

        while (fgets(line, sizeof(line), in_fp)) {
            // Keep original line without newline for reporting
            char orig_line[4096];
            strncpy(orig_line, line, sizeof(orig_line) - 1);
            orig_line[sizeof(orig_line) - 1] = '\0';
            orig_line[strcspn(orig_line, "\r\n")] = 0;

            if (parse_line(line, vec, dims)) {
                double score = 0.0, H_metric = 0.0, d_out = 0.0;
                geif_forest_score_detailed(forest, vec, &score, &H_metric, &d_out);

                analyzed++;
                bool is_outlier = (score >= threshold);
                if (is_outlier) outliers++;

                fprintf(out_fp, "%s,%.6f,%d,%.6f,%.6f\n",
                        orig_line, score, is_outlier ? 1 : 0, H_metric, d_out);
            }
        }

        free(vec);
        if (in_fp != stdin) fclose(in_fp);
        if (out_fp != stdout) fclose(out_fp);

        if (verbose) {
            fprintf(stderr, "Scored %llu rows. Outliers (>= %.2f): %llu (%.2f%%)\n",
                    (unsigned long long)analyzed, threshold,
                    (unsigned long long)outliers,
                    analyzed > 0 ? (100.0 * (double)outliers / (double)analyzed) : 0.0);
        }
    }

    if (forest) geif_forest_destroy(forest);
    return 0;
}
