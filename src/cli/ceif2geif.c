/**
 * @file ceif2geif.c
 * @brief CEIF JSON model to GEIF-1.0 JSON model migration tool.
 *
 * Author / Maintainer: Timo Savinen (AI-assisted)
 */

#include "geif/geif.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <getopt.h>

#define CEIF2GEIF_VERSION "1.1.0"

/**
 * @brief Validates and parses an algorithm name string for -B / --algo.
 */
static bool parse_algo_option(const char *name, geif_algo_type_t *algo_out)
{
    if (!name || name[0] == '\0') return false;
    if (strcasecmp(name, "bubble") == 0 ||
        strcasecmp(name, "spherical") == 0 ||
        strcasecmp(name, "voronoi") == 0 ||
        strcasecmp(name, "exemplar") == 0 ||
        strcasecmp(name, "knn") == 0 ||
        strcasecmp(name, "density") == 0 ||
        strcasecmp(name, "ceif") == 0 ||
        strcasecmp(name, "eif") == 0 ||
        strcasecmp(name, "gaussian") == 0) {
        *algo_out = geif_algo_from_name(name);
        return true;
    }
    return false;
}

/**
 * @brief Prints command-line help and usage instructions for ceif2geif.
 *
 * @param[in] prog Executable program name (argv[0]).
 */
static void print_usage(const char *prog)
{
    printf("ceif2geif v%s - Migrate CEIF JSON models to GEIF-1.0 JSON models\n", CEIF2GEIF_VERSION);
    printf("Author / Maintainer: Timo Savinen (AI-assisted)\n\n");
    printf("Usage:\n");
    printf("  %s [options] <input_ceif.json> [output_geif.json]\n", prog);
    printf("  %s - < input_ceif.json > output_geif.json\n\n", prog);
    printf("Options:\n");
    printf("  -o, --output <file>    Output GEIF model JSON file (default: stdout or 2nd argument)\n");
    printf("  -B, --algo <name>      Change algorithm during conversion (bubble|voronoi|exemplar|ceif)\n");
    printf("  -t, --trees <int>      Override tree count in forest (default: from CEIF or 100)\n");
    printf("  -s, --samples <int>    Override samples per tree (default: from CEIF or 256)\n");
    printf("  -v, --verbose          Print migration details and sub-forest statistics\n");
    printf("  -q, --quiet            Suppress all non-error output\n");
    printf("  -h, --help             Display this help message and exit\n");
    printf("  -V, --version          Display version and exit\n");
}

/**
 * @brief Main entry point for ceif2geif model migration utility.
 *
 * Ingests legacy CEIF JSON models, reconstructs sample reservoirs and trees,
 * optionally applies retrain overrides, and outputs clean GEIF-1.0 JSON models.
 *
 * @param[in] argc Argument count.
 * @param[in] argv Argument vector.
 * @return 0 on success, non-zero exit code on error.
 */
int main(int argc, char *argv[])
{
    const char *input_file = NULL;
    const char *output_file = NULL;
    bool verbose = false;
    bool quiet = false;
    int tree_override = 0;
    int samples_override = 0;
    bool algo_override_set = false;
    geif_algo_type_t algo_override = GEIF_ALGO_DEFAULT;

    static struct option long_options[] = {
        {"output",  required_argument, 0, 'o'},
        {"algo",    required_argument, 0, 'B'},
        {"trees",   required_argument, 0, 't'},
        {"samples", required_argument, 0, 's'},
        {"verbose", no_argument,       0, 'v'},
        {"quiet",   no_argument,       0, 'q'},
        {"help",    no_argument,       0, 'h'},
        {"version", no_argument,       0, 'V'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "o:B:t:s:vqhV", long_options, NULL)) != -1) {
        switch (opt) {
        case 'o': output_file = optarg; break;
        case 'B':
            if (!parse_algo_option(optarg, &algo_override)) {
                fprintf(stderr, "ceif2geif: error: unknown algorithm '%s' (expected bubble, voronoi, exemplar, or ceif)\n", optarg);
                return 1;
            }
            algo_override_set = true;
            break;
        case 't': tree_override = atoi(optarg); break;
        case 's': samples_override = atoi(optarg); break;
        case 'v': verbose = true; break;
        case 'q': quiet = true; break;
        case 'h': print_usage(argv[0]); return 0;
        case 'V': printf("ceif2geif version %s\n", CEIF2GEIF_VERSION); return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    // Process positional arguments
    if (optind < argc) {
        input_file = argv[optind++];
    }
    if (optind < argc && !output_file) {
        output_file = argv[optind++];
    }

    if (!input_file) {
        if (isatty(STDIN_FILENO)) {
            print_usage(argv[0]);
            return 1;
        }
        input_file = "-";
    }

    if (!output_file) {
        output_file = "-";
    }

    if (verbose && !quiet) {
        fprintf(stderr, "ceif2geif: Loading model from '%s'...\n", input_file);
    }

    geif_ensemble_t *ens = NULL;
    geif_status_t status = geif_ensemble_load_json(&ens, input_file);
    if (status != GEIF_OK || !ens) {
        fprintf(stderr, "ceif2geif: error: failed to load CEIF model from '%s': %s\n",
                input_file, geif_status_str(status));
        return 1;
    }

    // Apply algorithm, tree count, and sample count overrides
    if (algo_override_set || tree_override > 0 || samples_override > 0) {
        if (algo_override_set) {
            ens->config.algo = algo_override;
        }
        if (tree_override > 0) {
            ens->config.tree_count = (uint32_t)tree_override;
        }
        if (samples_override > 0) {
            ens->config.samples_per_tree = (uint32_t)samples_override;
        }
        if (verbose && !quiet) {
            fprintf(stderr, "ceif2geif: Applying overrides (algo=%s, trees=%u, samples=%u)...\n",
                    geif_algo_name(ens->config.algo),
                    (unsigned)ens->config.tree_count,
                    (unsigned)ens->config.samples_per_tree);
        }
        for (size_t i = 0; i < ens->count; i++) {
            geif_forest_t *f = ens->entries[i].forest;
            if (f) {
                geif_algo_type_t target_algo = algo_override_set ? algo_override : f->config.algo;
                uint32_t new_samples = (samples_override > 0) ? (uint32_t)samples_override : f->config.samples_per_tree;
                uint32_t new_trees;
                if (target_algo == GEIF_ALGO_EXEMPLAR) {
                    new_trees = 1;
                } else if (tree_override > 0) {
                    new_trees = (uint32_t)tree_override;
                } else if (f->config.algo == GEIF_ALGO_EXEMPLAR && f->tree_count <= 1) {
                    new_trees = GEIF_DEFAULT_TREE_COUNT;
                } else {
                    new_trees = f->tree_count;
                }

                // Free old trees
                for (uint32_t t = 0; t < f->tree_count; t++) {
                    if (f->trees[t].nodes) free(f->trees[t].nodes);
                    if (f->trees[t].normals_pool) free(f->trees[t].normals_pool);
                    if (f->trees[t].leaf_samples) free(f->trees[t].leaf_samples);
                }
                free(f->trees);

                f->config.algo = target_algo;
                f->tree_count = new_trees;
                f->config.tree_count = new_trees;
                f->config.samples_per_tree = new_samples;
                f->is_trained = false;
                f->trees = (geif_tree_t *)calloc(new_trees, sizeof(geif_tree_t));

                size_t needed_cap = (target_algo == GEIF_ALGO_EXEMPLAR) ? (size_t)new_samples : (size_t)new_trees * new_samples;
                if (needed_cap > f->pool_capacity) {
                    double *new_pool = (double *)realloc(f->sample_pool, needed_cap * f->dimensions * sizeof(double));
                    if (new_pool) {
                        f->sample_pool = new_pool;
                        f->pool_capacity = needed_cap;
                    }
                }
                if (f->pool_count > needed_cap) {
                    f->pool_count = needed_cap;
                }

                geif_forest_train(f);
            }
        }
    }

    if (verbose && !quiet) {
        size_t summary_cap = 4096 + ens->count * 384;
        char *summary = (char *)malloc(summary_cap);
        if (summary) {
            geif_ensemble_summary(ens, summary, summary_cap);
            fprintf(stderr, "%s", summary);
            free(summary);
        }
    }

    status = geif_ensemble_save_json(ens, output_file);
    if (status != GEIF_OK) {
        fprintf(stderr, "ceif2geif: error: failed to save GEIF model to '%s': %s\n",
                output_file, geif_status_str(status));
        geif_ensemble_destroy(ens);
        return 1;
    }

    if (verbose && !quiet) {
        fprintf(stderr, "ceif2geif: Successfully migrated to GEIF-1.0 model '%s'\n", output_file);
    }

    geif_ensemble_destroy(ens);
    return 0;
}
