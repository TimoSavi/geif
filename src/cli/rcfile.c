/**
 * @file rcfile.c
 * @brief Run-command (RC) configuration file parser for GEIF.
 */

#include "rcfile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/**
 * @brief Initializes run-command configuration to factory defaults.
 *
 * @param[out] rc RC configuration structure to initialize.
 */
void geif_rc_config_init(geif_rc_config_t *rc)
{
    if (!rc) return;
    memset(rc, 0, sizeof(geif_rc_config_t));
    rc->category_sep = ';';
}

/**
 * @brief Expands leading tilde ('~') in a file path to the user's HOME directory.
 *
 * @param[in]  in       Input path string.
 * @param[out] out      Output buffer for expanded absolute path.
 * @param[in]  out_size Capacity of output buffer.
 */
static void expand_path(const char *in, char *out, size_t out_size)
{
    if (in[0] == '~' && (in[1] == '/' || in[1] == '\0')) {
        const char *home = getenv("HOME");
        if (home) {
            snprintf(out, out_size, "%s%s", home, in + 1);
            return;
        }
    }
    strncpy(out, in, out_size - 1);
    out[out_size - 1] = '\0';
}

/**
 * @brief Extracts a configuration value matching a given key name from a line.
 *
 * Handles case-insensitive key comparison, optional whitespace, equal signs,
 * and double quotes around string values.
 *
 * @param[in,out] line Line buffer from configuration file.
 * @param[in]     key  Directive key name (e.g. "TREES").
 * @return Pointer to trimmed value string inside line buffer, or NULL if no match.
 */
static char *extract_value(char *line, const char *key)
{
    char *p = line;
    while (isspace((unsigned char)*p)) p++;

    size_t klen = strlen(key);
    if (strncasecmp(p, key, klen) != 0) return NULL;

    p += klen;
    if (*p != '\0' && !isspace((unsigned char)*p) && *p != '=') return NULL;

    while (isspace((unsigned char)*p) || *p == '=') p++;

    if (*p == '"') {
        p++;
        char *end = strchr(p, '"');
        if (end) *end = '\0';
        return p;
    }

    char *start = p;
    char *end = p + strlen(p);
    while (end > start && isspace((unsigned char)*(end - 1))) {
        end--;
        *end = '\0';
    }
    return start;
}

/**
 * @brief Parses an RC configuration file and populates an rc_config structure.
 *
 * Reads directives such as TREES, SAMPLES, DECIMALS, OUTLIER_SCORE, PRINT_DIMENSION,
 * and hex color codes. Ignores comments ('#') and blank lines.
 *
 * @param[in,out] rc       RC configuration instance.
 * @param[in]     filepath Path to configuration file.
 * @return True if file was read and parsed successfully, false on I/O error.
 */
bool geif_rc_parse_file(geif_rc_config_t *rc, const char *filepath)
{
    if (!rc || !filepath) return false;

    char resolved_path[1024];
    expand_path(filepath, resolved_path, sizeof(resolved_path));

    FILE *fp = fopen(resolved_path, "r");
    if (!fp) return false;

    char line[2048];
    while (fgets(line, sizeof(line), fp)) {
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#' || *p == '\r' || *p == '\n') continue;

        char *val;
        if ((val = extract_value(line, "TREES")) != NULL) {
            rc->tree_count = (uint32_t)atoi(val);
            rc->tree_count_set = true;
        } else if ((val = extract_value(line, "SAMPLES")) != NULL ||
                   (val = extract_value(line, "MAX_SAMPLES")) != NULL) {
            rc->samples_per_tree = (uint32_t)atoi(val);
            rc->samples_per_tree_set = true;
        } else if ((val = extract_value(line, "DECIMALS")) != NULL) {
            rc->decimals = atoi(val);
            rc->decimals_set = true;
        } else if ((val = extract_value(line, "OUTLIER_SCORE")) != NULL) {
            strncpy(rc->outlier_score_spec, val, sizeof(rc->outlier_score_spec) - 1);
            rc->outlier_score_set = true;
        } else if ((val = extract_value(line, "PRINT_DIMENSION")) != NULL) {
            strncpy(rc->print_dimension, val, sizeof(rc->print_dimension) - 1);
            rc->print_dimension_set = true;
        } else if ((val = extract_value(line, "LOW_RGB_COLOR")) != NULL) {
            rc->low_rgb = (uint32_t)strtoul(val, NULL, 16);
            rc->low_rgb_set = true;
        } else if ((val = extract_value(line, "HIGH_RGB_COLOR")) != NULL) {
            rc->high_rgb = (uint32_t)strtoul(val, NULL, 16);
            rc->high_rgb_set = true;
        } else if ((val = extract_value(line, "CATEGORY_SEPARATOR")) != NULL) {
            rc->category_sep = (val[0] == '"' && val[1]) ? val[1] : val[0];
            rc->category_sep_set = true;
        } else if ((val = extract_value(line, "LABEL_SEPARATOR")) != NULL) {
            rc->label_sep = (val[0] == '"' && val[1]) ? val[1] : val[0];
            rc->label_sep_set = true;
        } else if (extract_value(line, "AUTO_SCALE") != NULL ||
                   extract_value(line, "AUTO_WEIGTH") != NULL ||
                   extract_value(line, "NEAREST") != NULL ||
                   extract_value(line, "ANALYZE_SAMPLING") != NULL ||
                   extract_value(line, "DEBUG") != NULL ||
                   extract_value(line, "CLUSTER_SIZE") != NULL ||
                   extract_value(line, "DIM_PRINT_WIDTH") != NULL ||
                   extract_value(line, "IGNORE_EXPR_PARSE_ERROR") != NULL ||
                   extract_value(line, "CENTROID_THRESHOLD") != NULL ||
                   extract_value(line, "CENTROID_TRESSHOLD") != NULL) {
            // Recognized CEIF compatibility directives - safely ignored in GEIF
        }
    }

    fclose(fp);
    return true;
}

/**
 * @brief Discovers and loads the default user RC configuration file.
 *
 * Checks in priority order:
 *  1. ~/.geifrc
 *  2. ~/.ceifrc (legacy fallback)
 *
 * @param[in,out] rc RC configuration instance.
 * @return True if a file was loaded or if no configuration file was present.
 */
bool geif_rc_load_default(geif_rc_config_t *rc)
{
    char path[1024];

    // Priority 1: ~/.geifrc
    expand_path("~/.geifrc", path, sizeof(path));
    FILE *fp = fopen(path, "r");
    if (fp) {
        fclose(fp);
        return geif_rc_parse_file(rc, path);
    }

    // Priority 2: ~/.ceifrc
    expand_path("~/.ceifrc", path, sizeof(path));
    fp = fopen(path, "r");
    if (fp) {
        fclose(fp);
        return geif_rc_parse_file(rc, path);
    }

    return true; // No global config is a normal condition
}
