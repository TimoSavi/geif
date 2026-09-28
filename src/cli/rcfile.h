/**
 * @file rcfile.h
 * @brief Run-command (RC) configuration file parser for GEIF.
 */

#ifndef GEIF_RCFILE_H
#define GEIF_RCFILE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t tree_count;
    bool     tree_count_set;

    uint32_t samples_per_tree;
    bool     samples_per_tree_set;

    int      decimals;
    bool     decimals_set;

    char     outlier_score_spec[64];
    bool     outlier_score_set;

    char     print_dimension[1024];
    bool     print_dimension_set;

    uint32_t low_rgb;
    bool     low_rgb_set;

    uint32_t high_rgb;
    bool     high_rgb_set;

    char     category_sep;
    bool     category_sep_set;

    char     label_sep;
    bool     label_sep_set;
} geif_rc_config_t;

/**
 * @brief Initialize an rc config struct with default zeroed values.
 */
void geif_rc_config_init(geif_rc_config_t *rc);

/**
 * @brief Parse a single RC file into the config struct.
 * Expands '~' if leading path. Returns true on success, false if file cannot be opened.
 */
bool geif_rc_parse_file(geif_rc_config_t *rc, const char *filepath);

/**
 * @brief Load default global RC file (~/.geifrc, or fallback to ~/.ceifrc if present).
 * If neither exists, returns true (not an error).
 */
bool geif_rc_load_default(geif_rc_config_t *rc);

#ifdef __cplusplus
}
#endif

#endif // GEIF_RCFILE_H
