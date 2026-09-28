/**
 * @file template.c
 * @brief Template string formatter implementation for GEIF CLI.
 */

#include "template.h"
#include <stdio.h>
#include <string.h>

size_t geif_format_template(char *out,
                            size_t out_size,
                            const char *tmpl,
                            const geif_template_context_t *ctx)
{
    if (!out || out_size == 0) return 0;
    if (!tmpl || tmpl[0] == '\0') {
        out[0] = '\0';
        return 0;
    }

    size_t w = 0;
    const char *p = tmpl;

    while (*p != '\0' && w + 1 < out_size) {
        if (*p != '%') {
            out[w++] = *p++;
            continue;
        }

        // Hit '%'
        p++;
        if (*p == '\0') {
            out[w++] = '%';
            break;
        }

        char spec = *p++;
        int n = 0;

        switch (spec) {
        case '%':
            out[w++] = '%';
            break;

        case 's': // Score
            n = snprintf(out + w, out_size - w, "%.6f", ctx ? ctx->score : 0.0);
            if (n > 0) w += (size_t)n;
            break;

        case 'S': // Score percentage
            n = snprintf(out + w, out_size - w, "%.2f%%", ctx ? (ctx->score * 100.0) : 0.0);
            if (n > 0) w += (size_t)n;
            break;

        case 'l': // Label
            if (ctx && ctx->label && ctx->label[0] != '\0') {
                n = snprintf(out + w, out_size - w, "%s", ctx->label);
                if (n > 0) w += (size_t)n;
            }
            break;

        case 'c': // Category
        case 'C':
            if (ctx && ctx->category && ctx->category[0] != '\0') {
                n = snprintf(out + w, out_size - w, "%s", ctx->category);
                if (n > 0) w += (size_t)n;
            }
            break;

        case 'm': // Metric depth
            n = snprintf(out + w, out_size - w, "%.6f", ctx ? ctx->metric_depth : 0.0);
            if (n > 0) w += (size_t)n;
            break;

        case 'd': // Outer distance
            n = snprintf(out + w, out_size - w, "%.6f", ctx ? ctx->d_out : 0.0);
            if (n > 0) w += (size_t)n;
            break;

        case 'h': // H_max
            n = snprintf(out + w, out_size - w, "%.6f", ctx ? ctx->H_max : 0.0);
            if (n > 0) w += (size_t)n;
            break;

        case 'o': // Outlier flag
            n = snprintf(out + w, out_size - w, "%d", ctx ? ctx->is_outlier : 0);
            if (n > 0) w += (size_t)n;
            break;

        case 't': // Timestamp
            n = snprintf(out + w, out_size - w, "%ld", (long)(ctx ? ctx->timestamp : time(NULL)));
            if (n > 0) w += (size_t)n;
            break;

        case 'a': // Entire raw line
            if (ctx && ctx->orig_line && ctx->orig_line[0] != '\0') {
                n = snprintf(out + w, out_size - w, "%s", ctx->orig_line);
                if (n > 0) w += (size_t)n;
            }
            break;

        case 'v': // Feature vector
            if (ctx && ctx->vector && ctx->vector_dim > 0) {
                for (uint32_t j = 0; j < ctx->vector_dim && w + 24 < out_size; j++) {
                    n = snprintf(out + w, out_size - w, "%s%.6f", (j > 0 ? "," : ""), ctx->vector[j]);
                    if (n > 0) w += (size_t)n;
                }
            }
            break;

        default:
            // Unknown specifier: pass through % and specifier literally
            if (w + 2 < out_size) {
                out[w++] = '%';
                out[w++] = spec;
            }
            break;
        }

        if (w >= out_size) {
            w = out_size - 1;
        }
    }

    out[w] = '\0';
    return w;
}
