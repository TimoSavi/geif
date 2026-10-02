/**
 * @file template.c
 * @brief Full-featured output templating and attribution engine for GEIF CLI.
 */

#include "template.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static inline void srgb_companding(double *color)
{
    for (int i = 0; i < 3; i++) {
        double v = color[i] / 255.0;
        if (v <= 0.0031308) v = 12.92 * v;
        else v = 1.055 * pow(v, 1.0 / 2.4) - 0.055;
        color[i] = v * 255.0;
    }
}

static uint32_t score_to_rgb(double score, uint32_t low_rgb, uint32_t high_rgb)
{
    if (score == 0.0) return 0x000000;
    if (score < 0.0) score = 0.0;
    if (score > 1.0) score = 1.0;

    double r0 = (double)((low_rgb >> 16) & 0xFF);
    double g0 = (double)((low_rgb >> 8) & 0xFF);
    double b0 = (double)(low_rgb & 0xFF);

    double r1 = (double)((high_rgb >> 16) & 0xFF);
    double g1 = (double)((high_rgb >> 8) & 0xFF);
    double b1 = (double)(high_rgb & 0xFF);

    double color[3];
    color[0] = r0 + (r1 - r0) * score;
    color[1] = g0 + (g1 - g0) * score;
    color[2] = b0 + (b1 - b0) * score;

    srgb_companding(color);

    uint32_t ir = (uint32_t)(color[0] < 0.0 ? 0 : (color[0] > 255.0 ? 255 : color[0]));
    uint32_t ig = (uint32_t)(color[1] < 0.0 ? 0 : (color[1] > 255.0 ? 255 : color[1]));
    uint32_t ib = (uint32_t)(color[2] < 0.0 ? 0 : (color[2] > 255.0 ? 255 : color[2]));

    return (ir << 16) | (ig << 8) | ib;
}

static int format_double(char *buf, size_t buf_sz, double val, int decimals, const char *fmt)
{
    if (!buf || buf_sz == 0) return 0;
    if (fmt && fmt[0] != '\0') {
        return snprintf(buf, buf_sz, fmt, val);
    }
    if (decimals >= 0) {
        return snprintf(buf, buf_sz, "%.*f", decimals, val);
    }
    return snprintf(buf, buf_sz, "%g", val);
}

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
    char sep = (ctx && ctx->list_separator) ? ctx->list_separator : ';';
    int dec = (ctx && ctx->decimals >= 0) ? ctx->decimals : 6;
    uint32_t low_c = (ctx && ctx->low_rgb != 0) ? ctx->low_rgb : 0xFFFF00;
    uint32_t high_c = (ctx && ctx->high_rgb != 0) ? ctx->high_rgb : 0xFF0000;

    while (*p != '\0' && w + 1 < out_size) {
        if (*p != '%') {
            out[w++] = *p++;
            continue;
        }

        p++; // skip '%'
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
            n = format_double(out + w, out_size - w, ctx ? ctx->score : 0.0, dec, NULL);
            if (n > 0) w += (size_t)n;
            break;

        case 'S': // Score percentage
            n = snprintf(out + w, out_size - w, "%.*f%%", dec > 0 ? (dec <= 2 ? dec : 2) : 2, ctx ? (ctx->score * 100.0) : 0.0);
            if (n > 0) w += (size_t)n;
            break;

        case 'r': // Row / line index
            n = snprintf(out + w, out_size - w, "%llu", (unsigned long long)(ctx ? ctx->row_idx : 0));
            if (n > 0) w += (size_t)n;
            break;

        case 'l': // Label
        case 'L':
            if (ctx && ctx->label && ctx->label[0] != '\0') {
                n = snprintf(out + w, out_size - w, "%s", ctx->label);
                if (n > 0) w += (size_t)n;
            }
            break;

        case 'c': // Category from input data (fallback to assigned category)
            {
                const char *cat = (ctx && ctx->input_category && ctx->input_category[0] != '\0')
                                  ? ctx->input_category
                                  : (ctx ? ctx->category : NULL);
                if (cat && cat[0] != '\0') {
                    n = snprintf(out + w, out_size - w, "%s", cat);
                    if (n > 0) w += (size_t)n;
                }
            }
            break;

        case 'C': // Model / assigned category
            if (ctx && ctx->category && ctx->category[0] != '\0') {
                n = snprintf(out + w, out_size - w, "%s", ctx->category);
                if (n > 0) w += (size_t)n;
            }
            break;

        case 'm': // Metric depth or dimension expansion (-j)
            if (ctx && ctx->print_dimension && ctx->print_dimension[0] != '\0' &&
                ctx->vector && ctx->vector_dim > 0) {
                for (uint32_t i = 0; i < ctx->vector_dim && w + 1 < out_size; i++) {
                    if (i > 0 && sep != '\0') {
                        out[w++] = sep;
                        if (w + 1 >= out_size) break;
                    }

                    const char *d = ctx->print_dimension;
                    while (*d != '\0' && w + 1 < out_size) {
                        if (*d != '%') {
                            out[w++] = *d++;
                            continue;
                        }
                        d++; // skip '%'
                        if (*d == '\0') {
                            out[w++] = '%';
                            break;
                        }
                        char dspec = *d++;
                        int dn = 0;
                        switch (dspec) {
                        case 'd':
                            dn = format_double(out + w, out_size - w, ctx->vector[i], dec, ctx->printf_format);
                            if (dn > 0) w += (size_t)dn;
                            break;
                        case 'a':
                            dn = format_double(out + w, out_size - w, ctx->averages ? ctx->averages[i] : 0.0, dec, ctx->printf_format);
                            if (dn > 0) w += (size_t)dn;
                            break;
                        case 'e':
                            dn = format_double(out + w, out_size - w, ctx->attr_scores ? ctx->attr_scores[i] : 0.0, dec, NULL);
                            if (dn > 0) w += (size_t)dn;
                            break;
                        case 'i':
                            dn = snprintf(out + w, out_size - w, "%u", i + 1);
                            if (dn > 0) w += (size_t)dn;
                            break;
                        case '%':
                            out[w++] = '%';
                            break;
                        default:
                            if (w + 2 < out_size) {
                                out[w++] = '%';
                                out[w++] = dspec;
                            }
                            break;
                        }
                    }
                }
            } else {
                n = format_double(out + w, out_size - w, ctx ? ctx->metric_depth : 0.0, dec, NULL);
                if (n > 0) w += (size_t)n;
            }
            break;

        case 'd': // Dimension vector list (joined by sep) or outer distance
            if (ctx && ctx->vector && ctx->vector_dim > 0) {
                for (uint32_t i = 0; i < ctx->vector_dim && w + 1 < out_size; i++) {
                    if (i > 0 && sep != '\0') {
                        out[w++] = sep;
                        if (w + 1 >= out_size) break;
                    }
                    int dn = format_double(out + w, out_size - w, ctx->vector[i], dec, ctx->printf_format);
                    if (dn > 0) w += (size_t)dn;
                }
            } else {
                n = format_double(out + w, out_size - w, ctx ? ctx->d_out : 0.0, dec, NULL);
                if (n > 0) w += (size_t)n;
            }
            break;

        case 'e': // Single-dimension impact / attribution scores list (joined by sep)
            if (ctx && ctx->attr_scores && ctx->vector_dim > 0) {
                for (uint32_t i = 0; i < ctx->vector_dim && w + 1 < out_size; i++) {
                    if (i > 0 && sep != '\0') {
                        out[w++] = sep;
                        if (w + 1 >= out_size) break;
                    }
                    int dn = format_double(out + w, out_size - w, ctx->attr_scores[i], dec, NULL);
                    if (dn > 0) w += (size_t)dn;
                }
            }
            break;

        case 'a': // Category dimension averages list (joined by sep) or orig_line
            if (ctx && ctx->averages && ctx->vector_dim > 0) {
                for (uint32_t i = 0; i < ctx->vector_dim && w + 1 < out_size; i++) {
                    if (i > 0 && sep != '\0') {
                        out[w++] = sep;
                        if (w + 1 >= out_size) break;
                    }
                    int dn = format_double(out + w, out_size - w, ctx->averages[i], dec, ctx->printf_format);
                    if (dn > 0) w += (size_t)dn;
                }
            } else if (ctx && ctx->orig_line && ctx->orig_line[0] != '\0') {
                n = snprintf(out + w, out_size - w, "%s", ctx->orig_line);
                if (n > 0) w += (size_t)n;
            }
            break;

        case 'v': // Raw input tokens joined by sep, or feature vector
            if (ctx && ctx->raw_values && ctx->raw_value_count > 0) {
                for (uint32_t i = 0; i < ctx->raw_value_count && w + 1 < out_size; i++) {
                    if (i > 0 && sep != '\0') {
                        out[w++] = sep;
                        if (w + 1 >= out_size) break;
                    }
                    n = snprintf(out + w, out_size - w, "%s", ctx->raw_values[i]);
                    if (n > 0) w += (size_t)n;
                }
            } else if (ctx && ctx->orig_line && ctx->orig_line[0] != '\0') {
                n = snprintf(out + w, out_size - w, "%s", ctx->orig_line);
                if (n > 0) w += (size_t)n;
            }
            break;

        case 'x': // 6-hex RGB color
            n = snprintf(out + w, out_size - w, "%06X", score_to_rgb(ctx ? ctx->score : 0.0, low_c, high_c));
            if (n > 0) w += (size_t)n;
            break;

        case 'h': // H_max universal scale
            n = format_double(out + w, out_size - w, ctx ? ctx->H_max : 0.0, dec, NULL);
            if (n > 0) w += (size_t)n;
            break;

        case 'o': // Outlier flag
            n = snprintf(out + w, out_size - w, "%d", ctx ? ctx->is_outlier : 0);
            if (n > 0) w += (size_t)n;
            break;

        case 'n': // Total training rows
            n = snprintf(out + w, out_size - w, "%llu", (unsigned long long)(ctx ? ctx->total_rows : 0));
            if (n > 0) w += (size_t)n;
            break;

        case 't': // Timestamp
            n = snprintf(out + w, out_size - w, "%ld", (long)(ctx ? ctx->timestamp : time(NULL)));
            if (n > 0) w += (size_t)n;
            break;

        default:
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
