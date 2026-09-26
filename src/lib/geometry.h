/**
 * @file geometry.h
 * @brief High-performance SIMD geometric routines for Voronoi bisectors and outer space stadium.
 */

#ifndef GEIF_GEOMETRY_H
#define GEIF_GEOMETRY_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Evaluates vector dot product sum(a[j] * b[j]) for j = 0..d-1.
 * Uses restrict pointer annotations for compiler autovectorization.
 */
static inline double geif_dot(const double * restrict a,
                              const double * restrict b,
                              uint32_t d)
{
    double sum = 0.0;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (uint32_t i = 0; i < d; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * @brief Computes normalized Euclidean outer space distance outside the bounding stadium.
 * Points inside [min, max] have distance 0.0.
 */
static inline double geif_stadium_distance(const double * restrict x,
                                          const double * restrict env_min,
                                          const double * restrict env_max,
                                          const double * restrict effective_span,
                                          uint32_t d)
{
    double dist_sq = 0.0;
    for (uint32_t j = 0; j < d; j++) {
        double d_low = env_min[j] - x[j];
        double d_high = x[j] - env_max[j];
        double delta = 0.0;
        if (d_low > 0.0) {
            delta = d_low;
        } else if (d_high > 0.0) {
            delta = d_high;
        }

        if (delta > 0.0) {
            double normalized = delta / effective_span[j];
            dist_sq += normalized * normalized;
        }
    }
    return sqrt(dist_sq);
}

/**
 * @brief Calculates pre-baked scale-invariant Voronoi bisector hyperplane parameters.
 *
 * Normal vector: n_j = (B_j - A_j) / (span_j^2)
 * Scalar threshold: p_scalar = sum_j n_j * (A_j + B_j) / 2
 * Generator distance: delta = sqrt(sum_j ((B_j - A_j) / span_j)^2)
 *
 * Returns false if points A and B are identical (delta < 1e-12).
 */
static inline bool geif_compute_bisector(const double * restrict A,
                                        const double * restrict B,
                                        const double * restrict effective_span,
                                        const uint8_t * restrict dim_active,
                                        uint32_t d,
                                        double * restrict normal_out,
                                        double * restrict pdotn_out,
                                        double * restrict delta_out)
{
    double delta_sq = 0.0;
    double pdotn = 0.0;

    for (uint32_t j = 0; j < d; j++) {
        if (!dim_active[j]) {
            normal_out[j] = 0.0;
            continue;
        }

        double diff = B[j] - A[j];
        double span = effective_span[j];
        double norm_diff = diff / span;
        delta_sq += norm_diff * norm_diff;

        double span_sq = span * span;
        double nj = diff / span_sq;
        normal_out[j] = nj;

        double mid = 0.5 * (A[j] + B[j]);
        pdotn += nj * mid;
    }

    double delta = sqrt(delta_sq);
    *delta_out = delta;
    *pdotn_out = pdotn;

    return (delta >= 1e-12);
}

/**
 * @brief Computes normalized residual Euclidean distance from query point x to leaf point P.
 */
static inline double geif_residual_distance(const double * restrict x,
                                            const double * restrict P_leaf,
                                            const double * restrict effective_span,
                                            const uint8_t * restrict dim_active,
                                            uint32_t d)
{
    double dist_sq = 0.0;
    for (uint32_t j = 0; j < d; j++) {
        if (!dim_active[j]) {
            continue;
        }
        double diff = (x[j] - P_leaf[j]) / effective_span[j];
        dist_sq += diff * diff;
    }
    return sqrt(dist_sq);
}

#ifdef __cplusplus
}
#endif

#endif /* GEIF_GEOMETRY_H */
