/**
 * @file geometry.h
 * @brief High-performance SIMD geometric routines for Voronoi bisectors and outer space stadium.
 */

#ifndef GEIF_GEOMETRY_H
#define GEIF_GEOMETRY_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GEIF_DIST_AVG(d) ((d) / 1.5 + 1.0 / (2.4 * (d)) - 1.0 / 12.0)
#define GEIF_FAST_C_SAMPLES 2048

double geif_c(double n);
void geif_init_c_cache(void);

/**
 * @brief Linearly scales a value from [min, max] into a target range starting at scale_min.
 *
 * @param[in] value     Input value.
 * @param[in] range     Span of target range.
 * @param[in] scale_min Minimum of target range.
 * @param[in] min       Source domain lower bound.
 * @param[in] max       Source domain upper bound.
 * @return Linearly scaled value.
 */
static inline double geif_scale_value(double value, double range, double scale_min, double min, double max)
{
    if (max == min) return scale_min;
    return range * (value - min) / (max - min) + scale_min;
}

/**
 * @brief Computes squared Euclidean distance between two d-dimensional points.
 *
 * Provides specialized unrolled branches for 2D and 3D with compiler vectorization hints.
 *
 * @param[in] a First point coordinate array.
 * @param[in] b Second point coordinate array.
 * @param[in] d Number of dimensions.
 * @return Squared Euclidean distance.
 */
static inline double geif_dist_sq(const double * restrict a,
                                  const double * restrict b,
                                  uint32_t d)
{
    if (d == 2) {
        double d0 = a[0] - b[0];
        double d1 = a[1] - b[1];
        return d0 * d0 + d1 * d1;
    }
    if (d == 3) {
        double d0 = a[0] - b[0];
        double d1 = a[1] - b[1];
        double d2 = a[2] - b[2];
        return d0 * d0 + d1 * d1 + d2 * d2;
    }
    double dist_sq = 0.0;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (uint32_t i = 0; i < d; i++) {
        double diff = a[i] - b[i];
        dist_sq += diff * diff;
    }
    return dist_sq;
}

/**
 * @brief Generates standard normal Gaussian random numbers using Marsaglia-Bray Box-Muller transform.
 *
 * @return Pseudo-random sample from standard normal distribution N(0, 1).
 */
static inline double geif_gaussrand(void)
{
    static double U, V;
    static int phase = 0;
    double Z;

    if (phase == 0) {
        U = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        V = (double)rand() / ((double)RAND_MAX + 1.0);
        Z = sqrt(-2.0 * log(U)) * sin(2.0 * M_PI * V);
    } else {
        Z = sqrt(-2.0 * log(U)) * cos(2.0 * M_PI * V);
    }
    phase = 1 - phase;
    return Z;
}

/**
 * @brief Evaluates vector dot product sum(a[j] * b[j]) for j = 0..d-1.
 * Uses restrict pointer annotations for compiler autovectorization.
 */
static inline double geif_dot(const double * restrict a,
                              const double * restrict b,
                              uint32_t d)
{
    if (d == 2) return a[0] * b[0] + a[1] * b[1];
    if (d == 3) return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
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
