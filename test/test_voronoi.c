/**
 * @file test_voronoi.c
 * @brief Unit test for Voronoi bisector hyperplane calculation with extreme aspect ratios.
 */

#include "geif/geif.h"
#include "geometry.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

/**
 * @brief Unit test verifying Voronoi bisector hyperplane calculation with extreme aspect ratios.
 *
 * @return 0 on test success, or triggers assert failure.
 */
int main(void)
{
    printf("Testing Voronoi Bisector Normal Vector & Scale Invariance...\n");

    // Test Case 1: 5000:1 aspect ratio (complex2d simulation)
    // Dimension 0: Span 8000.0 (X from 1000 to 9000)
    // Dimension 1: Span 1.6 (Y from 0.2 to 1.8)
    uint32_t d = 2;
    double effective_span[2] = { 8000.0, 1.6 };
    uint8_t dim_active[2] = { 1, 1 };

    double A[2] = { 1000.0, 0.4 };
    double B[2] = { 9000.0, 1.6 };

    double normal[2] = { 0.0, 0.0 };
    double pdotn = 0.0;
    double delta = 0.0;

    bool ok = geif_compute_bisector(A, B, effective_span, dim_active, d, normal, &pdotn, &delta);
    assert(ok);
    assert(!isnan(normal[0]));
    assert(!isnan(normal[1]));
    assert(!isnan(pdotn));
    assert(!isnan(delta));

    // Midpoint (5000, 1.0) must sit exactly on the hyperplane: dot(mid, n) == pdotn
    double mid[2] = { 5000.0, 1.0 };
    double dot_mid = geif_dot(mid, normal, d);
    assert(fabs(dot_mid - pdotn) < 1e-9);

    // Sample A must fall on the left side: dot(A, n) < pdotn
    double dot_A = geif_dot(A, normal, d);
    assert(dot_A < pdotn);

    // Sample B must fall on the right side: dot(B, n) >= pdotn
    double dot_B = geif_dot(B, normal, d);
    assert(dot_B > pdotn);

    printf("  [PASS] 5000:1 aspect ratio hyperplane verified.\n");

    // Test Case 2: Exact duplicate points (A == B)
    double A_dup[2] = { 42.0, 3.14 };
    double B_dup[2] = { 42.0, 3.14 };
    ok = geif_compute_bisector(A_dup, B_dup, effective_span, dim_active, d, normal, &pdotn, &delta);
    assert(!ok); // Must reject duplicates cleanly without NaN
    printf("  [PASS] Duplicate sample pair rejection verified.\n");

    // Test Case 3: Zero-variance dimension
    uint8_t dim_active_zero[2] = { 1, 0 }; // Dimension 1 is constant
    double A_const[2] = { 10.0, 5.0 };
    double B_const[2] = { 20.0, 5.0 };
    ok = geif_compute_bisector(A_const, B_const, effective_span, dim_active_zero, d, normal, &pdotn, &delta);
    assert(ok);
    assert(normal[1] == 0.0); // Constant dimension must have 0 normal
    printf("  [PASS] Zero-variance dimension health masking verified.\n");

    printf("ALL VORONOI TESTS PASSED!\n");
    return 0;
}
