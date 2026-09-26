/**
 * @file test_stadium.c
 * @brief Unit test for Euclidean outer space stadium distance calculation.
 */

#include "geif/geif.h"
#include "geometry.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main(void)
{
    printf("Testing Outer Space Stadium Euclidean Distance...\n");

    uint32_t d = 2;
    double env_min[2] = { 0.0, 0.0 };
    double env_max[2] = { 10.0, 10.0 };
    double effective_span[2] = { 10.0, 10.0 };

    // Point 1: Inside the bounding envelope
    double inside[2] = { 5.0, 5.0 };
    double d1 = geif_stadium_distance(inside, env_min, env_max, effective_span, d);
    assert(d1 == 0.0);
    printf("  [PASS] Interior point distance is exactly 0.0.\n");

    // Point 2: Straight North outside the box (X = 5.0, Y = 15.0)
    // Delta Y = 5.0, normalized = 5.0 / 10.0 = 0.5
    double north[2] = { 5.0, 15.0 };
    double d2 = geif_stadium_distance(north, env_min, env_max, effective_span, d);
    assert(fabs(d2 - 0.5) < 1e-9);
    printf("  [PASS] Perpendicular outer excursion verified (d_out = 0.5).\n");

    // Point 3: North-East corner outside the box (X = 13.0, Y = 14.0)
    // Delta X = 3.0 (norm: 0.3), Delta Y = 4.0 (norm: 0.4)
    // Euclidean distance = sqrt(0.3^2 + 0.4^2) = 0.5 (rounded corner stadium!)
    double corner[2] = { 13.0, 14.0 };
    double d3 = geif_stadium_distance(corner, env_min, env_max, effective_span, d);
    assert(fabs(d3 - 0.5) < 1e-9);
    printf("  [PASS] Rounded corner stadium Euclidean arc verified (3-4-5 triangle: d_out = 0.5).\n");

    printf("ALL STADIUM TESTS PASSED!\n");
    return 0;
}
