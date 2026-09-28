/**
 * @file test_negative.c
 * @brief Rigorous verification of dimension scaling, Voronoi bisectors, and stadium decay
 *        with negative values, zero-crossing ranges, and coordinate translation invariance.
 */

#include "geif/geif.h"
#include "geometry.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

int main(void)
{
    printf("=================================================================\n");
    printf("VERIFYING GEIF DIMENSION SCALING WITH NEGATIVE VALUES & SHIFTS\n");
    printf("=================================================================\n");

    // -------------------------------------------------------------------------
    // TEST 1: Voronoi Bisector in Deep Negative Coordinate Space
    // -------------------------------------------------------------------------
    printf("\nTest 1: Voronoi Bisector in Deep Negative Coordinate Space...\n");
    uint32_t d = 2;
    double effective_span[2] = { 8000.0, 1.6 }; // Positive spans
    uint8_t dim_active[2] = { 1, 1 };

    // Points with purely negative coordinates
    double A_neg[2] = { -9000.0, -1.8 };
    double B_neg[2] = { -1000.0, -0.2 };

    double normal[2] = { 0.0, 0.0 };
    double pdotn = 0.0;
    double delta = 0.0;

    bool ok = geif_compute_bisector(A_neg, B_neg, effective_span, dim_active, d, normal, &pdotn, &delta);
    assert(ok);
    assert(!isnan(normal[0]) && !isnan(normal[1]));
    assert(!isnan(pdotn) && !isnan(delta));

    // Midpoint: (-5000.0, -1.0) must sit exactly on the hyperplane
    double mid_neg[2] = { -5000.0, -1.0 };
    double dot_mid = geif_dot(mid_neg, normal, d);
    printf("  Hyperplane scalar pdotn = %.8f, dot(midpoint, n) = %.8f\n", pdotn, dot_mid);
    assert(fabs(dot_mid - pdotn) < 1e-9);

    // Sample A must be on one side, Sample B on the other
    double dot_A = geif_dot(A_neg, normal, d);
    double dot_B = geif_dot(B_neg, normal, d);
    printf("  dot(A, n) = %.8f (< pdotn), dot(B, n) = %.8f (> pdotn)\n", dot_A, dot_B);
    assert(dot_A < pdotn);
    assert(dot_B > pdotn);
    printf("  [PASS] Voronoi bisector with negative coordinates verified.\n");

    // -------------------------------------------------------------------------
    // TEST 2: Exact Translation Invariance (Positive vs. Shifted Negative)
    // -------------------------------------------------------------------------
    printf("\nTest 2: Exact Translation Invariance (Shift by -100,000)...\n");
    double A_pos[2] = { 1000.0, 10.0 };
    double B_pos[2] = { 5000.0, 50.0 };
    double X_pos[2] = { 2500.0, 20.0 };

    double norm_pos[2], pdotn_pos, delta_pos;
    geif_compute_bisector(A_pos, B_pos, effective_span, dim_active, d, norm_pos, &pdotn_pos, &delta_pos);

    // Apply large negative shift: -100,000 on dim 0, -500 on dim 1
    double shift[2] = { -100000.0, -500.0 };
    double A_shift[2] = { A_pos[0] + shift[0], A_pos[1] + shift[1] };
    double B_shift[2] = { B_pos[0] + shift[0], B_pos[1] + shift[1] };
    double X_shift[2] = { X_pos[0] + shift[0], X_pos[1] + shift[1] };

    double norm_shift[2], pdotn_shift, delta_shift;
    geif_compute_bisector(A_shift, B_shift, effective_span, dim_active, d, norm_shift, &pdotn_shift, &delta_shift);

    printf("  Original delta = %.8f, Shifted delta = %.8f\n", delta_pos, delta_shift);
    assert(fabs(delta_pos - delta_shift) < 1e-12);

    printf("  Original normal = [%.8f, %.8f], Shifted normal = [%.8f, %.8f]\n",
           norm_pos[0], norm_pos[1], norm_shift[0], norm_shift[1]);
    assert(fabs(norm_pos[0] - norm_shift[0]) < 1e-12);
    assert(fabs(norm_pos[1] - norm_shift[1]) < 1e-12);

    // Decision sign for arbitrary query point X must be identical
    double diff_pos = geif_dot(X_pos, norm_pos, d) - pdotn_pos;
    double diff_shift = geif_dot(X_shift, norm_shift, d) - pdotn_shift;
    printf("  Original (X.n - p) = %.8f, Shifted (X'.n - p') = %.8f\n", diff_pos, diff_shift);
    assert(fabs(diff_pos - diff_shift) < 1e-9);
    printf("  [PASS] Translation invariance mathematically confirmed.\n");

    // -------------------------------------------------------------------------
    // TEST 3: Outer Space Stadium Metric with Negative Bounding Boxes
    // -------------------------------------------------------------------------
    printf("\nTest 3: Stadium Outer Distance with Negative Bounding Box...\n");
    // Envelope: X in [-100, -20] (span 80), Y in [-50, -10] (span 40)
    double env_min[2] = { -100.0, -50.0 };
    double env_max[2] = {  -20.0, -10.0 };
    double spans[2]   = {   80.0,  40.0 };

    // Point 1: Inside envelope -> d_out must be 0.0
    double pt_inside[2] = { -60.0, -30.0 };
    double d_in = geif_stadium_distance(pt_inside, env_min, env_max, spans, d);
    printf("  Inside point (-60, -30): d_out = %.6f (expected 0.0)\n", d_in);
    assert(d_in == 0.0);

    // Point 2: Negative outer space: X = -140 (40 units below min) -> normalized = 40/80 = 0.5
    double pt_below[2] = { -140.0, -30.0 };
    double d_below = geif_stadium_distance(pt_below, env_min, env_max, spans, d);
    printf("  Below point (-140, -30): d_out = %.6f (expected 0.5)\n", d_below);
    assert(fabs(d_below - 0.5) < 1e-9);

    // Point 3: Positive outer space: X = +20 (40 units above max) -> normalized = 40/80 = 0.5
    double pt_above[2] = { 20.0, -30.0 };
    double d_above = geif_stadium_distance(pt_above, env_min, env_max, spans, d);
    printf("  Above point (+20, -30):  d_out = %.6f (expected 0.5)\n", d_above);
    assert(fabs(d_above - 0.5) < 1e-9);

    // Point 4: Corner in outer space: X = -180 (80 units below), Y = -90 (40 units below)
    // norm_X = 80/80 = 1.0, norm_Y = 40/40 = 1.0 -> d_out = sqrt(1^2 + 1^2) = sqrt(2)
    double pt_corner[2] = { -180.0, -90.0 };
    double d_corner = geif_stadium_distance(pt_corner, env_min, env_max, spans, d);
    printf("  Corner point (-180, -90): d_out = %.6f (expected %.6f)\n", d_corner, sqrt(2.0));
    assert(fabs(d_corner - sqrt(2.0)) < 1e-9);
    printf("  [PASS] Stadium distance handles negative bounds and symmetric distance accurately.\n");

    // -------------------------------------------------------------------------
    // TEST 4: Full Forest Training & Scoring with Negative Telemetry Data
    // -------------------------------------------------------------------------
    printf("\nTest 4: Full Forest Training on Negative Gaussian Cluster...\n");
    geif_config_t cfg = geif_config_default();
    cfg.tree_count = 50;
    cfg.samples_per_tree = 128;

    geif_forest_t *forest_neg = NULL;
    geif_status_t st = geif_forest_create(&forest_neg, 2, &cfg);
    assert(st == GEIF_OK && forest_neg != NULL);

    // Train on a 2D Gaussian centered at (-500.0, -200.0)
    srand(42);
    size_t n_samples = 400;
    for (size_t i = 0; i < n_samples; i++) {
        double r1 = (double)rand() / (double)RAND_MAX;
        double r2 = (double)rand() / (double)RAND_MAX;
        double x0 = -500.0 + (r1 - 0.5) * 50.0; // [-525, -475]
        double x1 = -200.0 + (r2 - 0.5) * 20.0; // [-210, -190]
        double pt[2] = { x0, x1 };
        geif_forest_feed(forest_neg, pt);
    }

    st = geif_forest_train(forest_neg);
    assert(st == GEIF_OK);
    printf("  Trained forest on %zu negative samples. Envelope: [%.1f..%.1f], [%.1f..%.1f]\n",
           n_samples,
           forest_neg->envelope_min[0], forest_neg->envelope_max[0],
           forest_neg->envelope_min[1], forest_neg->envelope_max[1]);

    // Inlier test at center (-500, -200)
    double pt_core[2] = { -500.0, -200.0 };
    double score_core = 0.0;
    geif_forest_score(forest_neg, pt_core, &score_core);
    printf("  Inlier center (-500, -200) anomaly score: %.6f (expected < 0.35)\n", score_core);
    assert(score_core < 0.35);

    // Outlier in deep negative outer space (-700, -200)
    double pt_out_neg[2] = { -700.0, -200.0 };
    double score_out_neg = 0.0;
    geif_forest_score(forest_neg, pt_out_neg, &score_out_neg);
    printf("  Outlier deep negative (-700, -200) anomaly score: %.6f (expected > 0.70)\n", score_out_neg);
    assert(score_out_neg > 0.70);

    // Outlier in positive outer space (+100, +50)
    double pt_out_pos[2] = { 100.0, 50.0 };
    double score_out_pos = 0.0;
    geif_forest_score(forest_neg, pt_out_pos, &score_out_pos);
    printf("  Outlier positive outer (+100, +50) anomaly score: %.6f (expected > 0.90)\n", score_out_pos);
    assert(score_out_pos > 0.90);

    geif_forest_destroy(forest_neg);
    printf("  [PASS] Full forest scoring on negative coordinates verified.\n");

    printf("\n>>> ALL NEGATIVE VALUE AND TRANSLATION INVARIANCE TESTS PASSED! <<<\n");
    return 0;
}
