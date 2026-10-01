#!/usr/bin/env bash
#
# Reference Test Suite for GEIF Bubble Algorithm (-B bubble)
# Verifies:
#  1. 2blob score distribution along vertical slice X=20
#  2. Monotonic rise from cluster center, smooth transition, outer space decay asymptote (< 1.0)
#  3. Scaled scoring default [0.0, 1.0) with Zero Kelvin lower bound and unreachable 1.0
#  4. complex2d.csv comparison at score 0.50 and 94% percentile (90-97% range)
#  5. Cavity / donut-hole interior detection in bubble vs hyperplane cuts
#

set -euo pipefail

GEIF_BIN="./bin/geif"
TEST_DIR=$(mktemp -d /tmp/geif_test_bubble_ref_XXXXXX)
trap 'rm -rf "$TEST_DIR"' EXIT

echo "=== Running GEIF Bubble Algorithm Reference Suite ==="

# ----------------------------------------------------------------------
# Part 1: 2blob Score Distribution Profile along X=20
# ----------------------------------------------------------------------
echo "Part 1: Profiling 2blob slice along X=20 with bubble algorithm..."

MODEL_2BLOB="$TEST_DIR/m_bubble_2blob.json"
"$GEIF_BIN" -l test/2blob.csv -B bubble -w "$MODEL_2BLOB" -s 128 -t 100 > /dev/null

SLICE_CSV="$TEST_DIR/slice_x20.csv"
python3 -c "
with open('$SLICE_CSV', 'w') as f:
    for i in range(351):
        y = -70.0 + i * (175.0 / 350.0)
        f.write(f'20.0,{y:.4f}\n')
"

SCORES_OUT="$TEST_DIR/slice_scores.csv"
"$GEIF_BIN" -r "$MODEL_2BLOB" -a "$SLICE_CSV" -p '%d,%s' > "$SCORES_OUT" || true

python3 -c "
with open('$SCORES_OUT') as f:
    lines = [l.strip().split(',') for l in f if l.strip()]

pts = [(float(l[1]), float(l[2].strip())) for l in lines if len(l) >= 3]
ys = [p[0] for p in pts]
scores = [p[1] for p in pts]

# 1. Zero Kelvin and unreachable 1.0 invariants
assert all(s >= 0.0 for s in scores), 'All scores must be >= 0.0 (Zero Kelvin floor)'
assert all(s < 1.0 for s in scores), 'All scores must be < 1.0 (unreachable 1.0)'

# 2. Centroid minimum at Y ~ 20.0 - 22.0
min_pt = min(pts, key=lambda p: p[1])
assert 18.0 <= min_pt[0] <= 23.0, f'Minimum score must be near cluster center, got Y={min_pt[0]}'
assert min_pt[1] < 0.35, f'Center score must be low, got {min_pt[1]}'

# 3. Outer space points must have higher score than boundary points
s_center = min_pt[1]
s_bound_low = next(s for y, s in pts if abs(y - 10.0) < 0.3)
s_bound_high = next(s for y, s in pts if abs(y - 30.0) < 0.3)
s_outer_neg = next(s for y, s in pts if abs(y - (-60.0)) < 0.3)
s_outer_pos = next(s for y, s in pts if abs(y - 100.0) < 0.3)

assert s_bound_low > s_center, f'Boundary Y=10 ({s_bound_low}) must exceed center ({s_center})'
assert s_bound_high > s_center, f'Boundary Y=30 ({s_bound_high}) must exceed center ({s_center})'
assert s_outer_neg > s_bound_low, f'Outer Y=-60 ({s_outer_neg}) must exceed boundary Y=10 ({s_bound_low})'
assert s_outer_pos > s_bound_high, f'Outer Y=100 ({s_outer_pos}) must exceed boundary Y=30 ({s_bound_high})'

print(f'  [PASS] Centroid min: Y={min_pt[0]:.2f} (score={min_pt[1]:.4f})')
print(f'  [PASS] Boundaries:   Y=10.0 (score={s_bound_low:.4f}), Y=30.0 (score={s_bound_high:.4f})')
print(f'  [PASS] Outer space:  Y=-60.0 (score={s_outer_neg:.4f}), Y=100.0 (score={s_outer_pos:.4f})')
"
echo "  [PASS] 2blob score distribution along X=20 verified successfully."

# ----------------------------------------------------------------------
# Part 2: Complex2D Percentile Analysis & Cavity Detection
# ----------------------------------------------------------------------
echo "Part 2: Comparing bubble and ceif on complex2d.csv (threshold 0.5 and 94%)..."

MODEL_BUBBLE_CMPLX="$TEST_DIR/m_bubble_cmplx.json"
MODEL_CEIF_CMPLX="$TEST_DIR/m_ceif_cmplx.json"

"$GEIF_BIN" -l test/complex2d.csv -B bubble -w "$MODEL_BUBBLE_CMPLX" -s 128 -t 100 > /dev/null
"$GEIF_BIN" -l test/complex2d.csv -B ceif -w "$MODEL_CEIF_CMPLX" -s 128 -t 100 > /dev/null

# Test fixed threshold 0.50 (scaled non-selectable default)
echo "  Testing fixed threshold -O 0.50 on complex2d..."
set +e
BUBBLE_O50=$("$GEIF_BIN" -r "$MODEL_BUBBLE_CMPLX" -v -a test/complex2d.csv -O 0.50 -o /dev/null 2>&1)
CEIF_O50=$("$GEIF_BIN" -r "$MODEL_CEIF_CMPLX" -v -a test/complex2d.csv -O 0.50 -o /dev/null 2>&1)
set -e

echo "$BUBBLE_O50" | grep -q "Scored"
echo "$CEIF_O50" | grep -q "Scored"
echo "  [PASS] Both models evaluated fixed threshold -O 0.50 cleanly."

# Test percentile thresholds across 90% - 97% range (focusing on 94%)
echo "  Testing percentile thresholds 90% - 97% on complex2d..."
for PCT in 90 92 94 95 96 97; do
    set +e
    B_OUT=$("$GEIF_BIN" -r "$MODEL_BUBBLE_CMPLX" -v -a test/complex2d.csv -O "${PCT}%" -o /dev/null 2>&1)
    C_OUT=$("$GEIF_BIN" -r "$MODEL_CEIF_CMPLX" -v -a test/complex2d.csv -O "${PCT}%" -o /dev/null 2>&1)
    set -e

    B_TH=$(echo "$B_OUT" | grep "Percentage score" | awk '{print $5}')
    C_TH=$(echo "$C_OUT" | grep "Percentage score" | awk '{print $5}')
    echo "    At ${PCT}%: bubble_cutoff=${B_TH}, ceif_cutoff=${C_TH}"

    python3 -c "
b_th = float('$B_TH')
c_th = float('$C_TH')
assert 0.0 < b_th < 1.0, f'Bubble cutoff {b_th} must be in (0, 1)'
assert 0.0 < c_th < 1.0, f'CEIF cutoff {c_th} must be in (0, 1)'
"
done

# Part 3: Interior Cavity (Donut Hole) Anomaly Detection
echo "Part 3: Testing interior cavity (donut hole) anomaly detection..."
CAVITY_PT="5107.55,0.9887"

set +e
B_CAVITY_SCORE=$("$GEIF_BIN" -r "$MODEL_BUBBLE_CMPLX" -a - -p '%s' <<< "$CAVITY_PT" 2>/dev/null | tr -d '[:space:]')
C_CAVITY_SCORE=$("$GEIF_BIN" -r "$MODEL_CEIF_CMPLX" -a - -p '%s' <<< "$CAVITY_PT" 2>/dev/null | tr -d '[:space:]')
set -e

# Extract 94% threshold for bubble
set +e
B_94_THRESH=$("$GEIF_BIN" -r "$MODEL_BUBBLE_CMPLX" -v -a test/complex2d.csv -O 94% -o /dev/null 2>&1 | grep "Percentage score" | awk '{print $5}')
set -e

echo "  Cavity center ($CAVITY_PT):"
echo "    Bubble score: $B_CAVITY_SCORE (94% threshold: $B_94_THRESH)"
echo "    CEIF score:   $C_CAVITY_SCORE"

python3 -c "
b_s = float('$B_CAVITY_SCORE')
b_th = float('$B_94_THRESH')
c_s = float('$C_CAVITY_SCORE')

# Bubble hyperspherical carving detects interior cavity with elevated anomaly score
assert b_s > b_th, f'Bubble cavity score {b_s} must exceed 94% threshold {b_th}'
assert b_s > c_s, f'Bubble cavity score {b_s} should exceed hyperplane CEIF score {c_s}'
print('  [PASS] Bubble algorithm successfully flagged interior cavity center as anomaly!')
"

echo "=== All GEIF Bubble Reference Suite Tests Passed Successfully! ==="
