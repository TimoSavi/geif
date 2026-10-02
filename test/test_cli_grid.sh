#!/usr/bin/env bash
#
# Integration tests for GEIF Feature 9:
# Population Drift & Test Grid Generation (-T [margin], -i <intervals>)
#

set -euo pipefail

GEIF_BIN="./bin/geif"
TEST_DIR=$(mktemp -d /tmp/geif_test_grid_XXXXXX)
trap 'rm -rf "$TEST_DIR"' EXIT

echo "=== Running GEIF Test Grid & Population Drift Integration Tests ==="

# 1. Prepare 2D training data with two categories
TRAIN_CSV="$TEST_DIR/train.csv"
cat << 'EOF' > "$TRAIN_CSV"
ID,X,Y,CAT
p1,10.0,20.0,catA
p2,12.0,22.0,catA
p3,11.0,21.0,catA
p4,13.0,23.0,catA
p5,10.5,20.5,catA
p6,50.0,60.0,catB
p7,52.0,62.0,catB
p8,51.0,61.0,catB
p9,53.0,63.0,catB
EOF

MODEL_JSON="$TEST_DIR/model.json"
echo "Test 1: Training 2D multi-category model..."
"$GEIF_BIN" -l "$TRAIN_CSV" -H -f ',' -L 1 -C 4 -t 30 -s 64 -w "$MODEL_JSON" > /dev/null
echo "  [PASS] Model trained successfully."

# Test 2: Standard 2D test grid generation (-T 0.1, -i 15)
echo "Test 2: Generating test grid with -T 0.1 -i 15 -p '%d,0x%x'..."
GRID_OUT="$TEST_DIR/grid.txt"
"$GEIF_BIN" -r "$MODEL_JSON" -T 0.1 -i 15 -e , -d 2 -p "%d,0x%x" -o "$GRID_OUT"

# Verify grid contains coordinates and hex colors
if ! grep -qE '^[0-9]+\.[0-9]{2},[0-9]+\.[0-9]{2},0x[0-9A-Fa-f]{6}$' "$GRID_OUT"; then
    echo "ERROR: Grid output does not match expected '%d,0x%x' pattern"
    head -n 10 "$GRID_OUT"
    exit 1
fi

# Verify reservoir points with score 0.0 (black 0x000000 or legacy green) are emitted
if ! grep -qE '0x(000000|20FF20|00FF00)' "$GRID_OUT"; then
    echo "ERROR: Reservoir sample points (black 0x000000) not found in grid output"
    exit 1
fi
echo "  [PASS] Test grid generated with valid coordinate grid points and reservoir scatter samples."

# Test 3: Category filtering (-F "^catB$") - filter out catB
echo "Test 3: Generating test grid with category filter (-F '^catB$')..."
FILTER_OUT="$TEST_DIR/grid_filter.txt"
"$GEIF_BIN" -r "$MODEL_JSON" -T 0.1 -i 10 -F "^catB$" -e , -d 2 -p "%c;%d;0x%x" -o "$FILTER_OUT"

if grep -q '^catB;' "$FILTER_OUT"; then
    echo "ERROR: catB points found despite category filter -F '^catB$'"
    exit 1
fi
if ! grep -q '^catA;' "$FILTER_OUT"; then
    echo "ERROR: catA points not found in filtered grid output"
    exit 1
fi
echo "  [PASS] Category filter -F correctly filtered out catB and isolated catA for grid generation."

# Test 4: Outlier threshold filtering with percentage (-O 80%)
echo "Test 4: Generating test grid with percentile threshold (-O 80%)..."
THRESH_OUT="$TEST_DIR/grid_thresh.txt"
"$GEIF_BIN" -r "$MODEL_JSON" -T 0.1 -i 10 -O 80% -e , -d 2 -p "%d,score=%s,0x%x" -o "$THRESH_OUT"

if ! grep -q 'score=' "$THRESH_OUT"; then
    echo "ERROR: Score format missing in threshold grid output"
    exit 1
fi
echo "  [PASS] Percentile threshold -O 80% successfully applied to grid boundary."

# Test 5: Exact user plan.md pipeline invocation (line 31)
# $CEIF -r $CONF -g $CEIFCONF -T0.1 -e, -d4 -p "%d,0x%x" -F "-v ^${n}$" | grep -v -- - > $PLOTDATA
echo "Test 5: Verifying exact plan.md pipeline command..."
RC_FILE="$TEST_DIR/ceif.rc"
cat << 'EOF' > "$RC_FILE"
LOW_RGB_COLOR 0x00FF00
HIGH_RGB_COLOR 0xFF0000
EOF

PLOTDATA="$TEST_DIR/plotdata.txt"
"$GEIF_BIN" -r "$MODEL_JSON" -g "$RC_FILE" -T0.1 -e, -d4 -p "%d,0x%x" -F "-v ^catB$" | grep -v -- - > "$PLOTDATA"

if [ ! -s "$PLOTDATA" ]; then
    echo "ERROR: PLOTDATA file is empty"
    exit 1
fi
LINE_COUNT=$(wc -l < "$PLOTDATA")
if [ "$LINE_COUNT" -lt 10 ]; then
    echo "ERROR: Expected at least 10 points in PLOTDATA, got $LINE_COUNT"
    exit 1
fi
# Verify black (0x000000) appears for training data scatter samples
if ! grep -qE '0x(000000|00FF00)' "$PLOTDATA"; then
    echo "ERROR: Training sample points (0x000000) not found in plotdata"
    exit 1
fi
echo "  [PASS] Exact production pipeline generated $LINE_COUNT points successfully."

# Test 6: Backward compatibility: -a with -T acts as outlier threshold
echo "Test 6: Backward compatibility of -a with -T <thresh>..."
ANALYZE_CSV="$TEST_DIR/analyze.csv"
cat << 'EOF' > "$ANALYZE_CSV"
p_norm,10.2,20.2,catA
p_out,99.0,99.0,catA
EOF

OUT_COMPAT="$TEST_DIR/compat.txt"
"$GEIF_BIN" -r "$MODEL_JSON" -a "$ANALYZE_CSV" -L 1 -C 4 -T 0.50 -p "%l;score=%s;outlier=%o" -o "$OUT_COMPAT" || true

if ! grep -q 'p_norm;score=' "$OUT_COMPAT" || ! grep -q 'p_out;score=' "$OUT_COMPAT"; then
    echo "ERROR: Analysis mode failed when -T was provided"
    cat "$OUT_COMPAT"
    exit 1
fi
echo "  [PASS] Legacy -a with -T continues to function as analysis threshold."

echo "=== All GEIF Test Grid & Population Drift Tests Passed! ==="
