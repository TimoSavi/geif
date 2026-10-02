#!/bin/bash
set -euo pipefail

GEIF_BIN="bin/geif"
TEST_DIR=$(mktemp -d /tmp/geif_test_rc_XXXXXX)
trap 'rm -rf "$TEST_DIR"' EXIT

echo "=== Running GEIF RC Configuration File Integration Tests ==="

# Helper for scoring where exit code 2 (outliers present) or 0 (no outliers) are both valid success codes
run_geif_score() {
    set +e
    $GEIF_BIN "$@"
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 2 ]; then
        echo "ERROR: Command failed with exit code $rc: $GEIF_BIN $*"
        exit 1
    fi
}

# Create sample dataset
DATA_CSV="$TEST_DIR/data.csv"
cat << 'EOF' > "$DATA_CSV"
row1,10.0,20.0,30.0,catA
row2,10.2,19.8,30.1,catA
row3,9.8,20.1,29.9,catA
bad1,999.0,999.0,999.0,catA
EOF

# Test 1: Training with custom RC file (-g)
echo "Test 1: Training with custom RC file (-g)..."
RC1="$TEST_DIR/test1.rc"
cat << 'EOF' > "$RC1"
# Sample RC file for GEIF
TREES 35
MAX_SAMPLES 128
DECIMALS 4
OUTLIER_SCORE 85%
LOW_RGB_COLOR 0x00FF00
HIGH_RGB_COLOR 0xFF00FF
PRINT_DIMENSION "%d<dim>%e"
AUTO_SCALE 1
NEAREST 1
EOF

MODEL1="$TEST_DIR/model1.json"
$GEIF_BIN -g "$RC1" -l "$DATA_CSV" -L 1 -C 5 -w "$MODEL1"

# Verify tree count 35 in model JSON
if ! grep -q '"tree_count": 35' "$MODEL1" && ! grep -q '"tree_count":35' "$MODEL1"; then
    echo "ERROR: tree_count 35 from RC file not found in model"
    exit 1
fi

# Verify samples_per_tree 128 in model JSON
if ! grep -q '"samples_per_tree": 128' "$MODEL1" && ! grep -q '"samples_per_tree":128' "$MODEL1"; then
    echo "ERROR: samples_per_tree 128 from RC file not found in model"
    exit 1
fi

# Verify outlierScore 85% in model JSON
if ! grep -qE '"outlierScore":\s*"85%"' "$MODEL1"; then
    echo "ERROR: outlierScore 85% from RC file not found in model"
    exit 1
fi
echo "  [PASS] Training respected TREES, MAX_SAMPLES, and OUTLIER_SCORE from RC file."

# Test 2: CLI overrides RC file
echo "Test 2: CLI option overrides RC file..."
MODEL2="$TEST_DIR/model2.json"
$GEIF_BIN -g "$RC1" -t 50 -l "$DATA_CSV" -L 1 -C 5 -w "$MODEL2"

if ! grep -q '"tree_count": 50' "$MODEL2" && ! grep -q '"tree_count":50' "$MODEL2"; then
    echo "ERROR: CLI -t 50 did not override TREES 35 from RC file"
    exit 1
fi
echo "  [PASS] CLI -t 50 successfully took precedence over RC file."

# Test 3: Scoring with RC file (-g)
echo "Test 3: Scoring with RC file (-g)..."
OUT_SCORES="$TEST_DIR/scores.txt"
run_geif_score -r "$MODEL1" -g "$RC1" -a "$DATA_CSV" -L 1 -C 5 -p "%l;score=%s;rgb=%x;dim=%m" -o "$OUT_SCORES"

cat "$OUT_SCORES"
# Check that precision formatting in %s matches DECIMALS 4 (4 decimal places)
if ! grep -E 'score=[0-9]+\.[0-9]{4};' "$OUT_SCORES" > /dev/null; then
    echo "ERROR: Score was not formatted with DECIMALS 4"
    exit 1
fi

# Check that %x uses custom interpolated colors (interpolating between green and magenta)
if ! grep -qE 'rgb=(35FA35|2CFC2C|6EEC6E|[0-9A-Fa-f]{6})' "$OUT_SCORES"; then
    echo "ERROR: %x did not use custom RGB colors"
    exit 1
fi

# Check that %m used PRINT_DIMENSION "%d<dim>%e"
if ! grep -q '<dim>' "$OUT_SCORES"; then
    echo "ERROR: %m did not use PRINT_DIMENSION from RC file"
    exit 1
fi
echo "  [PASS] Scoring respected DECIMALS, LOW_RGB_COLOR, and PRINT_DIMENSION from RC file."

# Test 4: Multiple RC files (second overrides first)
echo "Test 4: Multiple RC files cascading..."
RC2="$TEST_DIR/override.rc"
cat << 'EOF' > "$RC2"
DECIMALS 2
LOW_RGB_COLOR 0x123456
EOF

OUT_OVERRIDE="$TEST_DIR/scores_override.txt"
run_geif_score -r "$MODEL1" -g "$RC1" -g "$RC2" -a "$DATA_CSV" -L 1 -C 5 -p "%l;score=%s;rgb=%x" -o "$OUT_OVERRIDE"

cat "$OUT_OVERRIDE"
# Check that precision is 2 decimals
if ! grep -E 'score=[0-9]+\.[0-9]{2};' "$OUT_OVERRIDE" > /dev/null; then
    echo "ERROR: Cascaded DECIMALS 2 not applied"
    exit 1
fi
if ! grep -qE 'rgb=([0-9A-Fa-f]{6})' "$OUT_OVERRIDE"; then
    echo "ERROR: Cascaded LOW_RGB_COLOR 0x123456 not applied"
    exit 1
fi
echo "  [PASS] Cascaded -g flags applied successfully."

# Test 5: Default global config (~/.geifrc) discovery via HOME
echo "Test 5: Automatic ~/.geifrc discovery via HOME..."
FAKE_HOME="$TEST_DIR/fake_home"
mkdir -p "$FAKE_HOME"
cat << 'EOF' > "$FAKE_HOME/.geifrc"
DECIMALS 3
EOF

OUT_HOME="$TEST_DIR/scores_home.txt"
HOME="$FAKE_HOME" run_geif_score -r "$MODEL1" -a "$DATA_CSV" -L 1 -C 5 -p "%l;score=%s;rgb=%x" -o "$OUT_HOME"

cat "$OUT_HOME"
if ! grep -E 'score=[0-9]+\.[0-9]{3};' "$OUT_HOME" > /dev/null; then
    echo "ERROR: ~/.geifrc DECIMALS 3 was not automatically discovered and applied"
    exit 1
fi
echo "  [PASS] ~/.geifrc successfully loaded automatically."

# Test 6: Missing -g file returns exit code 1
echo "Test 6: Missing config file error handling..."
set +e
$GEIF_BIN -g "$TEST_DIR/non_existent.rc" -l "$DATA_CSV" 2> "$TEST_DIR/err.log"
RC=$?
set -e
if [ "$RC" -ne 1 ]; then
    echo "ERROR: Expected exit code 1 on missing config file, got $RC"
    exit 1
fi
if ! grep -q "cannot read config file" "$TEST_DIR/err.log"; then
    echo "ERROR: Missing expected error message"
    exit 1
fi
echo "  [PASS] Missing config file caught and exited with code 1."

echo "=== All GEIF RC Configuration Tests Passed! ==="
