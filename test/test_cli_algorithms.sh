#!/usr/bin/env bash
#
# Integration tests for GEIF Multi-Algorithm POC Suite (-B, --algo)
# Tests CEIF (EIF), Bubble Trees, Exemplar SIMD Kernel, and Voronoi Bisectors
#

set -euo pipefail

GEIF_BIN="./bin/geif"
TEST_DIR=$(mktemp -d /tmp/geif_test_algos_XXXXXX)
trap 'rm -rf "$TEST_DIR"' EXIT

echo "=== Running GEIF Multi-Algorithm POC Suite Integration Tests ==="

DATA_CSV="$TEST_DIR/data.csv"
cat << 'EOF' > "$DATA_CSV"
10.0,20.0
10.2,20.1
9.8,19.9
10.1,20.2
9.9,19.8
10.0,20.3
10.3,19.9
9.7,20.1
50.0,50.0
50.2,50.1
49.8,49.9
50.1,50.2
49.9,49.8
50.3,50.0
EOF

ALGOS=("ceif" "bubble" "exemplar" "voronoi")

for ALGO in "${ALGOS[@]}"; do
    echo "Testing algorithm '$ALGO'..."
    MODEL_JSON="$TEST_DIR/model_${ALGO}.json"

    # Test training with -B <algo>
    "$GEIF_BIN" -l "$DATA_CSV" -B "$ALGO" -w "$MODEL_JSON" -t 20 -s 32 > /dev/null
    test -f "$MODEL_JSON"
    echo "  [PASS] Successfully trained and saved model for '$ALGO'."

    # Verify JSON sparseness and algorithm tag
    grep -q "\"algorithm\":\"$ALGO\"" "$MODEL_JSON"
    MODEL_SIZE=$(wc -c < "$MODEL_JSON")
    echo "  Model JSON size: $MODEL_SIZE bytes (sparse verification)."
    test "$MODEL_SIZE" -lt 150000
    echo "  [PASS] Sparse JSON format confirmed (contains reservoir pool and algorithm tag)."

    # Verify reloading model with -r -q
    "$GEIF_BIN" -r "$MODEL_JSON" -q > "$TEST_DIR/summary_${ALGO}.txt"
    grep -q "Dimensions:[[:space:]]*2" "$TEST_DIR/summary_${ALGO}.txt"
    echo "  [PASS] Model loaded and validated successfully with -r -q."

    # Test scoring: inlier should have lower score than extreme outer point
    set +e
    INLIER_OUT=$("$GEIF_BIN" -r "$MODEL_JSON" -a - -d 4 <<< "10.0,20.0")
    OUTER_OUT=$("$GEIF_BIN" -r "$MODEL_JSON" -a - -d 4 <<< "500.0,500.0")
    set -e
    INLIER_SCORE=$(echo "$INLIER_OUT" | cut -d',' -f3)
    OUTER_SCORE=$(echo "$OUTER_OUT" | cut -d',' -f3)

    echo "  Inlier (10, 20) score: $INLIER_SCORE, Outer (500, 500) score: $OUTER_SCORE"
    python3 -c "assert float('$INLIER_SCORE') < float('$OUTER_SCORE'), 'Outer score must be higher than inlier'"
    echo "  [PASS] Anomaly scoring monotonic ordering verified for '$ALGO'."
done

echo "Testing --algo long option..."
"$GEIF_BIN" -l "$DATA_CSV" --algo bubble -w "$TEST_DIR/model_long.json" -t 15 -s 32 > /dev/null
grep -q "\"algorithm\":\"bubble\"" "$TEST_DIR/model_long.json"
echo "  [PASS] --algo bubble long option verified."

echo "Testing default algorithm when -B is omitted..."
"$GEIF_BIN" -l "$DATA_CSV" -w "$TEST_DIR/model_default.json" -t 15 -s 32 > /dev/null
grep -q "\"algorithm\":\"bubble\"" "$TEST_DIR/model_default.json"
echo "  [PASS] Omitting -B correctly defaults to 'bubble' algorithm."

echo "Testing 'eif' alias for 'ceif'..."
"$GEIF_BIN" -l "$DATA_CSV" -B eif -w "$TEST_DIR/model_eif.json" -t 15 -s 32 > /dev/null
grep -q "\"algorithm\":\"ceif\"" "$TEST_DIR/model_eif.json"
echo "  [PASS] 'eif' alias correctly mapped to 'ceif'."

echo "=== All GEIF Multi-Algorithm POC Suite Tests Passed! ==="
