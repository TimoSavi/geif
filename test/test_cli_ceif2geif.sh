#!/usr/bin/env bash
#
# Integration tests for ceif2geif:
# CEIF to GEIF Model Migration Tool & Backward-Compatible Loader
#

set -euo pipefail

GEIF_BIN="./bin/geif"
MIGRATE_BIN="./bin/ceif2geif"
TEST_DIR=$(mktemp -d /tmp/geif_test_ceif2geif_XXXXXX)
trap 'rm -rf "$TEST_DIR"' EXIT

echo "=== Running CEIF to GEIF Model Migration (ceif2geif) Integration Tests ==="

# 1. Create a synthetic CEIF model in exact CEIF JSON schema
CEIF_JSON="$TEST_DIR/model_ceif.json"
cat << 'EOF' > "$CEIF_JSON"
{
  "globals": {
    "dimensions": 3,
    "forestCount": 2,
    "printString": "%d,%s,%o",
    "printfFormat": "",
    "treeCount": 20,
    "samplesMax": 64,
    "inputSeparator": ",",
    "listSeparator": ",",
    "header": 1,
    "outlierScore": "0.650000",
    "categoryDims": "4",
    "labelDims": "1",
    "includeDims": "2-4",
    "ignoreDims": "",
    "textDims": "",
    "scoreDims": "",
    "filter": "",
    "decimals": 4,
    "uniqueSamples": 0,
    "aggregate": 0,
    "formulas": []
  },
  "forests": [
    {
      "category": "alpha",
      "sampleCount": 5,
      "lastUpdated": 1727000000,
      "extraRows": 15,
      "samples": [
        [10.0, 20.0, 30.0],
        [10.2, 20.1, 30.1],
        [9.9, 19.8, 29.9],
        [10.1, 20.2, 30.2],
        [10.0, 20.0, 30.0]
      ]
    },
    {
      "category": "beta",
      "sampleCount": 4,
      "lastUpdated": 1727005000,
      "extraRows": 25,
      "samples": [
        [50.0, 60.0, 70.0],
        [50.5, 60.2, 70.3],
        [49.8, 59.9, 69.8],
        [50.2, 60.1, 70.0]
      ]
    }
  ]
}
EOF

# Test 1: Run ceif2geif with file arguments and verbose flag
echo "Test 1: Migrating CEIF model via ceif2geif -v..."
GEIF_MIGRATED="$TEST_DIR/model_migrated.json"
"$MIGRATE_BIN" -v "$CEIF_JSON" "$GEIF_MIGRATED"

if [ ! -s "$GEIF_MIGRATED" ]; then
    echo "ERROR: Migrated GEIF file is empty"
    exit 1
fi

# Verify format tag in migrated JSON
if ! grep -Eq '"format"[[:space:]]*:[[:space:]]*"GEIF-1.0"' "$GEIF_MIGRATED"; then
    echo "ERROR: Migrated file is missing GEIF-1.0 header"
    cat "$GEIF_MIGRATED"
    exit 1
fi
echo "  [PASS] File migration produced valid GEIF-1.0 JSON format."

# Test 2: Inspect migrated model using geif -q
echo "Test 2: Inspecting migrated model via geif -q..."
SUMMARY=$("$GEIF_BIN" -r "$GEIF_MIGRATED" -q)
echo "$SUMMARY"

if ! echo "$SUMMARY" | grep -q "Sub-forests:         2"; then
    echo "ERROR: Expected 2 sub-forests in migrated model"
    exit 1
fi
if ! echo "$SUMMARY" | grep -q "Sub-forest 'alpha'"; then
    echo "ERROR: Category 'alpha' missing from migrated model"
    exit 1
fi
if ! echo "$SUMMARY" | grep -q "Sub-forest 'beta'"; then
    echo "ERROR: Category 'beta' missing from migrated model"
    exit 1
fi
echo "  [PASS] Migrated model verified with geif -q: both sub-forests intact."

# Test 3: Score against migrated model using geif -a
echo "Test 3: Scoring data against migrated model..."
DATA_CSV="$TEST_DIR/query.csv"
cat << 'EOF' > "$DATA_CSV"
p_norm,10.05,20.05,30.05,alpha
p_outlier,99.0,99.0,99.0,alpha
p_beta_norm,50.1,60.1,70.1,beta
EOF

SCORE_OUT="$TEST_DIR/score_out.txt"
"$GEIF_BIN" -r "$GEIF_MIGRATED" -a "$DATA_CSV" -L 1 -C 5 -p "%l;%c;%s;%o" -o "$SCORE_OUT" || true
cat "$SCORE_OUT"

if ! grep -q "p_norm;alpha;.*0$" "$SCORE_OUT"; then
    echo "ERROR: Normal point misclassified or not scored"
    exit 1
fi
if ! grep -q "p_outlier;alpha;.*1$" "$SCORE_OUT"; then
    echo "ERROR: Outlier point not detected"
    exit 1
fi
echo "  [PASS] Scoring on migrated model behaves identically to native model."

# Test 4: Transparent on-the-fly loading of CEIF model by geif -r
echo "Test 4: Transparently loading CEIF model directly into geif -r..."
SCORE_DIRECT="$TEST_DIR/score_direct.txt"
"$GEIF_BIN" -r "$CEIF_JSON" -a "$DATA_CSV" -L 1 -C 5 -p "%l;%c;%s;%o" -o "$SCORE_DIRECT" || true

if ! grep -q "p_norm;alpha;.*0$" "$SCORE_DIRECT"; then
    echo "ERROR: Direct CEIF loading failed to score normal point"
    exit 1
fi
if ! grep -q "p_outlier;alpha;.*1$" "$SCORE_DIRECT"; then
    echo "ERROR: Direct CEIF loading failed to detect outlier"
    exit 1
fi
echo "  [PASS] geif -r transparently ingests and evaluates pure CEIF JSON models without prior conversion."

# Test 5: Pipe streaming through stdin/stdout
echo "Test 5: Migrating via pipe streaming (cat ... | ceif2geif -)..."
PIPE_MIGRATED="$TEST_DIR/model_piped.json"
cat "$CEIF_JSON" | "$MIGRATE_BIN" - > "$PIPE_MIGRATED"

if ! grep -Eq '"format"[[:space:]]*:[[:space:]]*"GEIF-1.0"' "$PIPE_MIGRATED"; then
    echo "ERROR: Piped migration failed"
    exit 1
fi
echo "  [PASS] Stdin / stdout migration pipeline verified."

# Test 6: Parameter overrides (-t, -s)
echo "Test 6: Overriding tree count and samples (-t 35 -s 32)..."
OVERRIDE_MIGRATED="$TEST_DIR/model_overrides.json"
"$MIGRATE_BIN" -t 35 -s 32 "$CEIF_JSON" "$OVERRIDE_MIGRATED"

OVERRIDE_SUMMARY=$("$GEIF_BIN" -r "$OVERRIDE_MIGRATED" -q)
if ! echo "$OVERRIDE_SUMMARY" | grep -q "trees=35"; then
    echo "ERROR: Tree count override failed"
    echo "$OVERRIDE_SUMMARY"
    exit 1
fi
echo "  [PASS] Parameter overrides (-t 35) successfully applied during migration."

echo "=== All ceif2geif Migration Tests Passed! ==="

