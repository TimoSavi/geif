#!/usr/bin/env bash
set -euo pipefail

GEIF="./bin/geif"
TMPDIR=$(mktemp -d /tmp/geif_test_aggregate_XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

echo "[1/5] Generating daily traffic datasets for -A (--aggregate) testing..."

# Generate 16 daily files with symmetric offsets around:
#   Category "12;In"  mean = (250, 2500)
#   Category "12;Out" mean = (110, 1100)
# Each day has multiple rows per category that must be summed by -A into 1 sample.
DX_LIST=(0 0 0 0 0 0 0 0 -8 -6 -4 -2 2 4 6 8)
DY_LIST=(0 0 0 0 0 0 0 0 -80 60 -40 20 -20 40 -60 80)

for idx in "${!DX_LIST[@]}"; do
    day=$((idx + 1))
    dx=${DX_LIST[$idx]}
    dy=${DY_LIST[$idx]}
    in_x=$((250 + dx))
    in_y=$((2500 + dy))
    out_x=$((110 + dx))
    out_y=$((1100 + dy))

    # Split each category's daily total across multiple rows
    in_x1=100; in_y1=1000
    in_x2=$((in_x - in_x1)); in_y2=$((in_y - in_y1))
    out_x1=50; out_y1=500
    out_x2=$((out_x - out_x1)); out_y2=$((out_y - out_y1))

    cat << EOF > "$TMPDIR/day${day}.csv"
Hour,Type,Inbytes,Outbytes
12,In,${in_x1},${in_y1}
12,In,${in_x2},${in_y2}
12,Out,${out_x1},${out_y1}
12,Out,${out_x2},${out_y2}
EOF
done

MODEL="$TMPDIR/traffic.json"

echo "[2/5] Training Day 1 with -A and updating Days 2-16 via -z (verifying -A persistence)..."
"$GEIF" -l "$TMPDIR/day1.csv" -C 1-2 -H -A -d 0 -w "$MODEL" > /dev/null

# Verify Day 1 model has 1 sample per category and aggregate=true saved
Q1=$("$GEIF" -r "$MODEL" -q)
if ! grep -q "Aggregate (-A):.*Yes" <<< "$Q1"; then
    echo "FAIL: Aggregate (-A): Yes not reported in -q summary"
    echo "$Q1"
    exit 1
fi
if ! grep -q "rows=1, samples=1" <<< "$Q1"; then
    echo "FAIL: Expected 1 aggregated sample per category after Day 1"
    echo "$Q1"
    exit 1
fi

# Update Days 2-16 using -z WITHOUT passing -A, -C, or -H
for day in $(seq 2 16); do
    "$GEIF" -z "$MODEL" -l "$TMPDIR/day${day}.csv" > /dev/null
done

Q16=$("$GEIF" -r "$MODEL" -q)
if ! grep -q "rows=16, samples=16" <<< "$Q16"; then
    echo "FAIL: Expected 16 aggregated samples per category after Day 16"
    echo "$Q16"
    exit 1
fi
echo "  PASS: 16 daily sessions aggregated into 16 samples per category with persisted -A."

echo "[3/5] Testing aggregated analysis (-a) with normal, anomalous, new (-N), and missing (-M) categories..."
# Test Day:
# "12;In": (125, 1250) + (125, 1250) = (250, 2500) -> exact mean of [240,250,260] x [2400,2500,2600] (inlier)
# "12;Out": (2500, 25000) + (2500, 25000) = (5000, 50000) -> huge spike vs [100,110,120] x [1000,1100,1200] (outlier)
# "99;New": unseen category -> should trigger -N immediately
cat << 'EOF' > "$TMPDIR/test_day.csv"
Hour,Type,Inbytes,Outbytes
12,In,125,1250
12,In,125,1250
12,Out,2500,25000
12,Out,2500,25000
99,New,10,100
EOF

ANA_OUT="$TMPDIR/ana_out.txt"
set +e
"$GEIF" -r "$MODEL" -a "$TMPDIR/test_day.csv" -O 0.5 -e ';' -d 0 \
    -p "ANOM;%C;%d;%a" -N "NEW;%c;%d" -v "SUM;%C;%o;%h;%a" > "$ANA_OUT"
RC=$?
set -e

if [ "$RC" -ne 2 ]; then
    echo "FAIL: Expected exit code 2 (anomalies detected), got $RC"
    cat "$ANA_OUT"
    exit 1
fi

# Verify NEW category reported
if ! grep -q "^NEW;99;New;10;100$" "$ANA_OUT"; then
    echo "FAIL: Expected NEW;99;New;10;100 in output"
    cat "$ANA_OUT"
    exit 1
fi

# Verify 12;Out reported as anomaly with summed dimensions 5000;50000 and average 110;1100
if ! grep -q "^ANOM;12;Out;5000;50000;110;1100$" "$ANA_OUT"; then
    echo "FAIL: Expected ANOM;12;Out;5000;50000;110;1100 in output"
    cat "$ANA_OUT"
    exit 1
fi

# Verify 12;In NOT reported as anomaly (inlier at centroid 250;2500)
if grep -q "^ANOM;12;In;" "$ANA_OUT"; then
    echo "FAIL: 12;In should not be flagged as anomaly"
    cat "$ANA_OUT"
    exit 1
fi

# Verify -v summary lines show 1 analyzed aggregated sample (%o = 1) per category
if ! grep -q "^SUM;12;In;1;0;250;2500$" "$ANA_OUT"; then
    echo "FAIL: Expected SUM;12;In;1;0;250;2500 in output"
    cat "$ANA_OUT"
    exit 1
fi
if ! grep -q "^SUM;12;Out;1;1;110;1100$" "$ANA_OUT"; then
    echo "FAIL: Expected SUM;12;Out;1;1;110;1100 in output"
    cat "$ANA_OUT"
    exit 1
fi
echo "  PASS: Aggregated analysis (-a) accurately sums dimensions and detects daily anomalies."

echo "[4/5] Testing missing category detection (-M) in aggregated analysis..."
cat << 'EOF' > "$TMPDIR/partial_day.csv"
Hour,Type,Inbytes,Outbytes
12,In,120,1200
12,In,130,1300
EOF

MISS_OUT="$TMPDIR/miss_out.txt"
set +e
"$GEIF" -r "$MODEL" -a "$TMPDIR/partial_day.csv" -O 0.5 -e ';' -d 0 -M "MISS;%C;%a" > "$MISS_OUT"
RC=$?
set -e
if [ "$RC" -ne 2 ]; then
    echo "FAIL: Expected exit code 2 for missing category, got $RC"
    cat "$MISS_OUT"
    exit 1
fi
if ! grep -q "^MISS;12;Out;110;1100$" "$MISS_OUT"; then
    echo "FAIL: Expected MISS;12;Out;110;1100 in output"
    cat "$MISS_OUT"
    exit 1
fi
echo "  PASS: Missing category (-M) reported cleanly with -A."

echo "[5/5] Testing aggregated categorization (-c)..."
# Group "probe;A" sums to (250, 2500) -> should match "12;In"
# Group "probe;B" sums to (110, 1100) -> should match "12;Out"
cat << 'EOF' > "$TMPDIR/cat_day.csv"
Hour,Type,Inbytes,Outbytes
probe,A,100,1000
probe,A,150,1500
probe,B,50,500
probe,B,60,600
EOF

CAT_OUT="$TMPDIR/cat_out.txt"
"$GEIF" -r "$MODEL" -c "$TMPDIR/cat_day.csv" -e ';' -d 0 -p "CAT;%c->%C;%d" > "$CAT_OUT"
if ! grep -q "^CAT;probe;A->12;In;250;2500$" "$CAT_OUT"; then
    echo "FAIL: Expected CAT;probe;A->12;In;250;2500 in categorization output"
    cat "$CAT_OUT"
    exit 1
fi
if ! grep -q "^CAT;probe;B->12;Out;110;1100$" "$CAT_OUT"; then
    echo "FAIL: Expected CAT;probe;B->12;Out;110;1100 in categorization output"
    cat "$CAT_OUT"
    exit 1
fi
echo "  PASS: Aggregated categorization (-c) matched summed groups to best categories."

echo "=== Feature 13 (-A, --aggregate) Tests Passed! ==="
