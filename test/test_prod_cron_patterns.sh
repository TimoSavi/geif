#!/usr/bin/env bash
#
# test_prod_cron_patterns.sh:
# End-to-End Production Cron Suite testing all usage patterns from plan.md
# against real-world datasets: winequality-red.csv & paddydataset.csv.
#
# Author / Maintainer: Timo Savinen (AI-assisted)
#

set -euo pipefail

GEIF="./bin/geif"
CEIF2GEIF="./bin/ceif2geif"
DATA_DIR="./test/data"
WINE_CSV="$DATA_DIR/winequality-red.csv"
PADDY_CSV="$DATA_DIR/paddydataset.csv"

TEST_DIR=$(mktemp -d /tmp/geif_test_prod_XXXXXX)
trap 'rm -rf "$TEST_DIR"' EXIT

echo "======================================================================"
echo "=== Running GEIF End-to-End Production Cron Suite (plan.md) ========"
echo "======================================================================"

if [ ! -f "$WINE_CSV" ] || [ ! -f "$PADDY_CSV" ]; then
    echo "ERROR: Test datasets missing in $DATA_DIR"
    exit 1
fi

# Create realistic ~/.ceifrc configuration file as described in plan.md
RC_FILE="$TEST_DIR/.ceifrc"
cat << 'EOF' > "$RC_FILE"
TREES 50
MAX_SAMPLES 256
OUTLIER_SCORE 55%
ANALYZE_SAMPLING 1000
LOW_RGB_COLOR 0x20FF20
HIGH_RGB_COLOR 0xFF0000
DECIMALS 4
PRINT_DIMENSION "%d<dim>%e"
EOF

echo "Using temporary RC configuration: $RC_FILE"
cat "$RC_FILE"
echo ""

# ====================================================================
# PART 1: Single-Column Category Workflows (winequality-red.csv)
# ====================================================================
echo "----------------------------------------------------------------------"
echo "Part 1: Production Workflows on Single-Category Dataset (Wine Quality)"
echo "----------------------------------------------------------------------"

WINE_CONF="$TEST_DIR/wine_model.json"

# Pattern 1 (plan.md line 10):
# grep "$regex" $YEAR | ceif -l - -L 1 -C "2-10" -m "%'.0f" -e ';' -d 0 -w $ceiffile
# Adapted for wine dataset: train from stdin with label dim 1, category dim 12, delim ';'
echo "Test 1: Streaming training from stdin with -g RC, -L, -C, -e ';', -d 0 (plan.md line 10)..."
tail -n +2 "$WINE_CSV" | "$GEIF" -g "$RC_FILE" -l - -L 1 -C 12 -m "%'.0f" -e ';' -d 0 -w "$WINE_CONF"

if [ ! -s "$WINE_CONF" ]; then
    echo "ERROR: Failed to train wine model"
    exit 1
fi
echo "  [PASS] Trained wine model saved to $WINE_CONF"

# Pattern 2 (plan.md line 11):
# grep "$regex" $YESTERDAY | ceif -r $ceiffile -g ~/.ceifrc -a - -S -F"-v ^$yvalue" -M"%t;;%C;%m" -N"NEW;%l;%c;%m" -p "%s;%l;%c;%m" | sort -nr
echo "Test 2: Stream analysis with -g RC, -S, -F '-v ^5', -M, -N, -p (plan.md line 11)..."
WINE_STREAM_OUT="$TEST_DIR/wine_stream.txt"
tail -n 100 "$WINE_CSV" | ( "$GEIF" -r "$WINE_CONF" -g "$RC_FILE" -a - -e ';' -S -F "-v ^5" \
    -M "%t;;%C;%m" -N "NEW;%l;%c;%m" -p "%s;%l;%c;%m" || [ $? -eq 2 ] ) | sort -nr > "$WINE_STREAM_OUT"

if [ ! -s "$WINE_STREAM_OUT" ]; then
    echo "ERROR: Stream output is empty"
    exit 1
fi
echo "  [PASS] Stream analysis generated $(wc -l < "$WINE_STREAM_OUT") lines. Top line: $(head -n 1 "$WINE_STREAM_OUT")"

# Pattern 3 (plan.md line 13):
# grep "$regex" $YESTERDAY | ceif -r $ceiffile -g ~/.ceifrc -l - -D30d -w $ceiffile
echo "Test 3: Streaming model update with age decay -D30d (plan.md line 13)..."
PRE_UPDATE_SUMMARY=$("$GEIF" -r "$WINE_CONF" -q)
tail -n 50 "$WINE_CSV" | "$GEIF" -r "$WINE_CONF" -g "$RC_FILE" -l - -e ';' -D30d -w "$WINE_CONF"
POST_UPDATE_SUMMARY=$("$GEIF" -r "$WINE_CONF" -q)
echo "  [PASS] Model updated with decay -D30d successfully."

# Pattern 4 (plan.md line 14 & 19):
# ceif -r $conf -k -w $conf
echo "Test 4: Reservoir outlier pruning and recalibration -k (plan.md line 14/19)..."
"$GEIF" -r "$WINE_CONF" -k -w "$WINE_CONF"
echo "  [PASS] Recalibrated model with -k successfully."

# Pattern 5 (plan.md line 15):
# grep "$regex" $YEAR | ceif -l - -L 1 -p "%s;%l;%c;%d;%a" -C "2-10" -m "%'.0f" -e ';' -W -d 0 -Oaverage -w $ceiffile
echo "Test 5: Training with -Oaverage, -W, dual formatting (plan.md line 15)..."
WINE_CONF_AVG="$TEST_DIR/wine_avg.json"
tail -n +2 "$WINE_CSV" | "$GEIF" -l - -L 1 -C 12 -m "%'.0f" -e ';' -W -d 0 -Oaverage -w "$WINE_CONF_AVG"
if ! grep -q '"outlier_score":"average"' "$WINE_CONF_AVG"; then
    echo "ERROR: -Oaverage not saved in model"
    exit 1
fi
echo "  [PASS] Model trained with -Oaverage successfully."

# Pattern 6 (plan.md line 30):
# $CEIF -r $CONF -g $CEIFCONF -a $DATAFILE "-e;" -v "K;%s;%S;%C;%o;%h;%a" > $MAILFILE
echo "Test 6: Inlier template output with -v 'K;%s;%S;%C;%o;%h;%a' (plan.md line 30)..."
MAILFILE="$TEST_DIR/wine_mail.txt"
( "$GEIF" -r "$WINE_CONF" -g "$RC_FILE" -a "$WINE_CSV" -e ';' -H -v "K;%s;%S;%C;%o;%h;%a" || [ $? -eq 2 ] ) > "$MAILFILE"
if [ ! -s "$MAILFILE" ]; then
    echo "ERROR: Mailfile is empty"
    exit 1
fi
echo "  [PASS] Inlier template output generated $(wc -l < "$MAILFILE") lines. Sample: $(head -n 1 "$MAILFILE")"

# Pattern 7 (plan.md line 31):
# $CEIF -r $CONF -g $CEIFCONF -T0.1 -e, -d4 -p "%d,0x%x" -F "-v ^${n}$" | grep -v -- - > $PLOTDATA
echo "Test 7: Population drift test grid generation -T0.1 (plan.md line 31)..."
PLOTDATA="$TEST_DIR/wine_plot.dat"
( "$GEIF" -r "$WINE_CONF" -g "$RC_FILE" -T0.1 -i 2 -e, -d4 -p "%d,0x%x" -F "-v ^5$" || [ $? -eq 2 ] ) | grep -v -- - > "$PLOTDATA" || true
if [ ! -s "$PLOTDATA" ]; then
    echo "ERROR: Plot data is empty"
    exit 1
fi
echo "  [PASS] Population drift test grid generated $(wc -l < "$PLOTDATA") points for category 5."

# ====================================================================
# PART 2: Multi-Column Category Workflows (paddydataset.csv)
# ====================================================================
echo "----------------------------------------------------------------------"
echo "Part 2: Production Workflows on Multi-Category Dataset (Paddy Dataset)"
echo "----------------------------------------------------------------------"

PADDY_CONF="$TEST_DIR/paddy_model.json"

# Pattern 8 (plan.md line 20):
# $CEIF -g $CEIFCONF -l $DATAFILE -C1-$CATFIELDS -A -d0 -H -p "A;%s;%C;%m;%a" "-e;" -S -m "%'.0f" -w $CONF
echo "Test 8: Multi-column category training with RC file: -C 2-4 -A -d0 -H -e, -S -w (plan.md line 20)..."
"$GEIF" -g "$RC_FILE" -l "$PADDY_CSV" -C 2-4 -A -d 0 -H -p "A;%s;%C;%m;%a" -e, -S -m "%'.0f" -w "$PADDY_CONF"

if [ ! -s "$PADDY_CONF" ]; then
    echo "ERROR: Paddy model file missing or empty"
    exit 1
fi
PADDY_SUMMARY=$("$GEIF" -r "$PADDY_CONF" -q)
echo "$PADDY_SUMMARY" | grep "Sub-forests"
echo "  [PASS] Multi-category paddy model trained with $(echo "$PADDY_SUMMARY" | grep "Sub-forests" | awk '{print $2}') categories."

# Pattern 9 (plan.md line 21):
# $CEIF -r $CONF -g $CEIFCONF -a $DATAFILE -F"-v ^$HOUR" -S -M"M;%t;%C;%a" -N"N;NEW;%c;%d" "-e;" -p "A;%s;%C;%m;%a"
echo "Test 9: Multi-category analysis with filtering: -F '-v ^Panruti' (plan.md line 21)..."
PADDY_OUT="$TEST_DIR/paddy_out.txt"
( "$GEIF" -r "$PADDY_CONF" -g "$RC_FILE" -a "$PADDY_CSV" -H -e, -F "-v ^Panruti" -S \
    -M "M;%t;%C;%a" -N "N;NEW;%c;%d" -p "A;%s;%C;%m;%a" || [ $? -eq 2 ] ) > "$PADDY_OUT"

if [ ! -s "$PADDY_OUT" ]; then
    echo "ERROR: Paddy analysis output is empty"
    exit 1
fi
echo "  [PASS] Multi-category analysis produced $(wc -l < "$PADDY_OUT") results."

# Pattern 10 (plan.md line 22 & 27):
# [ "$CTL" = "NOUPDATE" ] || $CEIF -g $CEIFCONF -r $CONF -g $CEIFCONF -l $DATAFILE -D7d -w $CONF
echo "Test 10: Conditional update with decay: -D7d (plan.md line 22/27)..."
"$GEIF" -g "$RC_FILE" -r "$PADDY_CONF" -g "$RC_FILE" -l "$PADDY_CSV" -H -e, -D7d -w "$PADDY_CONF"
echo "  [PASS] Conditional update with -D7d completed successfully."

# Pattern 11 (plan.md line 23 & 25):
# $CEIF -l $DATAFILE -C1-$CATFIELDS -R100 -W -A -d0 -H -O0.6 -p "A;%s;%C;%d;%a" "-e;" -S -m "%'.0f" -w $CONF
echo "Test 11: Training with timestamp touch -R100, -O0.6, -W (plan.md line 23/25)..."
PADDY_TOUCH_CONF="$TEST_DIR/paddy_touch.json"
"$GEIF" -l "$PADDY_CSV" -C 2-4 -R 100 -W -A -d 0 -H -O 0.6 -p "A;%s;%C;%d;%a" -e, -S -m "%'.0f" -w "$PADDY_TOUCH_CONF"
echo "  [PASS] Training with -R100 and -O0.6 completed successfully."

# Pattern 12 (plan.md line 26):
# $CEIF -r $CONF -a $DATAFILE -k -F"-v ^$HOUR" -S -M"M;%t;%C;%a" -N"N;NEW;%c;%d" "-e;" -W
echo "Test 12: Analysis with in-line outlier pruning -k (plan.md line 26)..."
( "$GEIF" -r "$PADDY_CONF" -a "$PADDY_CSV" -k -F "-v ^Panruti" -S -M "M;%t;%C;%a" -e, -W || [ $? -eq 2 ] ) > /dev/null
echo "  [PASS] Analysis with -k executed cleanly."

# Pattern 13 (plan.md line 32):
# [ "$CTL" = "NOUPDATE" ] || $CEIF -r $CONF -g $CEIFCONF -l $DATAFILE -D9d -w $CONF
echo "Test 13: Final model update with decay -D9d (plan.md line 32)..."
"$GEIF" -r "$PADDY_CONF" -g "$RC_FILE" -l "$PADDY_CSV" -H -e, -D9d -w "$PADDY_CONF"
echo "  [PASS] Final update with -D9d completed successfully."

# ====================================================================
# PART 3: CEIF to GEIF Full Migration with Production Datasets
# ====================================================================
echo "----------------------------------------------------------------------"
echo "Part 3: CEIF to GEIF Full Migration Pipeline (ceif2geif)"
echo "----------------------------------------------------------------------"

# Convert a CEIF-format model to GEIF, then score paddy data against it
echo "Test 14: Migrating model and executing production query..."
PADDY_MIGRATED="$TEST_DIR/paddy_migrated.json"
"$CEIF2GEIF" -v -t 50 -s 64 "$PADDY_CONF" "$PADDY_MIGRATED" 2> /dev/null

MIG_OUT="$TEST_DIR/mig_out.txt"
( "$GEIF" -r "$PADDY_MIGRATED" -a "$PADDY_CSV" -H -e, -S -p "%s;%C;%o" || [ $? -eq 2 ] ) | head -n 30 > "$MIG_OUT"

if [ ! -s "$MIG_OUT" ]; then
    echo "ERROR: Migrated scoring output is empty"
    exit 1
fi
echo "  [PASS] Migrated model scored 30 rows cleanly. Sample: $(head -n 1 "$MIG_OUT")"

echo ""
echo "======================================================================"
echo "=== All 14 Production Cron Patterns Passed Successfully! ============"
echo "======================================================================"
