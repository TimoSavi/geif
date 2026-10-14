# GEIF: User Manual & CLI Reference Guide

**Author / Maintainer:** Timo Savinen (AI-assisted)

`geif` (Geometric Extended Isolation Forest) is a high-performance command-line utility for multi-dimensional anomaly detection, spatial clustering, and streaming outlier scoring.

---

## 1. Synopsis & Basic Invocations

```bash
geif [OPTIONS]...
```

Input data is expected to be character-delimited ASCII/UTF-8 records (default comma `,`). A different separator can be specified with `-f`. Delimiters (`-f` and `-e`) are automatically serialized into the model JSON (`input_separator` and `list_separator`) and restored when evaluating (`-r`) or continuously updating (`-z`) models without requiring `-f` or `-e` on subsequent CLI invocations.

### Core Operational Modes

1. **Training Mode (`-l`, `-w`)**:
   Ingest training records and serialize the trained model ensemble to a JSON file (or stdout):
   ```bash
   ./bin/geif -l train.csv -w model.json -B bubble -t 100 -s 256
   ```

2. **Scoring / Streaming Analysis Mode (`-r`, `-a`, `-o`)**:
   Load a trained model and score test observations in real time:
   ```bash
   ./bin/geif -r model.json -a test.csv -o scores.csv -O 0.50
   ```

3. **Categorization Mode (`-r`, `-c`)**:
   Classify input records against all trained sub-forest categories and assign the best-matching category:
   ```bash
   ./bin/geif -r model.json -c test.csv -o categorized.csv -O 0.45
   ```

4. **Population Drift & Test Grid Generation (`-T`, `-i`)**:
   Synthesize a uniform coordinate grid covering the sample domain for visualization:
   ```bash
   ./bin/geif -l train.csv -T 0.1 -i 150 -O 0 -p "%d,0x%x" -o grid.csv
   ```

5. **Model Diagnostics (`-r`, `-q`)**:
   Inspect ensemble structure, sub-forest categories, and calibrated scale parameters:
   ```bash
   ./bin/geif -r model.json -q
   ```

---

## 2. Command-Line Options Reference

| Option | Argument | Description |
| :--- | :--- | :--- |
| `-B`, `--algo` | `NAME` | Spatial isolation algorithm: `bubble` (default), `voronoi`, `exemplar`, `ceif`. |
| `-l` | `FILE` | Ingest CSV dataset for model training (use `-` for stdin). |
| `-a` | `FILE` | Score and analyze samples from input CSV (use `-` for stdin). |
| `-c` | `FILE` | Categorize observations against all sub-forest categories and assign best fit. |
| `-w` | `FILE` | Write trained model ensemble to a JSON file (use `-` for stdout). |
| `-r` | `FILE` | Read model ensemble from a JSON file (supports GEIF-1.0 and legacy formats; auto-restores `-f` and `-e`). |
| `-z` | `FILE` | Update model in-place: read model from `FILE`, update with streaming data, and rewrite to `FILE` (auto-restores `-f`). |
| `-o` | `FILE` | Output file path for scoring results (default: stdout, `-` for stdout). |
| `-O` | `THRESH` | Anomaly threshold: float in $[0.0, 1.0]$ (default 0.50), percentage (e.g. `90%`), or `average`. |
| `-t` | `INT` | Number of isolation trees per sub-forest (default: 100). |
| `-s` | `INT` | Number of samples subsampled per tree (default: 256). |
| `-i` | `INT` | Test grid sampling interval (default: 256) or tree count alias. |
| `-T` | `[FLOAT]` | Generate synthetic test grid with optional domain extension margin (e.g. `0.1`). |
| `-k` | (none) | Prune the most extreme outlier sample from the reservoir pool and recalibrate (repeatable). |
| `-g` | `FILE` | Load configuration / RC file (overrides `~/.geifrc` and `~/.ceifrc`). Repeatable. |
| `-f` | `CHAR` | Input field delimiter character (default: `,`). Persisted in model JSON and restored automatically. |
| `-e` | `CHAR` | Output field separator character (default: `,`). Persisted in model JSON and restored automatically. |
| `-H`, `--header` | (none) | Ignore header line in input CSV files. Persisted in model JSON and restored automatically. |
| `--no-header` | (none) | Explicitly disable header skipping, overriding saved model header state. |
| `-I` | `RANGE` | Comma-separated list of 1-based column indices/ranges to ignore (e.g. `1,3,5-7`). |
| `-U` | `RANGE` | Comma-separated list of 1-based column indices/ranges to include exclusively. |
| `-L` | `RANGE` | Column indices/ranges to treat as descriptive labels (excluded from features). |
| `-C` | `RANGE` | Column indices/ranges to treat as category keys for multi-category routing. |
| `-F` | `REGEXP` | Filter out categories matching regular expression (prefix with `-v ` to invert / keep only). |
| `-R` | `INT` | Minimum row count required to retain a category across training, analysis (`-a`), categorization (`-c`), and maintenance. |
| `-D` | `INTERVAL`| Drop sub-forest categories older than specified age (e.g. `30d`, `7d`, `24h`, `3600s`). |
| `-A`, `--aggregate` | (none) | Instead of taking samples as individual rows, sum/aggregate new sample values per category across each `-l`, `-a`, or `-c` invocation. Only one aggregated sample per category is added per `-l` session. Persisted in model JSON and restored automatically. |
| `-N`, `--new` | `[TMPL]` | Output format template for newly observed, unseen categories during scoring or continuous learning (default: `"%v"`). Evaluated immediately for any new category sample regardless of sample counts. |
| `-M`, `--missing` | `[TMPL]` | Output format template for missed categories (trained but absent in test stream; default: `"%C"`). |
| `-p` | `TMPL` | Printf-style format template for scored anomaly rows or categorized data. In analysis mode (`-a`), only outlier rows (`score > threshold`) are printed; nominal inlier rows are suppressed. Default is `"%s %v"`. |
| `-v` | `[TMPL]` | Print average score and other summary statistics calculated from analyzed data using format `TMPL` (default: `"%C %r %h"`). When specified without an argument, enables verbose mode. |
| `-m` | `STRING` | Printf format for printing floating-point values for samples and sample averages (default: `"%.*f"`). |
| `-d` | `INT` | Decimal precision for floating-point values (default: 6). Persisted in model JSON and restored automatically. |
| `-j` | `TMPL` | Printf format for printing all single dimension attribute metrics together via directive `%m`. May contain directives `%d`, `%a`, `%e`, and `%i`. |
| `-S` | (none) | Silent mode: suppress nominal inlier rows and emit only detected anomalies in default output mode. |
| `-q` | (none) | Print model summary diagnostics and exit immediately. |
| `-h` | (none) | Display help summary and exit. |

---

## 3. Anomaly Scoring Semantics & Thresholds (`-O`)

GEIF scales all anomaly scores to the calibrated range $[0.0, 1.0]$ by default:
- **Zero Kelvin Calibration**: Dense cluster centroids naturally achieve scores near $0.00$.
- **Asymptotic Outer Space**: Points infinitely far outside the data domain smoothly approach $1.00$ without boundary overflow.

### Threshold Specifications:
1. **Fixed Numerical Threshold (`-O 0.50`)**:
   Points with score $\ge 0.50$ are flagged as anomalies.
2. **Percentile Distribution Threshold (`-O 94%`)**:
   Automatically calculates the empirical score cutoff from the training sample distribution such that 94% of nominal samples fall below the threshold.
3. **Average Score Threshold (`-O average`)**:
   Sets the anomaly cutoff to the ensemble mean score.

All input formatting and aggregation options (`-f`, `-e`, `-H`, `-A`, `-d`, `-C`, `-L`, `-I`, `-U`, `-O`) are persisted into the JSON model file during training (`-w`) or update (`-z`). When reading models (`-r` or `-z`), these settings are automatically restored and do not need to be re-specified.

---

## 4. Output Formatting & Template Directives (`-p`, `-v`, `-N`, `-M`, `-j`)

The `-p` (outlier), `-v` (summary), `-N` (unseen), and `-M` (missed) options accept printf-style formatting directives:

| Directive | Meaning |
| :--- | :--- |
| `%r` | Current input file row number. |
| `%s` | Calibrated anomaly score (e.g. `0.842100`). |
| `%S` | Average anomaly score for analyzed data. |
| `%n` | Number of training rows for a forest. |
| `%o` | Number of analyzed rows for a forest in summary mode (`-v`), or outlier flag (`1` / `0`) in row mode. |
| `%h` | Number of analyzed rows having score greater than outlier score threshold in summary mode (`-v`), or forest $H_{\text{max}}$ scale. |
| `%c` | Category string from input observation (from `-C` columns). The original category when categorizing data. |
| `%C` | Forest category string. The best-matching category when categorizing data (`-c`). |
| `%l` | Concatenated label values extracted from `-L` columns. |
| `%d` | Separated list of dimension values, or outer stadium distance. |
| `%u`, `%T` | Raw Unix epoch timestamp in seconds. |
| `%a` | Separated list of category dimension average values. |
| `%e` | Separated list of dimension attribute scores (how each attribute affects the total score). |
| `%m` | Separated list of dimension metrics printed attribute by attribute using format given by `-j`. Emits empty string if `-j` is omitted. |
| `%i` | Dimension attribute index (1-based, inside `-j` template). |
| `%v` | Current input row values joined by delimiter. |
| `%x` | Outlier score formatted as an RGB hex value (e.g. `FE0000`). |
| `%t` | Timestamp when category was last updated, in human-readable form using current locale (`strftime("%c")`). |
| `%: ` | Category value separator (default: `;`). |
| `%. ` | Label value separator (default: `-`). |
| `%%` | Literal percent sign. |

### Example Template Usage:
```bash
# Score observations, printing only anomalies exceeding 0.50 with category averages
./bin/geif -r wine.json -a wine_test.csv -O0.5 -p "A %s %C %m %a"

# Dimension expansion using -j and %m
./bin/geif -r model.json -a test.csv -j "%i:val=%d,imp=%e" -p "%l => %m"

# Print category summary statistics at conclusion of batch analysis
./bin/geif -r model.json -a batch.csv -v "SUMMARY: %C rows=%o outliers=%h avg_score=%S"
```

### Data Value Aggregation (`-A`, `--aggregate`):
When analyzing daily or hourly summaries (such as network traffic or log volume per category), invoke `geif` with `-A` so that each sub-forest represents an aggregation key and sums all rows per category within each invocation:

```bash
# Day 1: Train initial aggregated model using columns 1-2 as category key (-C 1-2)
./bin/geif -l day1_traffic.csv -C 1-2 -H -A -d 0 -w traffic.json

# Subsequent days: Update model in-place (-A, -C, -H, -d are restored automatically)
./bin/geif -z traffic.json -l day2_traffic.csv

# Analyze a new day's traffic by summing values per category and scoring the daily totals
./bin/geif -r traffic.json -a today_traffic.csv -O 0.5 -e ';' -p "ANOM;%s;%C;%d;%a" -N "NEW;%c;%d"
```

For comprehensive guidance on engine selection (`bubble`, `ceif`, `exemplar`, `voronoi`), visual heatmaps, population drift coverage, categorization accuracy, and throughput benchmarks, see [**`docs/algorithm_selection.md`**](algorithm_selection.md).

---

## 5. Cascading Configuration File (`~/.geifrc`)

GEIF automatically loads user defaults from `~/.geifrc` (falling back to legacy `~/.ceifrc`). Custom files can be passed using `-g <file>`.

### Sample `~/.geifrc`:
```ini
# Core parameters
ALGO bubble
TREES 100
MAX_SAMPLES 256
OUTLIER_SCORE 0.50
HEADER true

# Output preferences
DECIMALS 4
LOW_RGB_COLOR 4DF64D
HIGH_RGB_COLOR F25DF2
PRINT_DIMENSION 1
```

---

## 6. Return Code Contract

In adherence with standard UNIX pipeline conventions:
- **0**: Clean success; all evaluated observations are nominal inliers (score $< \text{threshold}$).
- **2**: Outliers detected; at least one observation equaled or exceeded the anomaly threshold.
- **1**: Error encountered (file not found, corrupted model JSON, invalid options).

---

## 7. Model Migration & Algorithm Conversion (`ceif2geif`)

The `ceif2geif` utility converts legacy CEIF models or existing GEIF models into `GEIF-1.0` JSON models, optionally changing the algorithm (`-B`) or overriding tree/sample counts (`-t`, `-s`):

```bash
# Convert a CEIF model to GEIF-1.0 format while changing algorithm to 'bubble'
./bin/ceif2geif -v -B bubble legacy_model.json geif_model.json

# Convert and override algorithm, tree count, and sample count
./bin/ceif2geif -B voronoi -t 150 -s 256 legacy_model.json -o geif_voronoi.json
```

| Flag | Long Option | Description |
| :--- | :--- | :--- |
| **`-o <file>`** | `--output <file>` | Output GEIF model JSON file (default: stdout or 2nd positional argument). |
| **`-B <name>`** | `--algo <name>` | Change algorithm during conversion (`bubble` \| `voronoi` \| `exemplar` \| `ceif`). |
| **`-t <int>`** | `--trees <int>` | Override tree count in forest (default: from input model or `100`). |
| **`-s <int>`** | `--samples <int>` | Override samples per tree (default: from input model or `256`). |
| **`-v`** | `--verbose` | Print migration details and sub-forest statistics. |
| **`-q`** | `--quiet` | Suppress all non-error output. |

