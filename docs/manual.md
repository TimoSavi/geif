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
| `-R` | `INT` | Minimum row count required to train a sub-forest for a category. |
| `-D` | `INTERVAL`| Drop sub-forest categories older than specified age (e.g. `30d`, `7d`, `24h`, `3600s`). |
| `-N`, `--new` | `[TMPL]` | Output format template for newly observed, unseen categories during scoring or continuous learning (default: `"%v"`). |
| `-M`, `--missing` | `[TMPL]` | Output format template for missed categories (trained but absent in test stream; default: `"%C"`). |
| `-p` | `TMPL` | Printf-style format template for scored anomaly rows. |
| `-v` | `[TMPL]` | Format template for nominal inlier rows, or verbose flag. |
| `-d` | `INT` | Decimal precision for floating-point values (default: 6). |
| `-j` | `TMPL` | Per-dimension expansion template for `%m` vector breakdown. |
| `-S` | (none) | Silent mode: suppress nominal inlier rows and emit only detected anomalies. |
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

---

## 4. Output Formatting & Template Directives (`-p`, `-v`, `-N`, `-M`, `-j`)

The `-p` (outlier), `-v` (inlier), `-N` (unseen), and `-M` (missed) options accept formatting directives:

| Directive | Description |
| :--- | :--- |
| `%s` | Calibrated anomaly score (e.g. `0.8421`). |
| `%S` | Anomaly score percentage (e.g. `84.21%`). |
| `%l` | Concatenated label string extracted from `-L` columns. |
| `%c` | Category string from input observation (from `-C` columns). |
| `%C` | Model / assigned category string. |
| `%d` | Vector of raw numerical feature values (comma-separated), or outer stadium distance. |
| `%m` | Dimension attribution metrics expanded via `-j` template, or continuous metric depth $H(x)$. |
| `%e` | Single-dimension impact / attribution scores contributing to anomaly. |
| `%a` | Category dimension averages. |
| `%v` | Raw input line tokens joined by delimiter. |
| `%x` | Hex RGB color code corresponding to anomaly score (interpolated between low and high colors). |
| `%h` | Forest $H_{\text{max}}$ universal scale. |
| `%o` | Outlier flag (`1` for outlier, `0` for inlier). |
| `%n` | Total training rows seen by sub-forest. |
| `%t` | Human-readable timestamp (local time via `strftime("%c")`, matching CEIF). |
| `%u`, `%T` | Raw Unix epoch timestamp in seconds. |

### Example Template Usage:
```bash
./bin/geif -r model.json -a stream.csv -p "row=%l;score=%s;category=%c;rgb=%rgb"
./bin/geif -r model.json -c test.csv -p "%C -> %c (score: %s)"
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
