# Compatibility Matrix: GEIF vs. CEIF

**Author / Maintainer:** Timo Savinen (AI-assisted)

This document analyzes the compatibility between **GEIF (Geometric Extended Isolation Forest)** and **CEIF (Categorized Extended Isolation Forest)** across CLI options, configuration files (`rcfile`), and serialized tree/model data.

---

## 1. Executive Summary

| Subsystem | Compatibility Status | Notes |
|---|---|---|
| **Core Workflow Options** | **Partially Compatible** | Core workflow flags (`-l`, `-a`, `-w`, `-r`, `-o`, `-s`, `-q`, `-v`) share identical semantics. Parameter tuning flags have divergent names (e.g., `-i` vs `-t` for tree counts, `-T` vs `-O` for threshold). |
| **Advanced CLI Options** | **Not Compatible** | CEIF features like multi-category partitioning (`-C`, `-c`, `-F`), output templating (`-p`), column masks (`-I`, `-U`), and text hashing (`-X`) are not currently implemented in the minimalist `geif` CLI. |
| **RC File (`~/.ceifrc`)** | **Not Compatible** | `geif` does not currently parse `~/.ceifrc` or support an rc file. Several CEIF directives (`AUTO_SCALE`, `NEAREST`) are mathematically obsolete in GEIF. |
| **Model / Tree Data File** | **Format Incompatible** | CEIF saves raw reservoir samples in a proprietary JSON/CSV schema and rebuilds trees randomly on load. GEIF serializes the complete trained tree structures, normals pool, Voronoi cut hyperplanes, and calibrated scale parameters (`GEIF-1.0` JSON). |

---

## 2. CLI Options Comparison

### 2.1 Core Options

| Option Purpose | `ceif` Syntax | `geif` Syntax | Compatibility / Differences |
|---|---|---|---|
| **Learn / Train** | `-l <file>` | `-l <file>` | **Identical**: Both ingest CSV rows (or `-` for stdin). |
| **Analyze / Score** | `-a <file>` | `-a <file>` | **Identical**: Evaluates rows against the model. |
| **Save Model** | `-w <file>` | `-w <file>` | **Semantics match**, but file contents differ (see Section 4). |
| **Load Model** | `-r <file>` | `-r <file>` | **Semantics match**, but file formats differ. |
| **Output Destination** | `-o <file>` | `-o <file>` | **Identical**: Specifies output file (defaults to stdout). |
| **Samples per Tree ($\psi$)** | `-s <int>` | `-s <int>` | **Identical**: Both default to `256`. |
| **Tree Count ($T$)** | `-t <int>` | `-i <int>` | **Flag divergence**: `ceif` uses `-t` (`--trees`), while `geif` uses `-i` (`--iterations/trees`). |
| **Anomaly Threshold** | `-O <float>` | `-T <float>` | **Flag divergence**: `ceif` uses `-O` (`--outlier-score`), while `geif` uses `-T` (`--threshold`). |
| **Inspect Model Info** | `-q` | `-q` | **Identical**: Prints summary diagnostics and exits. |
| **Verbose Mode** | `-v` | `-v` | **Similar**: Both output progress and scoring statistics. |

### 2.2 CEIF Flags Not Currently in GEIF

CEIF evolved over many iterations to include database-like preprocessing:
- **Categorization**: `-C` (category column), `-c` (categorize query), `-F` (category regex filter), `-R` (reset category).
- **Column Filtering**: `-I` (ignore columns), `-U` (include columns), `-L` (label columns), `-X` (text hashing).
- **Output Templating**: `-p "%d,0x%x,%s"` (custom printf format strings with RGB hex codes).
- **In-place updates**: `-z` (rolling updates).
- **Grid Generation**: `-T <margin>` (in `ceif`, `-T` generates synthetic test grids).

---

## 3. Configuration File (`~/.ceifrc`)

### 3.1 CEIF RC File Handling
CEIF automatically reads `~/.ceifrc` (or a custom path via `-g <file>`) to configure global defaults:
- `TREES`, `SAMPLES`, `DECIMALS`
- `AUTO_SCALE` (enabled by default to normalize large aspect ratio disparities)
- `NEAREST` (enabled by default to evaluate k-NN leaf distance)
- `OUTLIER_SCORE` (sets default threshold, e.g. `0.5s` or `97%`)
- `CATEGORY_SEPARATOR`, `LABEL_SEPARATOR`
- `LOW_RGB_COLOR`, `HIGH_RGB_COLOR`

### 3.2 Current GEIF Status
1. `geif` **does not read any rc file** at startup. It relies purely on CLI flags or C API struct configuration (`geif_config_default()`).
2. Several CEIF directives are **no longer needed** due to GEIF's geometric mathematical design:
   - `AUTO_SCALE`: GEIF uses Voronoi midpoint bisectors ($B - A$), which are intrinsically scale-invariant across 5000:1 aspect ratios.
   - `NEAREST`: GEIF incorporates Cauchy-Lorentz residual cell damping directly into tree inference without post-hoc k-NN search.
   - `OUTLIER_SCORE` scale suffixes (`s`, `%`): GEIF uses universal Zero Kelvin scale calibration where $0.50$ is already the standardized boundary.

---

## 4. Tree Data & Serialization Format

### 4.1 What CEIF Saves (`.ceif`)
CEIF does **not** persist tree structures. Its JSON format stores:
```json
{
  "GLOBALS": {
    "dimensions": 2,
    "trees": 100,
    "samples": 256,
    "decimals": 6
  },
  "FORESTS": [
    {
      "category": "default",
      "sample_count": 256,
      "last_updated": 1727341200,
      "extra_rows": 0,
      "SAMPLES": [
        [10.5, 20.2],
        [12.1, 19.8]
      ]
    }
  ]
}
```
* **Implication**: When loading with `ceif -r`, CEIF re-runs the randomized spherical tree-building algorithm from the saved reservoir samples. Two different runs from the same file yield slightly different trees.

### 4.2 What GEIF Saves (`.json`)
GEIF persists the **exact, pre-computed ensemble of geometric trees** (`GEIF-1.0`):
```json
{
  "format": "GEIF-1.0",
  "dimensions": 2,
  "tree_count": 100,
  "samples_per_tree": 256,
  "max_depth": 16,
  "kappa": 1.25,
  "H_train_max": 29.0298,
  "H_max": 36.2872,
  "delta_nominal": 0.3535,
  "envelope_min": [1000.0, 0.2],
  "envelope_max": [9000.0, 1.8],
  "effective_span": [8000.0, 1.6],
  "dim_active": [1, 1],
  "pool_count": 1465,
  "sample_pool": [...],
  "trees": [
    {
      "node_count": 511,
      "nodes": [
        {"left": 1, "right": 2, "offset": 0, "pdotn": 14.52, "weight": 0.85, "delta": 1.17, "samples": 256, "leaf_idx": -1}
      ],
      "normals": [...]
    }
  ]
}
```
* **Implication**: `geif -r` performs zero retraining on load. Inference is instantaneous and 100% deterministic.

---

## 5. Migration & Compatibility Path

If compatibility with existing CEIF workflows or datasets is desired, GEIF can be extended with:
1. **Option Alias Flag**: Add `-t` as an alias for `-i` (trees), and `-O` as an alias for `-T` (threshold).
2. **CEIF Model Importer**: A function `geif_forest_load_ceif_json()` that reads CEIF's `SAMPLES` array, feeds them into GEIF's reservoir pool, and trains the Voronoi forest.
3. **RC File Support**: A lightweight parser for `~/.geifrc` (or `~/.ceifrc`) reading tree counts, sample limits, and default thresholds.
