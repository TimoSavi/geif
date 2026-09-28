# GEIF — Geometric Extended Isolation Forest

**Author / Maintainer:** Timo Savinen (AI-assisted)

> [!NOTE]
> **AI-Assisted Development:** GEIF was designed and implemented by Timo Savinen with AI pair-programming assistance (Google Antigravity / Gemini). The mathematical specifications, C17 core library, CLI Unix pipelines, test suite, and architectural documentation were developed collaboratively under human engineering oversight.

**GEIF** (Geometric Extended Isolation Forest) is a high-performance C17 implementation of a geometric, scale-invariant anomaly detection algorithm. Building upon and refining the concepts of Isolation Forest (iForest) and Extended Isolation Forest (EIF / CEIF), GEIF introduces data-adaptive Voronoi bisector hyperplanes, continuous Euclidean envelope distance ("Stadium" metric), and universal scale calibration.

---

## Key Innovations

### 1. Data-Adaptive Voronoi Bisectors
Standard Isolation Forests make axis-aligned cuts, creating severe rectangular artifacts. Standard EIF draws arbitrary random normal vectors from uniform spherical distributions, which fail when dimensions have radically different physical scales (e.g. coordinates with a 5000:1 aspect ratio).

GEIF solves this by choosing two distinct points $A$ and $B$ from each node's subsample and placing the splitting hyperplane at their midpoint:
$$P_{\text{mid}} = \frac{A + B}{2}, \quad \vec{n} = \frac{B - A}{\|B - A\|}$$
- **Scale Invariant**: Adapts intrinsically to the local geometry of the data without requiring manual normalization or feature scaling.
- **Degenerate Feature Masking**: Automatically detects zero-variance features and isolates them safely.

### 2. Smooth Euclidean Stadium Metric ("Outer Space")
When an observation falls outside the bounding box observed during training:
- Conventional tree structures fail to differentiate between a point slightly outside the box and a point millions of units away in outer space.
- GEIF computes the exact Euclidean distance $d_{\text{out}}$ to the training envelope bounding box, effectively creating rounded "stadium" equi-distance shells.
- As $d_{\text{out}} \to \infty$, the anomaly score smoothly and continuously converges to $1.000000$.

### 3. "Zero Kelvin" Universal Scale Calibration
Anomaly scores in GEIF are normalized in the range $[0.0, 1.0]$ with intuitive semantics:
- **0.000000 ("Zero Kelvin")**: The theoretical absolute inlier — approachable asymptotically as sample density and depth increase, but never exceeded.
- **0.10 – 0.35**: Nominal cluster inliers.
- **0.50**: Default decision boundary threshold.
- **0.85 – 1.00**: Anomalies and points far outside the training distribution.

### 4. Streaming Reservoir Sampling
The training pool maintains a fixed maximum capacity:
$$N_{\text{pool}} = N_{\text{trees}} \times N_{\text{samples}} \quad (\text{e.g., } 100 \times 256 = 25{,}600)$$
Any streaming input of arbitrary length is ingested via Algorithm R reservoir sampling, guaranteeing an unbiased uniform sample even if the input stream is sorted or clustered.

---

## Directory Structure

```
geif/
├── include/
│   └── geif/
│       ├── geif.h      # Public C17 API
│       ├── types.h     # Internal structs & configuration
│       └── error.h     # Status codes and error reporting
├── src/
│   ├── lib/            # Library implementation
│   │   ├── forest.c    # Allocation & lifecycle
│   │   ├── reservoir.c # Streaming reservoir sampling
│   │   ├── train.c     # Voronoi tree builder & calibration
│   │   ├── evaluate.c  # Continuous metric depth & stadium distance
│   │   ├── json_io.c   # Model serialization (json-c)
│   │   ├── error.c     # Status strings
│   │   └── geometry.h  # Geometric distance & Voronoi math
│   └── cli/
│       └── main.c      # Command-line utility (geif)
├── test/
│   ├── test_voronoi.c  # Bisector & aspect ratio unit tests
│   └── test_stadium.c  # Stadium Euclidean distance unit tests
├── Makefile            # High-performance C17 build system
└── .gitignore
```

---

## Building

### Requirements
- A modern C compiler supporting **C17** (`gcc` $\ge 9$ or `clang` $\ge 10$)
- Standard C runtime and math library (`-lm`)
- `json-c` development libraries (`libjson-c-dev` on Debian/Ubuntu, `json-c-devel` on RHEL/Rocky)

### Build Targets

```bash
# Build static library, shared library, and CLI binary
make all

# Run unit test suite
make test

# Clean build artifacts
make clean
```

The build produces:
- `lib/libgeif.a` (Static Library)
- `lib/libgeif.so` (Shared Library)
- `bin/geif` (CLI Binary)

---

## CLI Usage

### 1. Train a Model
Train a GEIF forest from a CSV dataset and save the model:
```bash
./bin/geif -l dataset.csv -w model.json -i 100 -s 256 -v
```
- `-l <file>`: Input CSV file for training.
- `-w <file>`: Output path to save the JSON model.
- `-i <num>`: Number of trees (default: 100).
- `-s <num>`: Samples per tree $\psi$ (default: 256).

### 2. Inspect a Model
View summary statistics and calibration metrics:
```bash
./bin/geif -r model.json -q
```

Output:
```
GEIF Forest Summary:
  Dimensions:          2 (Active: 2, Inactive/Constant: 0)
  Trees:               100
  Samples/Tree (psi):  256
  Max Depth Cap:       16
  Sample Pool Count:   1465 / 25600 (Stream rows seen: 1465)
  Nominal Spacing:     0.353553
  H_train_max:         29.029803
  H_max (Scale):       36.287253 (Headroom: 1.25)
```

### 3. Score Streaming Data
Score queries from a CSV file or `stdin`:
```bash
./bin/geif -r model.json -a test.csv -o scores.csv -T 0.5 -v
```
Or stream through pipes:
```bash
cat new_samples.csv | ./bin/geif -r model.json -a - -T 0.5
```

Each output row appends:
`<original_features>,<anomaly_score>,<is_outlier_flag>,<metric_depth>,<stadium_distance>`

---

## C API Example

```c
#include <geif/geif.h>
#include <stdio.h>

int main(void) {
    geif_forest_t *forest = NULL;
    geif_config_t config = geif_config_default();

    // 1. Create a 2D forest
    geif_forest_create(&forest, 2, &config);

    // 2. Feed training points
    double p1[2] = {10.0, 20.0};
    double p2[2] = {10.5, 20.2};
    geif_forest_feed(forest, p1);
    geif_forest_feed(forest, p2);

    // 3. Train forest & calibrate scale
    geif_forest_train(forest);

    // 4. Score an observation
    double query[2] = {500.0, 500.0};
    double score = 0.0;
    geif_forest_score(forest, query, &score);

    printf("Anomaly score: %.6f\n", score);

    // 5. Cleanup
    geif_forest_destroy(forest);
    return 0;
}
```

---

## Development & Attribution
**GEIF** is an AI-assisted systems engineering project developed by Timo Savinen in pair-programming collaboration with Google Antigravity (Gemini).

---

## License
MIT License.
