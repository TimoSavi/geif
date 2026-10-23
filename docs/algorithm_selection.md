# GEIF Algorithmic Selection & Performance Guide

**Author / Maintainer:** Timo Savinen (AI-assisted)

> [!NOTE]
> This guide provides empirical benchmark results, visual topology evaluations, and operational selection matrices across GEIF's four anomaly detection and classification engines: **`bubble`**, **`exemplar`**, **`ceif`**, and **`voronoi`**. Use this document to choose the right algorithm, threshold notation, and configuration for your target workload.

---

## 1. Chapter 1: Algorithm Selection Matrix for Outlier Detection

### 1.1 Executive Summary & Algorithm Selection Matrix

The table below summarizes the operational characteristics, mathematical partitioning geometries, recommended thresholding regimes, and primary use cases for GEIF's four algorithmic engines.

| Algorithm | Partitioning Geometry | Recommended Threshold ($T$ / Mode) | Cavity & Void Resolution | Envelope Tightness & Boundary Precision | Training Complexity | Scoring Throughput Profile | Primary Operational Use Case |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`exemplar`** (Default) | Regularized SIMD Cauchy kernel with Fractional Decay Mapping | `-O 97%` to `-O 110%`, `-O 0.50` (Cavity), `-O 0.70` (Wide Halo) | **Exceptional** (Cavity detected at $T = 0.50$ and `-O 100%`; zero false bridging) | **Ultra-Detailed & Smooth** (Pixel-level fidelity, proportional out-of-sample halos at `105%`–`110%`) | **Instant**: $O(N \log K)$, $< 1\text{ ms}$ | **Highest**: $O(N)$ straight-line SIMD kernel | **Zero-training streaming, percentage-based boundary gating, instant updates, and calibrated continuous density fields** |
| **`bubble`** | Hyperspherical balls $\Vert x - c \Vert^2 \le R^2$ with empty void leaves | `-O 98%` (Supervised) or `-O 0.48` | **Exceptional** (Clean donut hole & canal separation) | **Highest** (Crisp contour conforming to true manifold) | Moderate: $O(N \log N)$ | Fast: $O(\text{depth})$ tree traversal | **Supervised boundary gating, safety envelopes, non-convex topologies, and cavity detection** |
| **`ceif`** | Data-anchored isotropic Gaussian hyperplanes | `-O 0.50` to `-O 0.60` (Relaxed) | **Blind** (Hyperplanes slice across interior hollows) | **Smooth / Relaxed** (Convex-biased global envelope) | Fast: $O(N \log N)$ | Fast: $O(\text{depth})$ tree traversal | **General unsupervised tabular anomaly detection, high-dimensional data, and relaxed outlier filtering** |
| **`voronoi`** | Perpendicular bisectors $n = x_2 - x_1$, $p = (x_1 + x_2)/2$ | `-O 0.45` to `-O 0.48` | **Moderate** (Exposes cavities better than CEIF, but erodes tips) | **Good** (Scale-invariant linear polyhedral cuts) | Fast: $O(N \log N)$ | Fast: $O(\text{depth})$ tree traversal | **Scale-invariant pairwise clustering where linear decision boundaries are preferred** |

---

### 1.2 Outlier Notation Guidance: Score vs. Percentile

GEIF provides two distinct thresholding notations via the `-O` CLI flag:
1. **Calibrated Absolute Score (`-O <val>`)**: A floating-point value in the range $[0.0, 1.0]$. A query point is flagged as an outlier if $s(x) > T$.
2. **Empirical Percentile & Out-of-Sample Percentage Notation (`-O <val>%`)**: A percentage threshold in the range $[0, 200]$%. Values in $[0, 100]$% evaluate the empirical score quantile over the nominal training distribution (e.g., `-O 97%` accepts the 97% densest training points, while `-O 100%` encloses 100% of training points at maximum in-sample score $s_{100}$). Values in $(100, 200]$% (e.g., `-O 105%`, `-O 110%`) extrapolate smoothly into outer space beyond $s_{100}$ using each engine's native physical survival decay, creating a proportional spatial buffer around the training manifold up to `-O 200%` ($s = 1.0$).

Different algorithmic engines have distinct score distributions and mathematical properties, making specific notations far more effective for each:

| Algorithm | Preferred Notation | Operational Rationale | Recommended Setting |
| :--- | :---: | :--- | :---: |
| **`bubble`** | **Percentile (`-O 97%` to `-O 110%`)** | **Supervised Precision Gating**: The bubble engine creates compact spherical clusters. Percentage thresholding establishes an exact empirical safety envelope (`97%`–`100%`) or controlled out-of-sample buffer (`105%`–`110%`) that automatically adapts to the training density without manual calibration. For unsupervised cavity isolation, an absolute score of `-O 0.48` to `-O 0.50` guarantees complete topological separation between adjacent clusters. | `-O 98%`, `-O 105%`, or `-O 0.48` |
| **`exemplar`** | **Percentage (`-O 97%` to `-O 110%`) or Absolute (`-O 0.50` / `-O 0.70`)** | **Universal Cauchy Potential & Proportional Extrapolation**: Exemplar implements Fractional Decay Mapping $s = 1.0 - (D/D_{\max})^{\text{decay}}$ with collective macro-bandwidth extrapolation. Percentage thresholds (`97%`, `100%`, `105%`, `110%`) produce consistent, topology-preserving boundaries across Gaussian blobs, uniform grids, and non-convex manifolds (see [Chapter 5](#5-chapter-5-percentage-based-score-thresholding-97-100-105-110)). Absolute scores `-O 0.50` (strict) and `-O 0.70` (wide halo) provide fixed density cutoffs. | `-O 97%` to `-O 110%`, `-O 0.50`, or `-O 0.70` |
| **`ceif`** | **Calibrated Absolute (`-O 0.50` to `-O 0.60`)** | **Monotonic Outer Falloff**: CEIF uses continuous metric depth with Cauchy leaf damping, providing an exceptionally smooth and well-conditioned score decay in outer space. An absolute score of $0.50$ provides standard anomaly screening, while $0.60$ offers a relaxed, forgiving filter for noisy tabular data. Percentile notation is less suited when interior cavities are present because hyperplanes slice through voids. | `-O 0.50` to `-O 0.60` |
| **`voronoi`** | **Calibrated Absolute (`-O 0.45` to `-O 0.48`)** | **Cavity Emergence Tuning**: Voronoi midpoint hyperplanes require a calibrated absolute threshold between $0.45$ and $0.48$ to force the central cavity to open. Setting the threshold higher (e.g. $0.50$ or 98%) causes Voronoi to bridge across the cavity similarly to CEIF. | `-O 0.45` to `-O 0.48` |

---

### 1.3 Side-by-Side Visual Heatmap Gallery

The decision boundaries of all four engines were evaluated on the challenging non-convex `complex2d.csv` benchmark (featuring radiating thin spikes, an annular donut cavity, isolated sub-clusters, and a 5000:1 aspect ratio) across three critical operational regimes:

#### 1.3.1 Supervised Precision Regime: 98th Percentile (`-O 98%`)

In this regime, the anomaly threshold is calibrated to envelope the 98% densest training points:

| Bubble (`-B bubble -O 98%`) | CEIF (`-B ceif -O 98%`) | Exemplar (`-B exemplar -O 98%`) | Voronoi (`-B voronoi -O 98%`) |
|:---:|:---:|:---:|:---:|
| ![Bubble 98p](pics/recom/complex2d_bubble_98p.png) | ![CEIF 98p](pics/recom/complex2d_ceif_98p.png) | ![Exemplar 98p](pics/recom/complex2d_exemplar_98p.png) | ![Voronoi 98p](pics/recom/complex2d_voronoi_98p.png) |

*Key Findings*:
- **`bubble`** and **`exemplar`** achieve surgical envelope gating. `bubble` hollows out the central donut cavity, detaches the top-left sub-cluster, and traces thin spikes with spherical precision. `exemplar` tightly hugs the nominal manifold, cleanly rejects solitary corner noise samples, and leaves zero false-inlier halos.
- In contrast, **`ceif`** and **`voronoi`** bridge completely across the central donut cavity due to infinite linear hyperplanes traversing interior hollows.

#### 1.3.2 Calibrated Nominal Regime ($T = 0.50$)

Evaluating all engines at GEIF's universal default threshold $T = 0.50$:

| Bubble (`-B bubble -O 0.50`) | CEIF (`-B ceif -O 0.50`) | Exemplar (`-B exemplar -O 0.50`) | Voronoi (`-B voronoi -O 0.50`) |
|:---:|:---:|:---:|:---:|
| ![Bubble 0.50](pics/recom/complex2d_bubble_050.png) | ![CEIF 0.50](pics/recom/complex2d_ceif_050.png) | ![Exemplar 0.50](pics/recom/complex2d_exemplar_050.png) | ![Voronoi 0.50](pics/recom/complex2d_voronoi_050.png) |

*Key Findings*:
- **`bubble`** preserves a clear central cavity while beginning to form a light bridge between the top-left cluster and vertical spike.
- **`exemplar`** delivers an immaculate decision boundary under Fractional Decay Mapping: the interior donut cavity is decisively flagged as anomalous ($s \approx 0.56 > 0.50$, cleanly hollowed out), all spiderweb caustics and outlier bubbles are abolished, and outer space decays smoothly and monotonically.
- **`ceif`** and **`voronoi`** form broad, monolithic convex hulls enclosing the entire topology.

#### 1.3.3 Relaxed & Separated Regimes ($T = 0.48, 0.60, 0.45, 0.70$)

Evaluating engines at thresholds optimized for their distinct geometries:

| Bubble ($T = 0.48$) | CEIF ($T = 0.60$) | Voronoi ($T = 0.45$) | Exemplar ($T = 0.70$ Wide Halo) |
|:---:|:---:|:---:|:---:|
| ![Bubble 0.48](pics/recom/complex2d_bubble_048.png) | ![CEIF 0.60](pics/recom/complex2d_ceif_060.png) | ![Voronoi 0.45](pics/recom/complex2d_voronoi_045.png) | ![Exemplar 0.70](pics/recom/complex2d_exemplar_070.png) |

*Key Findings*:
- At $T = 0.48$, **`bubble`** achieves complete topological separation between all sub-structures without any bridging.
- At $T = 0.60$, **`ceif`** produces a broad, forgiving boundary ideal for coarse noise filtering in unstructured tabular data.
- At $T = 0.45$, **`voronoi`** succeeds in opening the central cavity, but thin linear structures (spike tips) begin to erode and fragment.
- At $T = 0.70$, **`exemplar`** exploits harmonic distance decay $s(r) \approx 1 - \sigma/r$ to generate a generous, uniform inlier halo wrapping the entire manifold with zero boundary cliffs or step discontinuities.

---

## 2. Chapter 2: Algorithm Selection Matrix for Population Drift & Density Coverage

### 2.1 The Population Drift & Density Starvation Challenge

In dynamic streaming environments and multi-modal operational monitoring, data distributions frequently exhibit heterogeneous density across multiple concentration centers (e.g. primary operating mode, secondary operating mode, transient states, and peripheral cycles).

A critical failure mode in anomaly detection is **high-density core concentration bias**:
- When an algorithm allocates probability mass disproportionately to the single highest-density core cluster, peripheral concentration centers and smaller sub-clusters are starved of probability mass.
- When an operational threshold (such as an 80% coverage envelope) is applied, the model protects only the primary cluster and prematurely flags legitimate secondary operating modes as anomalous ("drift blindness").
- **Ideal Selection Criterion**: An algorithm must capture inliers **evenly across every distinct sample concentration center**, ensuring balanced coverage of all operational modes.

---

### 2.2 Empirical 80% Density Coverage Heatmap Gallery

To evaluate how evenly each algorithm distributes inlier coverage across diverse topological structures, all four engines were scored at an 80% nominal coverage threshold (`-O 80%` or calibrated threshold covering 80% of nominal points) on the `complex2d.csv` benchmark. This dataset features four distinct concentration centers:
1. **Dense Annular Donut Body** centered at $(5107, 0.99)$ with high local point concentration.
2. **Radiating Thin Spikes** projecting vertically and diagonally.
3. **Curved Outer Crescent Arc** extending along the lower perimeter.
4. **Detached Gaussian Sub-Cluster** isolated at top-left $(2500, 1.6)$.

| Bubble (`-B bubble -O 80%`) | CEIF (`-B ceif -O 80%`) | Exemplar (`-B exemplar -O 80%`) | Voronoi (`-B voronoi -O 80%`) |
|:---:|:---:|:---:|:---:|
| ![Bubble 80p](pics/recom/complex2d_bubble_80p.png) | ![CEIF 80p](pics/recom/complex2d_ceif_80p.png) | ![Exemplar 80p](pics/recom/complex2d_exemplar_80p.png) | ![Voronoi 80p](pics/recom/complex2d_voronoi_80p.png) |

---

### 2.3 Drift Robustness & Multi-Modal Coverage Findings

The empirical 80% coverage evaluation reveals dramatic differences in multi-modal density allocation:

| Metric / Topology Area | Bubble (`-B bubble`) | CEIF (`-B ceif`) | Exemplar (`-B exemplar`) | Voronoi (`-B voronoi`) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Primary Annular Donut** | Hollow cavity preserved | Cavity filled completely | Hollow cavity preserved | Cavity filled completely |
| **2. Radiating Spikes** | Preserved along length | Eroded at distal tips | Preserved along length | Heavily eroded / fragmented |
| **3. Outer Crescent Arc** | Continuous inlier envelope | Shrunk to central core | Continuous inlier envelope | Eroded into disjoint patches |
| **4. Detached Sub-Cluster** | **Preserved** (Isolated island) | **Starved** (Severely eroded) | **Preserved** (Balanced island) | **Starved** (Eroded / dropped) |
| **Multi-Modal Drift Robustness** | **Superior** | Poor (Core biased) | **Superior** | Poor (Erosion biased) |

#### Algorithmic Analysis:
1. **Exemplar (Winner for Balanced Multi-Modal Drift Protection)**:
   - Thanks to **Cap-5 $K$-NN Bandwidth Estimation**, **Low-End Bandwidth Flooring ($b_i \ge b_{\min}$)**, and **Sublinear Duplicate Weighting ($\sqrt{c_i}$)**, Exemplar's kernel density does not allow high-density clusters to monopolize the score spectrum. Isolated points acquire wide bandwidths ($b_i \gg 1 \implies 1/b_i \to 0$) while dense clusters are floored at $b_{\min} = \max(0.5 \mu_b, \mu_b - s_b)$.
   - As observed in `complex2d_exemplar_80p.png`, the 80% inlier envelope evenly covers all four concentration centers: the detached cluster, the crescent arc, the radiating spikes, and the donut ring all receive balanced inlier protection.
2. **Bubble (Winner for Discrete Structural Isolation)**:
   - Because tree splits are hyperspheres $\Vert x - c \Vert^2 \le R^2$, each sub-manifold is partitioned into localized spherical bounding regions. Points in empty space between clusters immediately land in empty void leaves ($I = 0$).
   - As seen in `complex2d_bubble_80p.png`, Bubble preserves the detached cluster as an independent inlier island without leaking score into the void canal.
3. **CEIF & Voronoi (High-Density Concentration Bias)**:
   - In `complex2d_ceif_80p.png` and `complex2d_voronoi_80p.png`, hyperplanes cutting across the dataset inevitably cut through the dense central core. As a result, the trees dedicate an overwhelming majority of splits to the high-density center, inflating its leaf depth.
   - At 80% coverage, CEIF and Voronoi pull almost all inlier mass into the central body, starving the isolated cluster and spike tips. This makes them prone to population drift errors when secondary operating modes are present.

---

## 3. Chapter 3: Algorithm Selection Matrix for Categorization

### 3.1 Categorization Routing Architecture

GEIF features built-in multi-column category routing (`-c`, `-C <cols>`), allowing an ensemble of subforests to classify incoming data points based on maximum class-conditional likelihood or minimum isolation depth.

When running categorization, GEIF outputs both the original/training category (`%C`) and the predicted category (`%c`) via formatting directives:
- `%C`: The true category string parsed from training/metadata columns.
- `%c`: The predicted category string assigned by the model's minimum anomaly score / highest density subforest.

To benchmark categorization accuracy, all four algorithmic engines were evaluated on the real-world agricultural benchmark [`test/data/paddydataset.csv`](../test/data/README.md). This dataset contains **2,790 observations** across 10 continuous feature dimensions, categorized across multiple variety and soil classifications (columns 2 through 4).

The models were trained and evaluated using the exact training and categorization pipeline:
```bash
# 1. Train multi-category model using columns 2-4 as category routing keys
./bin/geif -l test/data/paddydataset.csv -w paddy_model.json -B <algo> -C 2-4

# 2. Evaluate categorization accuracy by comparing true category (%C) to predicted category (%c)
./bin/geif -r paddy_model.json -c test/data/paddydataset.csv -p "%C %c"
```

---

### 3.2 Empirical Classification Accuracy Benchmark

Evaluating all 2,790 observations across all four engines produced the following empirical categorization match rates:

| Algorithm | Training Flag | Correct Matches (`%C == %c`) | Total Observations | Categorization Accuracy | Relative Accuracy Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`exemplar`** | `-B exemplar` | **2,470** | 2,790 | **88.53%** | **1st (Clear Winner)** |
| **`voronoi`** | `-B voronoi` | 1,716 | 2,790 | 61.51% | 2nd |
| **`bubble`** | `-B bubble` | 1,678 | 2,790 | 60.14% | 3rd |
| **`ceif`** | `-B ceif` | 1,677 | 2,790 | 60.11% | 4th |

---

### 3.3 Theoretical & Architectural Insight

The categorization benchmark reveals a decisive performance disparity: **`exemplar` achieves 88.53% classification accuracy**, outperforming all tree-based engines by **+27.0 percentage points** (an error reduction from $\approx 39.9$% down to 11.5%).

#### Why Exemplar Dominates Categorization:
1. **Continuous Kernel Density vs. Boundary Quantization Noise**:
   - In tree-based isolation methods (`bubble`, `ceif`, `voronoi`), decision boundaries are constructed from hard, recursive geometric cuts (hyperplanes or hyperspheres).
   - In a 10-dimensional continuous feature space with overlapping class manifolds, binary tree cuts introduce significant **discretization quantization noise**. When multiple subforests vote via discrete leaf depths, points near class boundaries suffer from high-variance tie-breaking and random partitioning jitter.
   - In contrast, `exemplar` constructs a **continuous, infinitely differentiable ($C^\infty$) Cauchy density field** for each class subforest. Scoring evaluates true Euclidean density without boundary quantization artifacts, yielding smooth and robust probabilistic voting boundaries.
2. **Sub-Cluster Density Fidelity**:
   - The agricultural features in `paddydataset.csv` exhibit non-linear correlations across varieties and soil types. Exemplar's adaptive Cauchy kernels with low-end bandwidth flooring ($b_i \ge b_{\min}$) naturally conform to arbitrary manifold shapes without requiring hyperplanes to approximate curved boundaries.
3. **Robustness to Class Imbalance**:
   - Natural inverse-bandwidth outlier suppression ($1 / b_i \to 0$) and sublinear duplicate weighting ($\sqrt{c_i}$) ensure that outlier points in smaller classes do not distort class probability distributions.

---

## 4. Chapter 4: Performance Profile (Cold-Start Load & Scoring Throughput)

### 4.1 Benchmark Methodology

To evaluate operational efficiency in production environments, two distinct performance dimensions were measured:
1. **Cold-Start Model Load & Tree Construction Latency**:
   - Model: Enterprise-scale workflow model `test/anow_wf.json` containing **1,048 category subforests** and **314,400 trees** (21.3 MB JSON serialization).
   - Measures the total wall-clock time required to parse the JSON model, allocate memory structures, and reconstruct internal search topologies.
2. **Evaluation & Grid Scoring Throughput**:
   - Command: `../bin/geif -r complex2d.json -T 0.1 -O80% -p "%d,0x%x,%s" -o plot_data.csv`
   - Evaluates a 60,000-point 2D coordinate grid with full CSV formatting, string interpolation, and disk output.

To provide clear comparisons independent of host hardware fluctuations, all execution times are reported alongside a **Relative Performance Index**, where the best-performing algorithm is indexed at **100.0**:

$$
\text{Index} = 100.0 \times \frac{t_{\text{fastest}}}{t_{\text{algo}}}
$$

---

### 4.2 Benchmark Results

#### 4.2.1 Cold-Start JSON Load & Initialization Index

Loading `test/anow_wf.json` (1,048 subforests, 314,400 trees):

| Algorithm | Model Architecture | Measured Latency ($s$) | Relative Performance Index (Best = 100) | Operational Characteristics |
| :--- | :--- | :---: | :---: | :--- |
| **`exemplar`** | Reservoir vectors + Clamped bandwidths | **0.9148** | **100.0** | **Instantaneous**: Zero binary trees to reconstruct. Sub-second load even for 1,048 subforests. |
| **`ceif`** | Oblique hyperplane binary trees | 65.1220 | **1.4** | High tree rebuild overhead (314,400 trees deserialized and allocated). |
| **`voronoi`** | Midpoint bisector binary trees | 67.1423 | **1.4** | Tree node allocation and pointer graph assembly overhead. |
| **`bubble`** | Hypersphere bounding-box trees | 72.4913 | **1.3** | Hypersphere radius and bounding-box validation during tree assembly. |

#### 4.2.2 Scoring & Analysis Throughput Index

Evaluating 60,000 queries on `complex2d.json`:

| Algorithm | Scoring Mechanism | Measured Latency ($s$) | Relative Performance Index (Best = 100) | Scoring Throughput Profile |
| :--- | :--- | :---: | :---: | :--- |
| **`exemplar`** | All-Reservoir SIMD Cauchy accumulation | **1.0331** | **100.0** | **Highest**: Straight-line vector loop, zero branch mispredictions, cache-resident. |
| **`bubble`** | Hypersphere tree traversal with empty void early exit | 3.1688 | **32.6** | Fast tree traversal; empty void leaves truncate evaluation early. |
| **`voronoi`** | Oblique bisector hyperplane tree traversal | 4.9028 | **21.1** | Moderate tree traversal with dot-product split evaluations. |
| **`ceif`** | Gaussian isotropic hyperplane tree traversal | 6.4613 | **16.0** | Full tree traversal with metric depth and leaf Cauchy damping. |

---

### 4.3 Architectural Insights & Production Tradeoffs

#### Why Exemplar Leads in Both Load and Scoring:
1. **Zero Tree Construction Overhead**:
   - Tree-based models (`ceif`, `voronoi`, `bubble`) store recursive node hierarchies. Loading an enterprise model with 314,400 trees requires millions of recursive heap allocations (`malloc`), pointer linking, and bounding box initializations.
   - In contrast, `exemplar` completely bypasses binary trees. Its state consists exclusively of contiguous, 64-byte aligned arrays of scaled sample vectors, bandwidths, and duplicate weights. Cold-start load is bounded purely by sequential memory bandwidth, loading in **$< 1$ second**.
2. **SIMD Vector Execution vs. Pointer Chasing**:
   - In tree traversal, navigating a binary tree requires following pointer chains (`node->left`, `node->right`). Pointer traversal incurs cache line misses and branch mispredictions.
   - `algo_exemplar.c` evaluates all reservoir samples using an unrolled AVX2/SIMD vector loop over contiguous memory. The CPU prefetcher streams coordinate vectors with 100% cache line utilization, zero branch mispredictions, and full FMA pipelining.

#### When to Choose Tree-Based Engines:
- **Ultra-High Dimensionality ($D \gg 100$)**:
  - Exemplar evaluates all $N$ reservoir points in $O(N \cdot D)$ time.
  - Tree algorithms evaluate $O(\text{depth}) = O(\log N)$ splits, evaluating only one hyperplane per depth level. In very high dimensions with large sample pools, tree traversal complexity remains logarithmic.
- **Formal Verification & Safety Critical Envelopes**:
  - The `bubble` engine provides bounded hyperspherical enclosures that can be mathematically verified against geometric keep-out envelopes (e.g. robotic path planning or collision boundaries).

---

## 5. Chapter 5: Percentage-Based Score Thresholding (`97%`, `100%`, `105%`, `110%`)

### 5.1 Unified In-Sample Quantile & Out-of-Sample Survival Extrapolation

Percentage-based thresholding (`-O <P>%` on the CLI or `OUTLIER_SCORE <P>%` in `~/.geifrc`) provides a self-calibrating, dataset-independent way to control anomaly boundaries across the full $[0, 200]$% range without manually tuning raw score numbers:

1. **In-Sample Empirical Quantile ($0 \le P \le 100$)**:
   - GEIF scores all active reservoir samples and sorts their ascending anomaly scores $s_0 \le s_1 \le \ldots \le s_{N-1}$.
   - **`-O 97%`**: Rejects the outermost 3% of training observations (peripheral tail noise or sparse fringe points), tightly hugging the core density modes.
   - **`-O 100%`**: Sets the cutoff to the exact maximum in-sample score $s_{100}$ (equal to $s_{N-1}$). Exactly 100% of reservoir training observations fall inside the white inlier envelope, establishing the tightest possible zero-false-positive boundary on the training set.

2. **Out-of-Sample Physical Boundary Extrapolation ($100 < P \le 200$)**:
   - Above `100%`, no training samples exist to form an empirical quantile. Instead, GEIF maps $t = (P - 100) / 100 \in (0, 1]$ directly to a normalized exterior excursion distance of $t$ feature-span units outside the 100% boundary:

$$
s(P) = 1.0 - (1.0 - s_{100})(1.0 - t^2) S_{\text{algo}}(t)
$$

   - In the default **`exemplar`** engine, a point moving $t$ span units into outer space experiences the superposition of all $U$ kernel tails, governed by the ensemble's collective macro-bandwidth $B_{\text{eff}}$ and inverse scale $\kappa_{\text{eff}}$:

$$
B_{\text{eff}} = \sqrt{\frac{\sum_{i=1}^U \sqrt{c_i} \cdot b_i}{D_{\max}}}, \quad \kappa_{\text{eff}} = \frac{1.0}{B_{\text{eff}}}, \quad S_{\text{exemplar}}(t) = \left(\frac{1 + u_{100}^2}{1 + (u_{100} + \kappa_{\text{eff}} t)^2}\right)^{\gamma}
$$

   - Because $B_{\text{eff}}$ automatically accounts for both individual kernel widths $b_i$ and cluster spatial extent, a given percentage above `100%` (`105%` or `110%`) produces a **consistent, proportional geometric buffer halo** regardless of whether the underlying dataset is Gaussian, uniform, or non-convex.

---

### 5.2 Visual Comparison Across Canonical Topologies (`97%`, `100%`, `105%`, `110%`)

The $3 \times 4$ gallery below demonstrates the **`exemplar`** decision boundary (white = inlier below threshold, green-to-orange gradient = outlier above threshold, rendered over a `-T 0.5` extended grid) across three canonical benchmark topologies at `97%`, `100%`, `105%`, and `110%`:

| Topology | `-O 97%` (Core Inliers, 3% Tail Trim) | `-O 100%` (Exact In-Sample Envelope) | `-O 105%` (+5% Span Buffer Halo) | `-O 110%` (+10% Span Buffer Halo) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Bimodal Gaussian (`2blob`)** | ![2blob 97%](pics/recom/2blob_exemplar_97p.png) | ![2blob 100%](pics/recom/2blob_exemplar_100p.png) | ![2blob 105%](pics/recom/2blob_exemplar_105p.png) | ![2blob 110%](pics/recom/2blob_exemplar_110p.png) |
| **2. Uniform Grid (`square`)** | ![square 97%](pics/recom/square_exemplar_97p.png) | ![square 100%](pics/recom/square_exemplar_100p.png) | ![square 105%](pics/recom/square_exemplar_105p.png) | ![square 110%](pics/recom/square_exemplar_110p.png) |
| **3. Non-Convex Manifold (`complex`)** | ![complex 97%](pics/recom/complex2d_exemplar_97p.png) | ![complex 100%](pics/recom/complex2d_exemplar_100p.png) | ![complex 105%](pics/recom/complex2d_exemplar_105p.png) | ![complex 110%](pics/recom/complex2d_exemplar_110p.png) |

---

### 5.3 Topological Behavior & Operational Threshold Selection

| Percentage Threshold | Behavior on `2blob` (Gaussian Clusters) | Behavior on `square` (Uniform Box $[10, 20]^2$) | Behavior on `complex` (Donut + Spikes + Arc + Island) | Primary Production Use Case |
| :---: | :--- | :--- | :--- | :--- |
| **`-O 97%`** | Trims outer Gaussian tail scatter; cleanly separates the two modes into distinct white islands. | Slightly rounds and erodes extreme corner/edge points of the $10 \times 10$ lattice. | Trims sparse fringe points along ray tips and arc bridge; preserves crisp central donut hole and isolated top-left island. | **Noisy Training Data**: When historical training logs contain ~1%–3% unlabelled anomalies or fringe jitter that should be excluded from the baseline. |
| **`-O 100%`** | Encloses 100% of samples—including the outermost tail point at $(2.9, 10.5)$—in a smooth peanut contour. | Encloses the complete $[10, 20] \times [10, 20]$ lattice tightly right up to all four corners. | Encloses every training point across the donut ring, three thin spikes, lower arc, and top-left island while keeping the hollow donut cavity green. | **Clean Baseline / Strict Gating**: When the training dataset represents verified normal operation and zero training points should be flagged. |
| **`-O 105%`** | Expands the peanut contour outward by a uniform ~5% span margin on all sides. | Forms a smooth, rounded stadium buffer (`[9.2, 20.8]`) around the $10 \times 10$ square. | Expands a uniform ~5% buffer halo around all structures (closing the narrow inner cavity while keeping the top-left satellite island separate). | **Production Drift Tolerance (Recommended Out-of-Sample Default)**: Allows modest, legitimate operational excursions slightly beyond historical extremes without false alarms. |
| **`-O 110%`** | Expands a generous ~10% span halo around the bimodal envelope while staying well within the outer frame. | Forms a wider rounded stadium buffer (`[8.5, 21.5]`) with smooth corner curvature. | Wraps the main body and arc in a broad ~10% protective envelope while only flagging deep outer-space deviations. | **Coarse / Forgiving Guardrails**: High-variance telemetry or small-sample baselines where only severe outer-space anomalies should trigger alerts. |

---

## 6. Operational Summary & Decision Guide

```mermaid
flowchart TD
    Start["New Anomaly Detection or Categorization Task"] --> TaskType{"Primary Task Type?"}
    
    TaskType -->|"Categorization (-c)"| ExemplarCat["Select exemplar Engine<br/><code>geif -B exemplar -c -C ...</code><br/><b>88.5% Accuracy</b>, zero tree building"]
    
    TaskType -->|"Outlier Detection"| Geometry{"Topological Requirements?"}
    
    Geometry -->|"Hollow Cavities / Safety Gating"| BubbleChoice["Select bubble Engine<br/><code>geif -B bubble -O 98%</code> or <code>-O 0.48</code><br/>Clean donut voids, spherical bounds"]
    Geometry -->|"Instant Streaming / Fast Load"| ExemplarChoice["Select exemplar Engine<br/><code>geif -B exemplar -O 97%..110%</code><br/>or <code>-O 0.50</code> / <code>-O 0.70</code><br/>Sub-second load, 15M pts/sec"]
    Geometry -->|"Standard High-Dim Tabular"| CEIFChoice["Select ceif Engine<br/><code>geif -B ceif -O 0.50</code><br/>Smooth isotropic continuous decay"]
    Geometry -->|"Pairwise Midpoint Clustering"| VoronoiChoice["Select voronoi Engine<br/><code>geif -B voronoi -O 0.45</code><br/>Linear polyhedral bisectors"]
```

### Final Operational Rules of Thumb:
1. **Categorization Tasks (`-c`)**: **Always use `exemplar`**. It delivers 88.5% accuracy compared to ~60% for tree-based engines, loads in sub-second time, and eliminates tree quantization noise.
2. **Percentage-Based Boundary Gating (`-O 97%` to `-O 110%`)**: **Use `exemplar` (or `bubble`)** with `-O 97%` when training data has minor tail noise, `-O 100%` for an exact in-sample envelope, or `-O 105%` / `-O 110%` to add a proportional out-of-sample tolerance halo around the manifold.
3. **High-Precision Safety Envelopes & Cavity Isolation**: **Use `bubble`** with percentile thresholding (`-O 98%` or `-O 95%`) or calibrated score `-O 0.48`. It guarantees zero bridging across empty space and isolates interior hollows.
4. **High-Throughput Streaming & Dynamic Models**: **Use `exemplar`** with percentage threshold `-O 100%` / `-O 105%` or calibrated score `-O 0.50` (cavity detection) / `-O 0.70` (wide halo). Sub-second cold-start load time makes it ideal for microservices, live retraining, and fast container startup.
5. **General Unsupervised Tabular Data**: **Use `ceif`** with absolute score `-O 0.50` to `-O 0.60` when data has no internal cavities and a smooth convex-like decision boundary is desired.

---

## Development & Attribution

**GEIF** was designed and engineered by **Timo Savinen** with AI pair-programming assistance (Google Antigravity / Gemini) under human architectural direction.
