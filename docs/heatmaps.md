# GEIF: Empirical Heatmaps, Multi-Algorithm Decision Manifolds & Topological Analysis

**Author / Maintainer:** Timo Savinen (AI-assisted)

This document provides visual anomaly score heatmaps and empirical decision manifold analyses for **GEIF (Geometric Extended Isolation Forest)** across a diverse suite of 2D synthetic topologies.

Visualizing the decision manifold in 2D illustrates how GEIF's algorithmic engines—**Hyperspherical Bubble Partitioning**, **Voronoi Hyperplane Bisectors**, **Exemplar Kernel Density**, and the **Continuous Hyperplane Engine**—govern model inference across challenging geometries, internal cavities, and non-convex topologies.

---

## 1. Benchmark Datasets

We evaluate GEIF across six standard topological benchmarks:
1. **Two Blobs (`2blob.csv`)**: Two separated Gaussian clusters (1,000 points).
2. **Square (`square.csv`)**: Uniform grid of points forming an open square boundary (76 points).
3. **Circle (`circle.csv`)**: A single circular ring distribution (360 points).
4. **Two Circles (`2circle.csv`)**: Concentric ring structure with a central and intermediate void (720 points).
5. **Complex 2D (`complex2d.csv`)**: Multi-scale non-convex teapot with an annular donut hole, 3 radiating needle spikes, an outer crescent arc, and an isolated Gaussian sub-cluster (1,465 points; 5000:1 aspect ratio).
6. **Elongated Diagonal (`Wtest.csv`)**: Narrow linear cluster with extreme dimension magnitude disparity ($X \in [5, 120]$ vs. $Y \in [10^5, 2 \times 10^6]$).

### Baseline Sample Distributions (Training Data)

| Two Blobs | Square | Circle |
|:---:|:---:|:---:|
| ![Two Blobs](pics/geif/2blob_raw.png) | ![Square](pics/geif/square_raw.png) | ![Circle](pics/geif/circle_raw.png) |

---

## 2. Multi-Algorithm Score Profiles Across Thresholds ($T \in \{0.0, 0.35, 0.50, 0.60\}$)

GEIF features **Zero Kelvin universal scale calibration** across all tree engines:
- $T = 0.00$: Full continuous score landscape (showing depth gradients everywhere; all points have score $\ge 0.00$).
- $T = 0.35$: Dense core inlier regime.
- $T = 0.50$: Calibrated universal default inlier/outlier boundary.
- $T = 0.60$: Extreme outlier regime.

The empirical score distributions and outlier percentages across all 6 datasets and 4 algorithm engines are summarized below:

### 2.1 Bubble Engine (`-B bubble`, Default)

Hyperspherical cavity carving trees with empty void leaves and in-place $O(N)$ quickselect median cuts:

| Dataset | Samples ($N$) | Min Score | Mean Score | Median Score | Max Score | Outliers $\ge 0.00$ | Outliers $\ge 0.35$ | Outliers $\ge 0.50$ | Outliers $\ge 0.60$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Two Blobs (`2blob`)** | 1,000 | 0.2241 | 0.3248 | 0.3182 | 0.4782 | 100.0% (1,000) | 26.5% (265) | **0.0% (0)** | 0.0% (0) |
| **Square (`square`)** | 250 | 0.2312 | 0.3155 | 0.3110 | 0.4529 | 100.0% (250) | 16.8% (42) | **0.0% (0)** | 0.0% (0) |
| **Circle (`circle`)** | 425 | 0.2542 | 0.3358 | 0.3348 | 0.4465 | 100.0% (425) | 30.1% (128) | **0.0% (0)** | 0.0% (0) |
| **Two Circles (`2circle`)** | 900 | 0.4060 | 0.4438 | 0.4434 | 0.4838 | 100.0% (900) | 100.0% (900) | **0.0% (0)** | 0.0% (0) |
| **Complex 2D (`complex2d`)** | 1,465 | 0.2296 | 0.3636 | 0.3627 | 0.4992 | 100.0% (1,465) | 68.1% (998) | **0.0% (0)** | 0.0% (0) |
| **Elongated (`Wtest`)** | 500 | 0.2238 | 0.3160 | 0.3106 | 0.4366 | 100.0% (500) | 18.0% (90) | **0.0% (0)** | 0.0% (0) |

*Key Characteristic*: The Bubble engine strictly bounds inliers beneath the universal $T = 0.50$ threshold (0.0% false outliers across all nominal datasets) while maintaining high sensitivity to internal voids.

---

### 2.2 Voronoi Engine (`-B voronoi`)

Pure perpendicular bisector splits between sample pairs:

| Dataset | Samples ($N$) | Min Score | Mean Score | Median Score | Max Score | Outliers $\ge 0.00$ | Outliers $\ge 0.35$ | Outliers $\ge 0.50$ | Outliers $\ge 0.60$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Two Blobs (`2blob`)** | 1,000 | 0.2052 | 0.3204 | 0.3129 | 0.5112 | 100.0% (1,000) | 26.0% (260) | 0.2% (2) | 0.0% (0) |
| **Square (`square`)** | 250 | 0.2180 | 0.3188 | 0.3162 | 0.4891 | 100.0% (250) | 20.8% (52) | 0.0% (0) | 0.0% (0) |
| **Circle (`circle`)** | 425 | 0.2655 | 0.3403 | 0.3392 | 0.4309 | 100.0% (425) | 35.1% (149) | 0.0% (0) | 0.0% (0) |
| **Two Circles (`2circle`)** | 900 | 0.4251 | 0.4899 | 0.4906 | 0.5617 | 100.0% (900) | 100.0% (900) | 38.8% (349) | 0.0% (0) |
| **Complex 2D (`complex2d`)** | 1,465 | 0.2357 | 0.3430 | 0.3358 | 0.5836 | 100.0% (1,465) | 43.9% (643) | 1.2% (17) | 0.0% (0) |
| **Elongated (`Wtest`)** | 500 | 0.2084 | 0.3121 | 0.3051 | 0.5133 | 100.0% (500) | 17.8% (89) | 0.4% (2) | 0.0% (0) |

*Key Characteristic*: Naturally scale-invariant due to midpoint bisectors, providing crisp linear partitions between neighboring clusters.

---

### 2.3 Exemplar Engine (`-B exemplar`)

Non-tree direct SIMD Cauchy kernel density estimation on reservoir samples:

| Dataset | Samples ($N$) | Min Score | Mean Score | Median Score | Max Score | Outliers $\ge 0.00$ | Outliers $\ge 0.35$ | Outliers $\ge 0.50$ | Outliers $\ge 0.60$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Two Blobs (`2blob`)** | 1,000 | 0.0000 | 0.2541 | 0.2443 | 0.7246 | 100.0% (1,000) | 17.4% (174) | 4.2% (42) | 1.3% (13) |
| **Square (`square`)** | 250 | 0.0000 | 0.2355 | 0.2241 | 0.5894 | 100.0% (250) | 14.4% (36) | 2.8% (7) | 0.0% (0) |
| **Circle (`circle`)** | 425 | 0.0000 | 0.2451 | 0.2333 | 0.6366 | 100.0% (425) | 17.6% (75) | 3.8% (16) | 0.7% (3) |
| **Two Circles (`2circle`)** | 900 | 0.0000 | 0.2445 | 0.2334 | 0.6366 | 100.0% (900) | 16.8% (151) | 3.2% (29) | 0.4% (4) |
| **Complex 2D (`complex2d`)** | 1,465 | 0.0000 | 0.2595 | 0.2454 | 0.7372 | 100.0% (1,465) | 19.7% (288) | 4.0% (58) | 1.0% (15) |
| **Elongated (`Wtest`)** | 500 | 0.0000 | 0.2326 | 0.2209 | 0.6453 | 100.0% (500) | 16.6% (83) | 3.8% (19) | 0.4% (2) |

*Key Characteristic*: Direct local density kernel yielding a theoretical 0.0000 floor at sample locations with smooth continuous decay.

---

### 2.4 Hyperplane Engine (`-B ceif`)

Data-anchored isotropic Gaussian cuts with continuous depth traversal and Zero Kelvin calibration:

| Dataset | Samples ($N$) | Min Score | Mean Score | Median Score | Max Score | Outliers $\ge 0.00$ | Outliers $\ge 0.35$ | Outliers $\ge 0.50$ | Outliers $\ge 0.60$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Two Blobs (`2blob`)** | 1,000 | 0.1948 | 0.3184 | 0.3090 | 0.5809 | 100.0% (1,000) | 26.6% (266) | 1.8% (18) | 0.0% (0) |
| **Square (`square`)** | 250 | 0.1557 | 0.3296 | 0.3386 | 0.5489 | 100.0% (250) | 42.0% (105) | 3.2% (8) | 0.0% (0) |
| **Circle (`circle`)** | 425 | 0.2532 | 0.3239 | 0.3187 | 0.4592 | 100.0% (425) | 24.7% (105) | 0.0% (0) | 0.0% (0) |
| **Two Circles (`2circle`)** | 900 | 0.4392 | 0.5048 | 0.5046 | 0.5926 | 100.0% (900) | 100.0% (900) | 54.4% (490) | 0.0% (0) |
| **Complex 2D (`complex2d`)** | 1,465 | 0.2117 | 0.3214 | 0.3099 | 0.6174 | 100.0% (1,465) | 34.7% (509) | 1.6% (23) | 0.1% (1) |
| **Elongated (`Wtest`)** | 500 | 0.1898 | 0.2970 | 0.2852 | 0.5328 | 100.0% (500) | 20.0% (100) | 1.4% (7) | 0.0% (0) |

---

## 3. Decision Boundaries & Calibrated Thresholds (Bubble Visualized)

In these anomaly maps evaluated using GEIF's default **Bubble** algorithm:
- **Black dots**: Original training samples.
- **White regions**: Inliers (points with anomaly score $< T$).
- **Yellow $\to$ Orange $\to$ Red gradient**: Outliers (points with anomaly score $\ge T$ saturated towards 1.0).

| Outlier Threshold ($T$) | Two Blobs (`2blob`) | Square (`square`) | Circle (`circle`) |
|:---|:---:|:---:|:---:|
| **Training Data** | ![2blob raw](pics/geif/2blob_raw.png) | ![square raw](pics/geif/square_raw.png) | ![circle raw](pics/geif/circle_raw.png) |
| **$T = 0.00$** (Full Field) | ![2blob T0](pics/geif/2blob_T0.png) | ![square T0](pics/geif/square_T0.png) | ![circle T0](pics/geif/circle_T0.png) |
| **$T = 0.35$** (Core Inliers) | ![2blob T35](pics/geif/2blob_T35.png) | ![square T35](pics/geif/square_T35.png) | ![circle T35](pics/geif/circle_T35.png) |
| **$T = 0.50$** (Calibrated Default) | ![2blob T50](pics/geif/2blob_T50.png) | ![square T50](pics/geif/square_T50.png) | ![circle T50](pics/geif/circle_T50.png) |

### Key Observations:
1. **Zero Kelvin Floor**: At $T = 0.00$, the densest cluster centroids approach the theoretical inlier floor without premature saturation.
2. **Smooth Boundary**: The transition from inlier white to outlier color is smooth and continuous, free from sharp grid axis-aligned cuts.
3. **Consistency**: $T = 0.50$ cleanly isolates both Gaussian blobs, the square perimeter, and the circular ring without requiring topological re-tuning.
4. **Inter-Cluster Independence**: Neighboring clusters do not distort or contaminate each other's local score contours; each manifold is isolated independently according to its own intrinsic radial geometry.

---

## 4. Complex Topologies: Concentric Rings & The Teapot (Bubble Engine)

Non-convex manifolds with internal cavities (such as concentric rings or the annular donut hole in `complex2d.csv`) pose a fundamental challenge to unconstrained linear cuts, which tend to slice across empty interior voids.

GEIF's default **Bubble** algorithm resolves this natively through **Hyperspherical Cavity Carving**:
- Bounded hyperspherical envelopes $\mathcal{B}(c, R) = (x \in \mathbb{R}^D : \Vert x - c \Vert^2 \le R^2)$ isolate clusters without traversing through empty central hollows.
- When query points fall into an empty cavity, child nodes terminate into empty void leaves that calculate leaf relative distance to the bounding box of nearest data points:

$$d_{\text{rel}}(x) = \frac{1}{2^D} \sum_{k=1}^{2^D} \frac{\Vert x - p_k \Vert}{\delta_{\text{nominal}}}$$

This cleanly exposes voids as high-anomaly regions without requiring external k-NN searches.

| Threshold | Concentric Rings (`2circle.csv`) | Complex Teapot (`complex2d.csv`) |
|:---|:---:|:---:|
| **Training Data** | ![2circle raw](pics/geif/2circle_raw.png) | ![complex2d raw](pics/geif/complex2d_raw.png) |
| **$T = 0.00$** (Continuous Depth) | ![2circle T0](pics/geif/2circle_T0.png) | ![complex2d T0](pics/geif/complex2d_T0.png) |
| **$T = 0.45$** (Manifold Separation) | ![2circle T45](pics/geif/2circle_T45.png) | ![complex2d T45](pics/geif/complex2d_T45.png) |
| **$T = 0.50$** (Tight Inlier Envelope) | ![2circle T50](pics/geif/2circle_T50.png) | ![complex2d T50](pics/geif/complex2d_T50.png) |

### Analysis of Complex 2D with Bubble:
- **Donut Hole (Cavity)**: The interior void at center $(5107.55, 0.9887)$ is cleanly detected as anomalous (score $0.4679$, exceeding the nominal inlier threshold).
- **Needle Spikes**: The narrow vertical and diagonal needle projections maintain connected inlier envelopes.
- **Isolated Cluster**: The compact Gaussian spot in the top-left quadrant is cleanly resolved as a distinct inlier island.
- **Banana Arc**: The curved crescent manifold on the lower right is crisply traced without merging into the main body.

---

## 5. Extreme Aspect Ratio Invariance (5000:1 Disparity) (Bubble Engine)

When features possess vastly different physical units (e.g., milliseconds vs. packet bytes, or coordinates spanning $[1000, 9000]$ vs. $[0.2, 1.8]$ in `complex2d.csv` and $[5, 120]$ vs. $[10^5, 2 \times 10^6]$ in `Wtest.csv`), naive Euclidean distance degenerates into 1D vertical slicing.

GEIF's **Bubble** algorithm natively enforces scale invariance by evaluating normalized squared Euclidean distances weighted by regularized feature spans:

$$\Vert x - c \Vert_{\text{scaled}}^2 = \sum_{j=1}^D \left(\frac{x_j - c_j}{\text{span}_j}\right)^2$$

Because all dimensions contribute isotropically to the radial partitioning, the splitting spheres naturally conform to the true cluster shape regardless of numerical magnitude.

| Metric | `Wtest.csv` (Span: $X \approx 110, \; Y \approx 1,900,000$) |
|:---|:---:|
| **Raw Training Samples** | ![Wtest raw](pics/geif/Wtest_raw.png) |
| **$T = 0.00$ (Continuous Landscape)** | ![Wtest T0](pics/geif/Wtest_T0.png) |
| **$T = 0.50$ (Inlier Envelope)** | ![Wtest T50](pics/geif/Wtest_T50.png) |

**Result**: The narrow diagonal linear band is cleanly enveloped with zero axis distortion and zero configuration overhead.

---

## 6. Outer Space Continuum & Stadium Metric (Bubble Engine)

A known weakness of tree-based partitioning in unbounded Euclidean space is that distant points outside the training bounding box receive arbitrary scores based on whatever leaf boundaries happen to extend outwards.

In GEIF's **Bubble** engine, outer space points calculate the exact Euclidean distance $d_{\text{out}}$ to the training envelope bounding box:

$$d_{\text{out}}(x) = \sqrt{\sum_{j=1}^D \left(\frac{\max\left(0, \min_j - x_j, x_j - \max_j\right)}{\text{span}_j}\right)^2}$$

As $d_{\text{out}} > 0$, the anomaly score decays asymptotically towards 1.0 without boundary saturation:

$$s_{\text{final}}(x) = 1.0 - (1.0 - s_{\text{tree}}(x)) \cdot \exp(-0.10 \cdot d_{\text{out}}(x))$$

This creates rounded, continuous "stadium" equi-distance shells:

| Wide Area Evaluation ($T = 0.50$) | Deep Outer Space Perimeter ($T = 0.85$) |
|:---:|:---:|
| ![Outer T50](pics/geif/complex2d_outer_T50.png) | ![Outer T85](pics/geif/complex2d_outer_T85.png) |

### Key Properties:
- **No Starburst Rays**: Decision boundaries remain strictly convex and smoothly rounded in outer space.
- **Asymptotic Convergence to 1.0**: As points travel further into open space, their anomaly score monotonically approaches $1.000000$ without ever exceeding it.

---

## 7. Generating These Heatmaps

To train a model and generate population drift test grids:

```bash
# 1. Train model from dataset (Bubble algorithm by default)
./bin/geif -l test/complex2d.csv -w model_complex.json -B bubble -t 100 -s 256

# 2. Inspect forest summary
./bin/geif -r model_complex.json -q

# 3. Generate test grid with RGB colors for gnuplot visualization
./bin/geif -l test/complex2d.csv -T 0.1 -i 150 -O 0 -p "%d,0x%x" -B bubble -o test/plot_complex2d.csv

# 4. Render plot with gnuplot
gnuplot -e "
  set datafile separator ',';
  set terminal pngcairo size 800,800 enhanced font 'Helvetica,10';
  set output 'test/complex2d_bubble.png';
  plot 'test/plot_complex2d.csv' using 1:2:3 with points pt 7 ps 0.4 lc rgb variable notitle;
"
```
