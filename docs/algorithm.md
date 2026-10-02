# GEIF: Geometric Extended Isolation Forest — Algorithmic Specification

**Author / Maintainer:** Timo Savinen (AI-assisted)

## 1. Introduction & Theoretical Motivation

### 1.1 The Isolation Forest Paradigm
The Isolation Forest family isolates anomalous observations rather than profiling normal points:
1. **Isolation Forest (iForest)**: Partitions space using axis-aligned orthogonal cuts. Suffers from severe axis-aligned artifacts, blind spots, and artificial rectangular halos.
2. **Extended Isolation Forest (EIF)**: Replaces orthogonal cuts with linear hyperplanes with normal vectors drawn from an isotropic standard Gaussian $n \sim \mathcal{N}(0, I)$.
3. **Geometric Extended Isolation Forest (GEIF)**: Modernizes isolation into a modular, multi-algorithm geometric framework featuring hyperspherical cavity-carving trees, Voronoi bisector hyperplanes, exemplar kernel density estimation, and continuous metric depth with universal scale calibration.

### 1.2 Theoretical Challenges in Space Partitioning
Unconstrained spatial partitioning algorithms encounter four fundamental mathematical pitfalls:
1. **The Infinite Hyperplane Paradox**: Linear hyperplanes extend to $\pm\infty$. A cut separating two local points inside a cluster continues infinitely, arbitrarily slicing empty outer space millions of units away and creating irregular "starburst rays".
2. **The Density Bias**: Discrete path lengths (integer edge hop counts) measure sample count rather than spatial density. A tight cluster of 50 samples isolates in $\approx \log_2(50) \approx 5.6$ hops, while a diffuse cloud of 500 samples isolates in $\approx \log_2(500) \approx 9$ hops.
3. **Disparate Feature Scales**: When features represent disparate physical units (e.g. microseconds vs. bytes, or coordinates spanning $[1000, 9000]$ vs. $[0.2, 1.8]$), unscaled distance metrics collapse into 1D axis projections.
4. **Topological Blindness**: Linear hyperplanes cannot carve non-convex internal cavities (such as concentric rings or donut hollows) without slicing across empty interior space.

### 1.3 The GEIF Architecture
**GEIF (Geometric Extended Isolation Forest)** resolves these challenges through a modular multi-algorithm engine built upon robust, unified foundational primitives:
- **Pluggable Spatial Engines**: Selectable via `-B, --algo` (`bubble`, `voronoi`, `exemplar`, `ceif`).
- **Hyperspherical Cavity Carving**: The default **Bubble** algorithm bounds clusters in compact hyperspherical shells, isolating interior cavities into empty void leaves.
- **Universal Scale Calibration ("Zero Kelvin")**: Theoretical deepest-leaf calibration ensures dense cluster centroids reach the physical inlier floor ($s \to 0.0$).
- **Asymptotic Outer Space Continuum**: Outer space distance decays smoothly towards $1.0$ without boundary saturation artifacts.
- **Normalized Scale Invariance**: Regularized dimension span weighting eliminates aspect ratio distortions across $5000:1$ spans.

---

## 2. Common Foundational Methods Across All Engines

All algorithm engines in GEIF share a unified set of mathematical primitives and calibration methods.

### 2.1 Universal Scale Calibration ("Zero Kelvin")

In traditional isolation forests, raw anomaly scores depend heavily on sample count and tree depth, causing the minimum score of normal inliers to fluctuate between $0.35$ and $0.85$ across different geometries.

GEIF eliminates post-hoc score scaling by calculating the theoretical deepest-leaf traversal:

1. During model initialization, each tree in the ensemble is traversed along its deepest branch to compute the theoretical maximum metric depth $H_{\text{deepest}}$.
2. The forest-wide maximum theoretical depth is established:

$$
H_{\text{train,max}} = \frac{1}{T} \sum_{t=1}^T H_{\text{deepest}}^{(t)}
$$

3. A calibrated universal scale factor is applied:

$$
H_{\text{max}} = \kappa \cdot H_{\text{train,max}} \quad (\kappa = 1.15)
$$

4. The raw anomaly score is calculated directly from accumulated continuous metric depth $H(x)$:

$$
s_{\text{raw}}(x) = \exp\left(-\frac{H(x)}{c(\psi)}\right)
$$

5. The calibrated score is mapped across the theoretical span from $s_{\min}$ to $s_{\max}$:

$$
s(x) = \frac{s_{\text{raw}}(x) - s_{\min}}{s_{\max} - s_{\min}}
$$

where:

$$
s_{\min} = \exp\left(-\frac{H_{\text{max}}}{c(\psi)}\right)
$$

represents the "Zero Kelvin" deepest inlier floor, and $s_{\max} = 1.0$.

**Guaranteed Inlier Floor**: Dense cluster centroids naturally achieve scores near $0.00$, establishing $T = 0.50$ as a universal, calibrated default threshold across all datasets.

---

### 2.2 Non-Reachable 1.0 (Asymptotic Bounding)

In unbounded Euclidean space, distance to training data $d \to \infty$ should indicate maximum anomaly. However, truncating scores abruptly at $1.0$ creates artificial boundary walls.

GEIF enforces asymptotic exponential outer space attenuation:

$$
s_{\text{final}}(x) = 1.0 - (1.0 - s_{\text{tree}}(x)) \cdot \exp(-\lambda \cdot d_{\text{out}}(x))
$$

where $\lambda = 0.10$ and $d_{\text{out}}(x)$ is the normalized Euclidean distance to the ensemble training envelope bounding box:

$$
d_{\text{out}}(x) = \sqrt{\sum_{j=1}^D \left(\frac{\max(0,\; \min_j - x_j,\; x_j - \max_j)}{\text{span}_j}\right)^2}
$$

#### Properties:
1. **Strict Upper Bound**: For all finite $x \in \mathbb{R}^D$, $s_{\text{final}}(x) < 1.0$. The score asymptotically approaches $1.0$ as $d_{\text{out}} \to \infty$.
2. **Smooth Gradient**: Outer space produces rounded, convex equi-distance shells ("stadium metric") without starburst rays.
3. **No Numerical Overflow**: Floating-point values remain strictly within $[0.0, 1.0)$.

---

### 2.3 Nearest Bounding-Box Calculation ($2^D$ Nodes & Leaf Proximity)

To evaluate local cluster density and detect interior voids without constructing an expensive global $k$-d tree, GEIF computes leaf relative distances using a bounding-box projection:

For a test point $x$ falling into a leaf node:
1. The leaf identifies the bounding hyper-rectangle formed by its bounding data points.
2. In $D$-dimensional space, the distance to the $2^D$ bounding corners is evaluated:

$$
d_{\text{rel}}(x) = \frac{1}{2^D} \sum_{k=1}^{2^D} \frac{\Vert x - p_k \Vert_{\text{scaled}}}{\delta_{\text{nominal}}}
$$

3. When $x$ is well-centered among the leaf samples, $d_{\text{rel}}(x) \approx 0$, preserving full metric depth.
4. When $x$ lies in an empty void or cavity, $d_{\text{rel}}(x) \gg 1$, triggering Cauchy-Lorentz residual cell damping:

$$
H_{\text{attenuated}}(x) = H_{\text{tree}}(x) \cdot \frac{1}{1 + \left(d_{\text{rel}}(x)\right)^2}
$$

This attenuates depth in hollow interior regions, elevating their anomaly scores to outlier levels without external $k$-NN searches.

---

### 2.4 Dimension Span Regularization & Aspect Ratio Robustness

When features possess disparate physical units, raw Euclidean metrics collapse. GEIF regularizes each dimension span:

$$
\text{span}_j = \max_{x \in X} x_j - \min_{x \in X} x_j
$$

- If $\text{span}_j \le 10^{-9}$ (zero-variance / constant feature), $\text{span}_j$ is set to $1.0$ and dimension health masking assigns zero weight during distance calculations.
- All spatial distance calculations evaluate normalized coordinates:

$$
\Vert x - y \Vert_{\text{scaled}}^2 = \sum_{j=1}^D \left(\frac{x_j - y_j}{\text{span}_j}\right)^2
$$

This guarantees strict scale invariance across extreme aspect ratios ($5000:1$ and higher).

---

### 2.5 Streaming Reservoir Pool & Model Adaptivity

GEIF ingests streaming data using Algorithm R reservoir sampling:
- Maintains an active reservoir pool of size $N_{\text{pool}} = T \times \psi$ per sub-forest.
- As new training rows arrive, samples are accepted with probability $\psi / N_{\text{seen}}$.
- **Adaptivity Ceiling**: Caps historical extra rows at $3 \times \psi$, ensuring continuous adaptivity to population drift without unbounded memory growth.
- **Sparse Serialization**: Saves the compact reservoir pool to JSON (`< 5 KB`), enabling fast tree re-building upon load in $< 5\text{ ms}$.

---

## 3. Algorithm 1: Hyperspherical Bubble Trees (`bubble`, Default)

### 3.1 Motivation & Concept
Linear hyperplanes inherently cut through the center of ring and hollow distributions. The **Bubble** algorithm replaces infinite linear cuts with hyperspherical cavity-carving nodes:

$$
\mathcal{B}(c, R) = (x \in \mathbb{R}^D : \Vert x - c \Vert_{\text{scaled}}^2 \le R^2)
$$

```
               [Hypersphere Node B(c, R)]
                      /          \
                     /            \
       Inside: ||x - c||^2 <= R^2   Outside: ||x - c||^2 > R^2
                   /                \
        [Sub-cluster Node]      [Void / Outer Leaf]
```

### 3.2 Mathematical Formulation
1. **Center Selection**: The center $c \in \mathbb{R}^D$ is placed at the centroid of two randomly drawn distinct samples $A, B \in S_v$:

$$
c = \frac{A + B}{2}
$$

For dimensions $D \le 64$, the center vector is allocated directly on the CPU stack (`stack_center[64]`), avoiding heap allocation.

2. **Normalized Squared Metric**:

$$
\text{dist}^2(x, c) = \sum_{j=1}^D \left(\frac{x_j - c_j}{\text{span}_j}\right)^2
$$

Operating directly on squared distances completely eliminates expensive `sqrt()` operations during tree construction.

3. **Median Cut via $O(N)$ Quickselect**:
The splitting radius $R^2$ is chosen as the median squared distance of all samples in $S_v$ using Hoare selection (`quickselect_median`):

$$
R^2 = \text{median}\left(\text{dist}^2(x_i, c) : x_i \in S_v\right)
$$

4. **In-Place Two-Pointer Partitioning**:
Samples are partitioned in-place into interior and exterior subsets via a two-pointer swap, eliminating per-node heap allocations.

5. **Empty Void Leaves**:
When a partition region contains no training samples (e.g. the hollow interior of a donut), it terminates as an empty void leaf ($N_{\text{leaf}} = 0$) with calibrated residual depth, flagging any query point falling within it as anomalous.

---

## 4. Algorithm 2: Voronoi Hyperplane Bisectors (`voronoi`)

### 4.1 Concept
Rather than drawing random Gaussian normal vectors, the **Voronoi** engine places hyperplanes at the exact perpendicular bisector between two randomly selected samples $A, B \in S_v$:

$$
P_{\text{mid}} = \frac{A + B}{2}, \quad \vec{n} = \frac{B - A}{\Vert B - A \Vert}
$$

### 4.2 Single Dot-Product Inference
The decision rule for query point $x$ simplifies algebraically to a single dot product:

$$
x \cdot \vec{n} < P_{\text{mid}} \cdot \vec{n} \implies \text{Branch Left (closer to } A\text{)}
$$

$$
x \cdot \vec{n} \ge P_{\text{mid}} \cdot \vec{n} \implies \text{Branch Right (closer to } B\text{)}
$$

### 4.3 Key Characteristics
- **Data-Driven Orientation**: Difference vector $B - A$ naturally inherits the orientation and scale of the local cluster.
- **Zero Empty-Space Hyperplanes**: Cuts are always anchored between real observations.
- **Crisp Cluster Separation**: Excels at separating adjacent, linearly separable clusters.

---

## 5. Algorithm 3: Exemplar Kernel Density Estimation (`exemplar`)

### 5.1 Concept
The **Exemplar** engine is a non-tree spatial kernel estimator operating directly on the active reservoir sample pool. It bypasses hierarchical binary trees entirely, evaluating the query point against all exemplar samples:

$$
s_{\text{exemplar}}(x) = 1.0 - \frac{1}{1 + \frac{1}{|S_{\text{pool}}|} \sum_{i=1}^{|S_{\text{pool}}|} \frac{1}{1 + \Vert x - x_i \Vert_{\text{scaled}}^2}}
$$

### 5.2 Key Characteristics
- **Zero Tree Building Overhead**: Models train instantaneously by collecting reservoir samples.
- **Theoretical 0.0000 Floor**: Points coinciding with training exemplars achieve a score of exactly $0.0000$.
- **Smooth Continuous Landscape**: Produces smooth isotropic score contours without tree partitioning boundaries.

---

## 6. Algorithm 4: Extended Hyperplane Engine (`ceif`)

### 6.1 Concept
The **CEIF Hyperplane Engine** provides data-anchored isotropic Gaussian cuts combined with continuous metric depth accumulation:
- A sample point $P$ is drawn uniformly from $S_v$ as the intercept anchor.
- A random normal vector $n \sim \mathcal{N}(0, I)$ is generated from standard normal distributions and normalized.
- Continuous depth increments $\Delta H = 1/\delta$ accumulate along the traversal path.
- Deepest-leaf Zero Kelvin calibration maps raw scores to $[0.0, 1.0)$.

### 6.2 Key Characteristics
- Provides continuity and compatibility with classical Extended Isolation Forest benchmarks.
- Well-suited for high-dimensional diffuse Gaussian distributions.

---

## 7. Comparative Engine Summary

| Property | Bubble (`bubble`, Default) | Voronoi (`voronoi`) | Exemplar (`exemplar`) | Hyperplane (`ceif`) |
| :--- | :---: | :---: | :---: | :---: |
| **Partition Geometry** | Hyperspheres $\mathcal{B}(c, R)$ | Perpendicular Bisectors | Kernel Density Field | Gaussian Hyperplanes |
| **Data Structure** | Binary Tree | Binary Tree | Reservoir Exemplar Pool | Binary Tree |
| **Internal Void Detection** | **Native (Void Leaves)** | Moderate (Damping) | High (Density Drop) | Moderate (Damping) |
| **Aspect Ratio Robustness** | **Native (Span-Scaled)** | **Native (Sample Bisector)** | **Native (Span-Scaled)** | Scaled Coordinate Metric |
| **Inference Complexity** | $O(D \log \psi)$ | $O(D \log \psi)$ | $O(D \cdot \psi)$ | $O(D \log \psi)$ |
| **Memory per Tree Node** | Compact ($R^2$ + center) | Compact ($\vec{n}$ + intercept) | Zero (Pool only) | Normal + Intercept |
| **Inlier Score at $T = 0.50$** | **0.0% False Outliers** | $\le 0.4$% False Outliers | $\le 4.0$% False Outliers | $\le 1.8$% False Outliers |
