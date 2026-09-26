# GEIF: Algorithmic Challenges, Edge Cases & Solutions

This document identifies and resolves the five core algorithmic subtleties and mathematical edge cases in the **GEIF (Geometric Extended Isolation Forest)** architecture.

---

## 1. Challenge 1: Constant Features & Zero-Variance Dimensions ($\text{span}_j = 0$)

### 1.1 The Issue
In production datasets (e.g. constant HTTP status codes, locked sensor channels, or one-hot flags that never fire):
$$\max_{x \in X} x_j = \min_{x \in X} x_j \implies \text{span}_j = 0$$

If GEIF naively calculates the pre-baked normal:
$$n_j = \frac{B_j - A_j}{\text{span}_j^2} \implies \frac{0}{0} = \text{NaN}$$

### 1.2 The Solution
1. **Dimension Health Masking (Train Time):**
   During dataset inspection, GEIF marks any dimension with $\text{span}_j < 10^{-12}$ as **Degenerate (Zero-Variance)**.
2. **Zero-Weight Normal:**
   For degenerate dimensions, $n_j$ is forced to $0.000000$. It consumes zero weight in dot products.
3. **Outer Space Handling:**
   If a test point arrives with a different value on a zero-variance feature ($x_j \ne \text{mean}_j$), it represents an instantaneous structural anomaly. The outer distance evaluates with a designated nominal scale (or user epsilon) rather than dividing by zero.

---

## 2. Challenge 2: Duplicate Points & Inseparable Node Leaves ($A == B$)

### 2.1 The Issue
Real-world data contains exact duplicate rows (identical telemetry packets, rounded integer counters). If a node picks two samples $A$ and $B$ that have identical coordinates ($A_j = B_j \; \forall j$):
$$B - A = \mathbf{0} \implies n = \mathbf{0}$$
The hyperplane $x \cdot \mathbf{0} = 0$ is undefined and cannot partition space.

### 2.2 The Solution
1. **Candidate Pair Sampling:**
   When selecting split generators, the algorithm attempts up to $K$ trials (default: $K=5$) to pick two distinct samples ($A \ne B$).
2. **Identical Subset Terminal Leaf:**
   If all samples in node $S_v$ are identical, the node stops immediately and forms an **Inseparable Leaf Node** with sample count $k = |S_v|$.
3. **Analytic Depth Contribution:**
   In classic iForest, an unresolved leaf with $k$ duplicates receives the harmonic path length $c(k) = 2(\ln(k-1) + 0.5772) - \frac{2(k-1)}{k}$. In GEIF, this contributes an equivalent analytic depth increment:
   $$H_{\text{leaf}} = H_{\text{current}} + \log_2(k) \cdot \delta_{\text{local}}$$

---

## 3. Challenge 3: Topological Cavities & The Non-Convex Void Problem (The Donut Hole)

### 3.1 The Issue
Linear hyperplanes are flat surfaces that extend across the entire data bounds. If data forms an annular ring (a donut with a hollow center, as in `complex2d.csv`):
```text
           . - ~ - ~ - .
        :                 :
       :     ( HOLE )      :  ← Points here are in empty space!
        :                 :
           ' - ~ - ~ - '
```
A straight bisector cut between a point on the North ring and a point on the South ring passes directly through the empty center. A query point landing in the center hole might not be sliced off if cuts only separate opposite sides.

### 3.2 The Solution: Residual Cell Distance (Leaf Proximity Regularization)
While tree traversal navigates the query point into its nearest Voronoi cell (associated with leaf sample $P_{\text{leaf}}$), GEIF evaluates the **residual normalized distance** to the cell generator:
$$d_{\text{residual}}(x) = \|\tilde{x} - \tilde{P}_{\text{leaf}}\| = \sqrt{\sum_{j=1}^D \left(\frac{x_j - P_{\text{leaf}, j}}{\text{span}_j}\right)^2}$$

* **Points inside the ring manifold:** $d_{\text{residual}} \le r_{\text{cluster}}$ (very close to their leaf sample). Depth remains high $\implies$ Score nominal.
* **Points in the donut hole:** $d_{\text{residual}} \gg r_{\text{cluster}}$ (far from any training point).
* **The Void Attenuation:**
  $$H_{\text{final}}(x) = H_{\text{tree}}(x) \cdot \frac{1}{1 + \left(\frac{d_{\text{residual}}(x)}{R_{\text{nominal}}}\right)^2}$$
  Points in the empty interior cavity have their metric depth attenuated, correctly revealing the donut hole as an anomaly with zero starburst artifacts.

---

## 4. Challenge 4: Continuous Metric Path Length vs. Integer Hop Depth

### 4.1 The Issue: The Classic Density Bias
In classic iForest, depth is discrete: each node branch adds $+1$ integer hop.
* A tight cluster of 50 samples isolates in $\approx \log_2(50) \approx 5.6$ hops.
* A diffuse cluster of 500 samples isolates in $\approx \log_2(500) \approx 9.0$ hops.
Classic iForest falsely scores points in the dense 50-sample cluster as *more anomalous* than points in the sparse 500-sample cloud!

### 4.2 The Solution: Density-Compensated Metric Depth
In GEIF, each split is defined by generator distance:
$$\delta = \|\tilde{B} - \tilde{A}\|$$
* In dense regions: samples are packed close together $\implies \delta$ is small.
* In sparse regions: samples are far apart $\implies \delta$ is large.

Instead of adding a flat $+1$, each branch adds a continuous metric increment inversely proportional to local scale:
$$\Delta H = \frac{1}{\delta^\gamma} \quad (\text{or } \Delta H = -\ln(\delta))$$

Where $\gamma \in [0.5, 1.0]$.
* Navigating through a dense cluster accumulates large metric depth $H$ very quickly, properly identifying tight clusters as the most nominal inliers.
* Sparse regions accumulate depth slowly, naturally registering higher anomaly scores without needing post-hoc cluster density balancing.

---

## 5. Challenge 5: Extreme Tree Imbalance & Skewed Splits

### 5.1 The Issue
If two random points $A$ and $B$ are drawn such that $A$ is an isolated outlier on the fringe and $B$ is in the dense core:
* The left branch gets 1 sample.
* The right branch gets 255 samples.
While this is desirable for isolating an outlier, if this happens repeatedly inside nominal data, trees could degrade into $O(N)$ depth linked lists.

### 5.2 The Solution
1. **Guaranteed Depth Cap:**
   Forest construction enforces a hard maximum depth:
   $$h_{\max} = 2 \cdot \lceil \log_2(\psi) \rceil$$
   For standard sub-sample size $\psi = 256$, $h_{\max} = 16$.
2. **Sub-Sample Reservoir Isolation ($\psi = 256$):**
   Because each tree is trained on an independently drawn reservoir sub-sample of $\psi = 256$ points, catastrophic unbalanced chains in one tree are statistically smoothed out across the ensemble of $T = 100$ trees.
