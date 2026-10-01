# Beyond EIF: Fundamental Algorithmic Evolution for Isolation Forests

**Author / Maintainer:** Timo Savinen (AI-assisted)

---

## 1. Executive Summary & Context

The Extended Isolation Forest (EIF, Hariri et al., 2019) was proposed to solve the axis-aligned bias of Liu et al.'s original Isolation Forest (2008) by using random linear hyperplanes. However, real-world deployment reveals that both algorithms suffer from severe theoretical limitations:

1. **The Infinite Hyperplane Paradox**: Flat hyperplanes extend to $\pm\infty$, arbitrarily slicing empty space millions of units away and causing radial starburst artifacts.
2. **The Density Bias**: Discrete path length (tree depth) conflates sample count with spatial density. Dense clusters require many cuts to isolate, while diffuse outliers can mimic cluster depths.
3. **Topological Blindness**: Infinite linear cuts struggle to resolve non-convex geometries (cavities, concentric donuts, winding canals) without bridging empty voids.
4. **Dimension Magnitude Distortion**: Pure isotropic Gaussian normals assume unit variance across all features, failing when attributes have unequal physical units.

In `ceif`, practical engineering tackled each of these flaws through dedicated mechanisms:
- `auto_weigth` (dimension scaling)
- Pairwise sample interpolation for $p$ with quadratic depth decay margin
- `NEAREST` distance scaling at leaves
- Exterior Euclidean distance decay in deep space
- Structural Zero Kelvin deepest-leaf minimum score calibration

While highly effective, these mechanisms were initially perceived as "guardrails" around the core EIF tree. The `geif` project was built to test whether these properties could emerge **naturally from first principles** in a next-generation geometric forest. 

Crucially, **empirical testing in `geif` has now validated or invalidated several key theoretical hypotheses**. Notably, while outer stadium decay and continuous metric depth proved highly successful, **data-driven Voronoi bisector splitting did not yield better results** than isotropic Gaussian normals with sample-anchored projections. 

This paper synthesizes these empirical findings, explains why pure Voronoi bisectors degrade ensemble performance, documents the current state-of-the-art architecture, and outlines the next set of fertile geometric theories to test.

---

## 2. Deconstructing the Flaws of Classic EIF

```
Classic EIF Model:
      Random Gaussian Normal n ~ N(0, I)
                 +
      Uniform Intercept p ~ U[min, max]
                 ↓
      Infinite Hyperplane: (x - p) · n = 0
```

| Classic EIF Assumption | Real-World Failure | Why the "Trick" was Needed |
|---|---|---|
| **Hyperplanes extend to $\infty$** | A cut dividing two local points slices through empty space at $X = 80,000$. | Needed exterior distance decay to force outer monotonicity. |
| **Normal vector $n$ is independent of data** | In sparse or manifold data, a random normal cuts across empty voids or misses high-variance axes. | Needed pairwise $(x_2 - x_1)$ interpolation for $p$ and coordinate scaling. |
| **Path length = integer edge hops** | A step of distance 0.001 in a dense core counts the same as a step of distance 10,000 across a void. | Needed `NEAREST` relative distance scaling. |
| **Linear separating primitives** | Surrounding a circular void (donut hole) requires 6–10 straight cuts, bridging clusters in between. | Needed leaf-level density adjustments. |

---

## 3. Four Core Algorithmic Evolutions: Theory vs. Empirical Findings

### Evolution 1: Bounded-Cell Partitioning & Stadium Outer Space
**The Concept:** Instead of an infinite hyperplane in $\mathbb{R}^D$, every tree node represents a **bounded convex polytope** (or oriented bounding box) $C_v \subset \mathbb{R}^D$.

- At the root node, $C_{\text{root}}$ is the calibrated bounding envelope of the training data:

$$
C_{\text{root}} = [\min_j - \text{margin}_j, \; \max_j + \text{margin}_j]
$$

- When a node splits, hyperplane $H$ partitions cell $C_v$ into:

$$
C_{\text{left}} = C_v \cap H^-, \quad C_{\text{right}} = C_v \cap H^+
$$

#### Empirical Validation in GEIF & CEIF:
In practice, full polytope mesh storage in memory during inference is computationally prohibitive ($O(N \cdot D)$ facets per cell). However, the **mathematical essence of bounded partitioning was successfully achieved** via two unified mechanisms:
1. **Euclidean Stadium Outer Decay:** Any point outside the bounding envelope $C_{\text{root}}$ bypasses internal tree traversal artifacts and decays exponentially toward 1.0 based on its true Euclidean distance $d_{\text{out}}$ to the data hull:

$$
S(x) = 1.0 - (1.0 - S_{\text{edge}}) e^{-\beta \cdot d_{\text{out}}}
$$

2. **Zero Kelvin Lower Bound Calibration:** By traversing all trees to derive the theoretical maximum tree depth $\bar{h}_{\max}$, the minimum score is pinned to the structural lower bound:

$$
S_{\min} = 2^{-\bar{h}_{\max} / c}
$$

eliminating artificial score distortion while preserving headroom for inliers.

---

### Evolution 2: Voronoi Bisector Splitting vs. Empirical Findings

#### The Initial Theoretical Hypothesis:
Rather than generating an independent random normal $n \sim \mathcal{N}(0, I)$ and an intercept $p$, generate splits directly as **Voronoi perpendicular bisectors** between random sample pairs $(x_a, x_b) \in S_{\text{node}}$:

$$
n = \frac{x_b - x_a}{\Vert x_b - x_a \Vert}, \quad p = \frac{x_a + x_b}{2}
$$

The theoretical appeal was intuitive: every cut separates at least two real data points, cuts automatically align with the principal variance of local clusters, and dimension scaling seemed naturally absorbed.

#### Empirical Results from GEIF Testing:
Extensive testing across synthetic (`2blob`, `complex2d`) and real-world benchmark datasets demonstrated that **pure Voronoi bisector splitting does NOT yield better results than CEIF's sample-anchored Gaussian hyperplanes**, and in several key metrics performs noticeably worse. In commit `68582b0`, GEIF replaced Voronoi bisectors with isotropic Gaussian normals.

#### Why Voronoi Bisectors Failed (Root Causes):

1. **Ensemble Diversity Collapse (Angular Starvation):**
   The mathematical power of an Isolation Forest relies heavily on **high spherical angular entropy**—having hundreds of trees slicing the feature space from every continuous angle. 
   When split normals are constrained strictly to sample difference vectors $n = x_b - x_a$, the cuts become heavily correlated with the internal chord directions of clusters. Instead of isotropic multi-angle carving, trees generate repetitive, nearly parallel cuts. This loss of ensemble diversity reduces the forest's ability to smoothly approximate curved or non-convex boundaries.

2. **High-Dimensional Chord Degeneracy:**
   In higher dimensions ($D > 3$), due to the concentration of distances, pairwise difference vectors between randomly chosen samples become noisy and poorly conditioned. A single outlier or edge sample in a node frequently gets paired with a cluster core sample, producing an aggressive cut that isolates the outlier prematurely but slices the rest of the cluster at an unfavorable angle.

3. **Polygonal Facet Artifacts vs. Smooth Probability Contours:**
   Because Voronoi bisectors are pinned to discrete sample pairs, decision contours exhibit jagged, polygonal facet edges with sharp vertices. In contrast, Gaussian cuts produce smooth, continuous, probabilistic density isolines when integrated across an ensemble.

#### The Winning Architecture: Data-Anchored Isotropic Gaussian Cuts
The superior solution that emerged from CEIF and GEIF is a hybrid approach:
- **Direction:** Draw $n \sim \mathcal{N}(0, I)$ isotropically (preserving $360^\circ$ continuous angular entropy).
- **Placement:** Anchor $p$ to actual sample projections (using pairwise sample projection intervals $[z_1, z_2]$ and coordinate stretching).
This completely eliminates blind cuts into infinite voids without sacrificing spherical ensemble diversity.

---

#### Evolution 3: Continuous / Metric Path Length (Solving the Density Bias)
**The Concept:** In classic iForest, every split edge adds exactly $+1$ to path length, regardless of physical scale. In a continuous geometric forest, edge traversal accumulates a **metric distance weight**:

$$
\Delta h = \frac{\text{dist}(x, \text{boundary})}{\text{scale}_{\text{node}}}
$$

or

$$
\Delta h = \frac{\text{Volume}(C_{\text{child}})}{\text{Volume}(C_{\text{parent}})}
$$

#### Empirical Validation in GEIF:
GEIF validated this principle at the leaf level using a **Cauchy-Lorentz relative distance kernel**:
When a query point arrives at a leaf node containing samples $s_1 \dots s_k$, it evaluates its minimum Euclidean distance to the nearest leaf sample:

$$
\text{dist}_{\min} = \min_{i} \Vert x - s_i \Vert
$$

$$
\text{rel-dist} = \frac{\text{dist}_{\min}}{\text{dist}_{\text{avg}}} + \text{MIN-REL-DIST}
$$

The effective sample count $n$ is modulated by the spatial density:

$$
n' = \frac{n}{\text{rel-dist}}
$$

and the leaf contributes $c(n')$ to the depth sum. This prevents isolated points falling into large, sparse leaves from mimicking the score of dense cluster cores.

---

### Evolution 4: Quadratic & Spherical Primitives (Native Cavity & Void Resolution)
**The Concept:** Linear hyperplanes are degree-1 polynomials. Non-convex topologies (donuts, crescent canals, interlocking rings) require degree-2 primitives:

A split can choose between:
1. **A Linear Hyperplane**: $(x - p) \cdot n = 0$ (for separating distinct clusters).
2. **A Hyperspherical Bubble**: $\Vert x - c \Vert^2 \le R^2$ (for enclosing clusters or isolating central voids).

```
        Linear Split (EIF)                     Spherical Bubble Split
         \                                             . - ~ - .
          \     Data Cluster                         :     c     :  (Cluster or Void
           \                                          .  (R)   .     isolated in 1 cut!)
            \                                            ' - ~ - '
```

- **Algorithmic Implication:**
  - A central cavity (such as the donut hole in `complex2d`) can be isolated in a **single spherical split**, rather than requiring dozens of intersecting planar cuts.
  - Completely eliminates the "bridging effect" where linear planes accidentally join two disconnected clusters across a central void.

---

## 4. Architectural Comparison: EIF vs. CEIF vs. GEIF

| Architectural Dimension | Classic EIF (Hariri 2019) | CEIF (Engineered SOTA) | GEIF (Pure Bubble Forest - Theory 1) |
|---|---|---|---|
| **Split Primitive** | Infinite flat hyperplane | Linear hyperplane + pairwise $p$ | Finite Hyperspherical Bubble $\Vert x - c \Vert^2 \le R^2$ |
| **Normal Vector / Splits** | Isotropic Gaussian $\mathcal{N}(0, I)$ | Stratified / Isotropic + `auto_weigth` | Center $c \in S_{\text{node}}$, radius $R \sim \mathcal{U}[r\text{-min}, r\text{-max}]$ |
| **Voronoi Bisector Cuts** | Not evaluated | Tested & rejected (loss of angular diversity) | Tested & rejected (hyperplanes extend to $\pm\infty$) |
| **Coordinate Normalization**| None (raw coords) | Per-dimension heuristic scaling | Full isotropic unit normalization $[0, 1]^D$ |
| **Void & Cavity Damping** | None (infinite rays) | `NEAREST` distance scaling at leaves | Continuous Cauchy-Lorentz leaf kernel $D(x)$ |
| **Outer Space Behavior** | Severe starburst rays & wedges | Euclidean exterior distance decay ($\beta = 0.10$) | **Naturally scaled to 1.0000** (compact bubble support) |
| **Lower Bound Calibration** | Clamped at arbitrary 0.0 | Structural Zero Kelvin leaf calibration | **Naturally scaled to 0.0000** ($I_{\max}$ inlier peak) |
| **Score Post-Processing** | Clamped $[0, 1]$ | Min-Max score remapping & stadium decay | **None (Base-2 exponential $S = 2^{-\bar{I} / I_{\text{bound}}}$)** |
| **Memory Footprint** | Complete node hierarchy | Lightweight sample reservoir / JSON schema | Ultralight nodes (no normal vectors needed) |

---

## 5. Theory 1: Pure Hyperspherical Bubble Forest with Empty Void Carving

### Core Mathematical Formulation
In Theory 1, `geif` discards flat hyperplanes completely in favor of **spatially compact hyperspherical bubbles**. Every tree node partitions data into:

$$
\text{Inside (Left): } \Vert x - c \Vert^2 \le R^2, \quad \text{Outside (Right): } \Vert x - c \Vert^2 > R^2
$$

All features are normalized into unit space $[0, 1]^D$:

$$
x'_j = \frac{x_j - \min_j}{\text{span}_j}
$$

### The Internal Cavity Challenge & Empty Void Carving
Standard hyperspheres are convex sets. When bubble centers are restricted solely to observed data points $s \in S_{\text{node}}$, any sphere encompassing samples on opposite sides of an internal topological cavity (such as a donut hole or crescent chasm) inevitably sweeps across the empty space between them.

To achieve robust topological resolution without planar bridging, GEIF incorporates **Multi-Trial Midpoint Void Probing** and **Capped Leaf Sizing**:
1. **Multi-Trial Candidate Midpoints:** During node splitting (when sample count $N \ge 3$), the tree evaluates up to $K = 4$ random candidate sample pairs $(s_a, s_b \in S_{\text{node}})$ and tests their midpoint:

$$
c = \frac{s_a + s_b}{2}
$$

2. **Void Detection Criterion:** The tree computes the nearest sample distance to each candidate midpoint:

$$
r_{\min} = \min_{s \in S_{\text{node}}} \Vert s - c \Vert
$$

3. **Empty Void Bubble Carving:** If $r_{\min} \ge 2.5 \times \text{NN-avg}$ (where $\text{NN-avg}$ is the ensemble's average nearest-neighbor distance), the node selects the candidate with the largest empty clearance and carves an empty bubble of radius $R = 0.95 \cdot r_{\min}$.
   - The inside child (`left_child`) is designated as an **Empty Void Leaf** (`flags = 1`, `sample_count = 0`). Any point falling inside this bubble has zero inlier probability: $I(x) = 0.0 \implies S(x) = 1.0$.
   - The outside child (`right_child`) receives all data points in $S_{\text{node}}$ and continues recursive tree construction at $\text{depth} + 1$.
   - Multi-trial probing ($K = 4$) ensures narrow canals and curved chasms between distinct clusters are reliably carved before regular splits divide the clusters into disjoint subtrees.

### Unified Continuous Inlier Probability Function
During traversal, every tree evaluates a continuous inlier metric combining hierarchical depth and local leaf sample density:

$$
I(x) = W(x) \cdot D(x)
$$

where:
1. **Hierarchical Isolation Depth Weight:**

$$
W(x) = \frac{\text{depth}(x)}{H_{\max}} \in [0, 1]
$$

   measuring the degree of structural nested containment within the ensemble.
2. **Cauchy-Lorentz Local Density Kernel with Capped Leaf Sizing:**

$$
D(x) = \frac{1}{1 + (d_{\text{leaf}} / \sigma_{\text{leaf}})^2} \in [0, 1]
$$

   where distance:

$$
d_{\text{leaf}} = \Vert x' - s_{\text{leaf}} \Vert
$$

   is the Euclidean distance to the representative training sample in the leaf, and $\sigma_{\text{leaf}}$ is calibrated to local data spacing:

$$
\sigma_{\text{leaf}} = \min(r_{\text{pairwise}}, 1.5 \cdot \text{NN-avg})
$$

   *Mathematical Rationale for Capped Sizing:* If $\sigma_{\text{leaf}}$ were allowed to expand to the unconstrained pairwise distance between points that could not be split further (e.g. at maximum tree depth across a canal), the leaf would cast a massive inlier halo across the void, causing inlier leakage ("bridging"). Capping $\sigma_{\text{leaf}} \le 1.5 \cdot \text{NN-avg}$ ensures inlier halos cover local Voronoi cells without bridging across inter-cluster gaps. Conversely, setting $\sigma_{\text{leaf}} \ge 1.0 \cdot \text{NN-avg}$ ensures cluster cores remain solidly contiguous without artificial internal holes.
3. **Void Leaf Short-Circuit:** If traversal terminates in an Empty Void Leaf, $I(x) = 0.0$ immediately.

### Calibrated Natural Anomaly Scoring (Base-2 Exponential)
Classic Isolation Forest mapped tree depth to anomaly score via base-2 exponential normalization: $S(x) = 2^{-E(h)/c(n)}$, anchoring average depth to an anomaly score of $0.5000$.

GEIF applies this principle to continuous inlier probability $\bar{I}(x)$, eliminating the linear peak-normalization problem (where boundary inliers were pushed up to $0.84$ due to normalization by an extreme single peak $I_{\max}$):

$$
S(x) = 2^{-\frac{\bar{I}(x)}{I_{\text{bound}}}}
$$

where:

$$
\bar{I}(x) = \frac{1}{T} \sum_{t=1}^T I_t(x)
$$

and $I_{\text{bound}}$ is the calibrated inlier boundary (calibrated to the 95th percentile inlier coverage in the training pool, i.e., 5th percentile of $I$):
- **Cluster Core ($\bar{I} \gg I_{\text{bound}}$):** $S(x) \to 0.00 - 0.05$.
- **Cluster Median ($\bar{I} \approx 2 \cdot I_{\text{bound}}$):** $S(x) \approx 0.15 - 0.25$.
- **Cluster Boundary ($\bar{I} = I_{\text{bound}}$):** $S(x) = 2^{-1} = \mathbf{0.5000}$ exactly.
- **Canal Voids & Mild Outliers ($\bar{I} < I_{\text{bound}}$):** $S(x) \in [0.60, 0.75]$.
- **Internal Cavities & Outer Space ($\bar{I} \to 0.0$):** $S(x) = 2^0 \to \mathbf{1.0000}$ exactly.

### Empirical Validation Results
- **Cluster Cores (`2blob` & `complex2d`):** Reach natural minimum scores $S(x) \in [0.03, 0.15]$.
- **Cluster Boundary:** Anchored precisely at $S(x) = 0.5000$ (95th percentile of inliers).
- **Internal Topological Voids (`complex2d` Donut Hole & Crescent Canal):** Attain anomaly scores $S(x) \in [0.68, 0.94]$, scoring cleanly above the $0.50$ decision boundary and rendering as completely coherent, continuous red/amber anomaly regions without any white bridging artifacts.
- **Outer Space (`2blob` & `complex2d`):** Because bubbles have finite spatial support, for any point in deep space $d_{\text{leaf}} \to \infty \implies D(x) \to 0 \implies I(x) \to 0 \implies S(x) \to 1.0000$ exactly, eliminating all starburst rays.
- **Visual Validation:** Verified via high-resolution 2D test grids (`test/pic.png`), confirming that both the central donut hole and the curved canal between the donut and crescent arch are cleanly carved and completely coherent.

---

## 6. Additional Theories to Test with GEIF

### Theory 2: Subspace Feature Bagging (High-Dimensional Sparsity)
- **Problem:** In high dimensions ($D \ge 10$), drawing an isotropic Gaussian normal with non-zero weights in all dimensions dilutes anomalous signals across uninformative noise features (the curse of dimensionality).
- **Hypothesis:** Implement node-level or tree-level subspace sampling: for each split, randomly select a subset of $k \ll D$ active dimensions (e.g., $k = \lceil\sqrt{D}\rceil$ or $k = 2, 3$) and generate the normal vector only in that subspace.
- **Expected Outcome:** Significantly higher detection sensitivity on sparse, high-dimensional datasets and tabular feature spaces with noisy attributes.

---

### Theory 3: Anisotropic / Mahalanobis Ellipsoidal Cuts
- **Problem:** Real-world metrics and telemetry features frequently exhibit strong covariance (e.g. CPU vs. Memory usage, latency vs. throughput). Spherical cuts slice awkwardly across diagonal correlations.
- **Hypothesis:** Instead of isotropic hyperspheres, use local covariance-weighted ellipsoidal splits:

$$
(x - c)^T \Sigma^{-1} (x - c) \le R^2
$$

where $\Sigma$ is approximated using diagonal feature variance or pairwise two-sample difference covariance.
- **Expected Outcome:** Trees adapt tightly to elongated correlation manifolds with far fewer cuts.

---

### Theory 4: Continuous Path Integration (Intermediate Void Traversal Weighting)
- **Problem:** Currently, internal tree edges add a uniform $+1.0$ hop to depth, and continuous metric distance is only evaluated at the final leaf.
- **Hypothesis:** Accumulate distance-weighted increments $\Delta h$ along internal split edges:

$$
\Delta h_i = 1.0 + \alpha \cdot \frac{\text{dist}(x, H)}{\text{margin}_{\text{node}}}
$$

Points traversing through large empty internal gaps between clusters accumulate larger depth reductions immediately, without waiting to reach a leaf.
- **Expected Outcome:** Faster discrimination of internal cavities and multi-modal cluster boundaries.

---

### Theory 5: Dynamic Ensemble Temperature / Multi-Scale Headroom
- **Problem:** The Zero Kelvin calibration ($S_0 = 2^{-\bar{h}_{\max}/c}$) establishes an absolute theoretical lower bound, but operational alerting pipelines often require tunable discrimination between core inliers and boundary samples.
- **Hypothesis:** Introduce a continuous temperature / sharpness exponent $\tau$:

$$
S(x) = 2^{-(h(x) / c)^\tau}
$$

Adjusting $\tau$ modulates the steepness of the anomaly knee without altering tree structures or model weights.
- **Expected Outcome:** Flexible operational tuning for production alert thresholds (e.g. centering the alert knee cleanly around $0.65$).

---

### Theory 6: Streaming Reservoir Aging & Time-Decayed Anomaly Scoring
- **Problem:** Production workloads undergo gradual seasonal drift; obsolete patterns in the reservoir can dilute anomaly detection for emerging distributions.
- **Hypothesis:** Augment the reservoir with exponential time decay weights $w_i = e^{-\lambda (t_{\text{now}} - t_i)}$. When evaluating leaf sample density $c(n')$, weight samples by their recency.
- **Expected Outcome:** Continuous self-adapting anomaly detection on streaming, non-stationary time series without retraining from scratch.
