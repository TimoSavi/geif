# GEIF: Geometric Extended Isolation Forest — Algorithmic Specification

**Author / Maintainer:** Timo Savinen (AI-assisted)

## 1. Introduction & Theoretical Motivation

### 1.1 The Lineage of Isolation-Based Anomaly Detection
The Isolation Forest family isolates anomalous observations rather than profiling normal points:
1. **Isolation Forest (iForest, Liu et al., 2008)**: Partitions space using axis-aligned orthogonal cuts. Suffers from severe axis-aligned artifacts, blind spots, and artificial rectangular halos.
2. **Extended Isolation Forest (EIF, Hariri et al., 2019)**: Replaces orthogonal cuts with linear hyperplanes with normal vectors drawn from an isotropic standard Gaussian $n \sim \mathcal{N}(0, I)$.
3. **C-Extended Isolation Forest (CEIF, Savinen, 2021–2026)**: Introduced high-performance C architecture, streaming reservoir updates, pairwise $p$ interpolation with depth-decay margins, nearest-neighbor leaf density regularization (`NEAREST`), and exterior distance decay.

### 1.2 Theoretical Shortcomings of Classic EIF
Despite practical success, classic EIF is constrained by four mathematical flaws:

```
Classic EIF:
  Random Normal n ~ N(0, I) + Uniform Intercept p ∈ [min, max]
       ↓
  Infinite Hyperplane in R^D: (x - p) · n = 0
```

1. **The Infinite Hyperplane Paradox**: Hyperplanes extend to $\pm\infty$. A cut separating two local points inside a cluster continues infinitely, arbitrarily slicing empty outer space millions of units away. This creates irregular "starburst rays" and random outer wedges.
2. **The Density Bias**: Classic iForest models assume that path length (integer edge hops) directly correlates with density. However, a tight cluster of 50 samples isolates in $\approx \log_2(50) \approx 5.6$ hops, while a diffuse cloud of 500 samples isolates in $\approx \log_2(500) \approx 9$ hops. Dense cluster points falsely appear more anomalous than diffuse points.
3. **Isotropic Normal Distortion**: Drawing $n \sim \mathcal{N}(0, I)$ assumes all features have equal unit variance. When features represent disparate physical units (e.g. microseconds vs. bytes), Gaussian normals align arbitrarily and lose discriminative power.
4. **Topological Blindness**: Infinite linear hyperplanes cannot isolate non-convex topological cavities (e.g., donut voids, winding crescent canals) without intersecting and bridging disconnected clusters across empty space.

### 1.3 The GEIF Paradigm Shift
**GEIF (Geometric Extended Isolation Forest)** shifts isolation from blind, unguided hyperplane slicing to **data-driven geometric partitioning**:
- **Eliminates random Gaussian $n$ vectors**: Split directions are generated directly from sample pairs ($n = B - A$).
- **Eliminates infinite hyperplane leakage**: Bounded by an initial surrounding outer envelope where the outer edge is defined strictly as anomaly score $1.000000$.
- **Eliminates density bias**: Uses the generator separation $\|B - A\|$ as an intrinsic density probe, accumulating continuous metric depth rather than discrete hop counts.
- **Eliminates post-hoc score scaling**: Scores map naturally and monotonically to $[0.0, 1.0]$.

---

## 2. Geometric Primitives & Mathematical Foundations

### 2.1 Data-Driven Voronoi Bisector Splits
Rather than generating an artificial normal $n \sim \mathcal{N}(0, I)$ and an independent intercept $p$, a node split in GEIF is defined by two sample points chosen from the node's sample subset $S_v$:

$$\text{Select } A, B \in S_v \quad (A \ne B)$$

The splitting boundary is the **perpendicular bisector (Voronoi facet)** separating $A$ and $B$:
$$\text{Boundary } H(A, B): \quad \|x - A\|^2 = \|x - B\|^2 \quad (x \in \mathbb{R}^D)$$

#### 2.1.1 Algebraic Reduction to a Single Dot Product
The Euclidean distance condition expands as:
$$\sum_{j=1}^D (x_j - A_j)^2 = \sum_{j=1}^D (x_j - B_j)^2$$
$$\sum_{j=1}^D \left( x_j^2 - 2 x_j A_j + A_j^2 \right) = \sum_{j=1}^D \left( x_j^2 - 2 x_j B_j + B_j^2 \right)$$
$$-2 x \cdot A + \|A\|^2 = -2 x \cdot B + \|B\|^2$$
$$2 x \cdot (B - A) = \|B\|^2 - \|A\|^2 = (B - A) \cdot (B + A)$$
$$x \cdot (B - A) = \left(\frac{A + B}{2}\right) \cdot (B - A)$$

Let:
- **Normal vector**: $n = B - A$
- **Midpoint intercept**: $p = \frac{A + B}{2}$

The decision rule for a query point $x$ becomes:
$$x \cdot n < p \cdot n \implies \text{Branch Left (closer to } A\text{)}$$
$$x \cdot n \ge p \cdot n \implies \text{Branch Right (closer to } B\text{)}$$

```
                   Sample A
                      *
                     /
                    /   Branch Left: dot(x, n) < pdotn
      -------------+-------------  ← Bisector Hyperplane:
                  /                  Normal: n = B - A
                 /                   Point:  p = (A + B) / 2
                *   Branch Right: dot(x, n) >= pdotn
             Sample B
```

#### 2.1.2 Intrinsic Properties of the Voronoi Split
1. **Computational Speed**: Requires exactly one dot product $O(D)$ per node during inference, identical to classic EIF.
2. **Automatic Scale & Covariance Alignment**: The difference vector $B - A$ automatically inherits the scale, units, and principal orientation of the data manifold. No manual feature rescaling (`auto_weigth`) is required.
3. **Zero Empty-Space Cuts**: Because cuts are anchored halfway between real samples, hyperplanes never slice empty regions where no data exists.

---

### 2.2 The Initial Surrounding Outer Envelope ($1.0$ Boundary)

To eliminate the infinite hyperplane paradox without post-hoc exponential decay patches, GEIF defines an **initial surrounding outer envelope** $C_{\text{root}}$ at the root of the forest:

```
        +-----------------------------------------------+  ← Outer Border:
        |                                               |    Score = 1.000000
        |         . - ~ - .                             |    Depth H = 0
        |       :  Cluster  :                           |
        |         ' - ~ - '                             |
        |                         * Outlier             |
        |                                               |
        +-----------------------------------------------+
```

For each dimension $j \in \{1, \dots, D\}$:
$$\text{span}_j = \max_{x \in X} x_j - \min_{x \in X} x_j$$
$$C_{\text{root}} = \prod_{j=1}^D \left[ \min_{x \in X} x_j - \alpha \cdot \text{span}_j,\; \max_{x \in X} x_j + \alpha \cdot \text{span}_j \right]$$

Where $\alpha \ge 0$ is the boundary margin factor (default: $\alpha = 0.5$ to $1.0$).

#### Outer Boundary Rule:
For any query point $x$:
$$\text{If } x \notin C_{\text{root}} \implies H(x) = 0.0 \implies \text{Score}(x) = 1.000000$$

- Outside $C_{\text{root}}$, space is unconditionally classified as a maximum anomaly ($1.000000$).
- No internal hyperplane is ever evaluated for points beyond the envelope.
- Ghost rays, starbursts, and radial wedges are mathematically impossible.

---

### 2.3 Local Density Factor & Continuous Metric Depth ($H$)

#### 2.3.1 Using $\|B - A\|$ as the Density Probe
The vector magnitude $d = \|B - A\|$ provides a direct measure of local spatial sparsity:
- In a dense cluster: Samples are tightly packed $\implies \|B - A\|$ is small.
- In a sparse cloud or cavity: Samples are distant $\implies \|B - A\|$ is large.

#### 2.3.2 Continuous Metric Accumulation
Instead of counting discrete tree hops ($h \leftarrow h + 1$), the traversal accumulates **metric continuous depth** $H$:

At each traversed node $v$ with generator points $A_v, B_v$:
$$\Delta H_v = \frac{\|B_{\text{root}} - A_{\text{root}}\|}{\|B_v - A_v\| + \epsilon}$$
or in normalized volume form:
$$\Delta H_v = \ln\left(1.0 + \frac{\text{diam}(C_{\text{root}})}{\|B_v - A_v\|}\right)$$

- Traversing across an empty void or wide gap yields small $\Delta H$ (rapid isolation $\to$ high outlier score).
- Traversing within a dense cluster yields large $\Delta H$ (deep continuous depth $\to$ low outlier score).

---

### 2.4 Native Normalized Scoring (Option C)

Classic iForest scores rely on the non-linear formula $2^{-h/c(n)}$, which compresses nominal inlier scores to $\approx 0.35$ and requires empirical scaling.

GEIF defines the anomaly score directly from the accumulated metric depth $H(x)$:

$$\text{Score}(x) = \max\left(0.0,\; 1.0 - \frac{H(x)}{H_{\max}}\right)$$

Where:
- $H(x) = \frac{1}{T} \sum_{t=1}^T H_t(x)$ is the ensemble average metric depth.
- $H_{\max}$ is the reference maximum depth of the forest.

#### The "Zero Kelvin" Principle for $H_{\max}$
Rather than artificially pegging the best observed training sample to $0.000000$ (which risks saturation when unseen, higher-density observations arrive), GEIF defines $0.0$ as an **asymptotic lower bound—like Absolute Zero (0 Kelvin)**: you can approach it arbitrarily closely, but no finite sample can reach it.

$$H_{\max} = \frac{1}{T} \sum_{t=1}^T \max_{\text{leaf} \in t} H(\text{leaf})$$

By deriving $H_{\max}$ from the structural maximum leaf paths across all trees:
1. **Mathematical Invariant**: Since no single observation can physically land in the deepest leaf of all $T$ trees simultaneously, $H(x) < H_{\max}$ holds strictly for all $x \in \mathbb{R}^D$. Consequently, $\text{Score}(x) > 0.0$ strictly—artificial clamping is never required.
2. **Headroom for Unseen Inliers**: Leaves natural headroom for future, even more prototypical cluster centroids to achieve lower anomaly scores without hitting a flat saturation ceiling.
3. **Decisive for Categorization (`-c`)**: In competitive multi-category routing, points located near cluster cores preserve fine, continuous discriminatory margins (e.g., Category A scores $0.038$ vs. Category B scores $0.052$) rather than collapsing into ambiguous $0.000000$ ties.

```
Score Continuum (Zero Kelvin Principle):
 (0 Kelvin)
  0.000000        ~0.04 - 0.08               0.500000                   1.000000
     |-----------------*------------------------|--------------------------|
Asymptote        Deepest Real Inlier         Nominal Boundary           Outer Border
(Unreachable)    (Headroom preserved)        (H = 0.5 * H_max)            (H = 0)
```

---

## 3. Algorithmic Workflows & Formal Pseudo-Code

### 3.1 Model Training: `BuildTree`

```text
Algorithm: BuildGEIFTree(S, depth, max_depth, C_node)
Input:
    S: Subset of sample indices at current node
    depth: Current tree depth
    max_depth: Maximum tree depth limit (e.g. log2(|S_root|))
    C_node: Bounding cell of the current node
Output:
    Node v (internal node or leaf)

1. If |S| <= 1 or depth >= max_depth:
       v.is_leaf = true
       v.sample_count = |S|
       v.density = CalculateLeafDensity(S)
       return v

2. Select two distinct samples A, B from S:
       Option 1 (Uniform): A, B ~ UniformRandom(S), A != B
       Option 2 (Distance-Weighted): P(A, B) ∝ ||A - B||^2

3. Compute split geometry:
       n = B - A
       p = (A + B) / 2
       split_val = dot(p, n)
       d_AB = ||B - A||

4. Partition samples S into S_left and S_right:
       S_left  = { s in S | dot(s, n) < split_val }
       S_right = { s in S | dot(s, n) >= split_val }

5. If |S_left| == 0 or |S_right| == 0:
       v.is_leaf = true
       v.sample_count = |S|
       return v

6. Compute child bounding cells:
       C_left  = C_node ∩ { x | dot(x, n) < split_val }
       C_right = C_node ∩ { x | dot(x, n) >= split_val }

7. Create internal node v:
       v.is_leaf = false
       v.normal = n
       v.pdotn = split_val
       v.step_weight = MetricWeight(d_AB)
       v.left_child  = BuildGEIFTree(S_left,  depth + 1, max_depth, C_left)
       v.right_child = BuildGEIFTree(S_right, depth + 1, max_depth, C_right)
       return v
```

---

### 3.2 Model Inference: `EvaluatePoint`

```text
Algorithm: EvaluateGEIFPoint(x, Forest)
Input:
    x: D-dimensional query point
    Forest: Trained ensemble of T trees, root envelope C_root, and H_max
Output:
    score: Anomaly score in [0.0, 1.0]

1. // Check initial surrounding outer envelope
   if not PointInEnvelope(x, Forest.C_root):
       return 1.000000

2. total_metric_depth = 0.0

3. for each tree t in Forest.trees:
       node = t.root
       tree_depth = 0.0
       
       while not node.is_leaf:
           tree_depth += node.step_weight
           if dot(x, node.normal) < node.pdotn:
               node = node.left_child
           else:
               node = node.right_child
               
       tree_depth += node.leaf_depth_weight
       total_metric_depth += tree_depth

4. H_avg = total_metric_depth / Forest.tree_count

5. // Compute normalized geometric score
   score = 1.0 - (H_avg / Forest.H_max)
   if score < 0.0:
       score = 0.0
   if score > 1.0:
       score = 1.0

6. return score
```

---

## 4. Complexity & Theoretical Properties

| Metric | Classic EIF | CEIF (Engineered) | GEIF (Geometric Native) |
|---|---|---|---|
| **Training Time** | $O(T \cdot N \log N \cdot D)$ | $O(T \cdot N \log N \cdot D)$ | $O(T \cdot N \log N \cdot D)$ |
| **Inference Time (per point)** | $O(T \cdot \bar{h} \cdot D)$ | $O(T \cdot \bar{h} \cdot D + \text{leaf})$ | $O(T \cdot \bar{h} \cdot D)$ *(Fastest)* |
| **Memory per Node** | Normal $n$, $p$ | Normal $n$, $p$, bounds | Normal $n$, $p\cdot n$, $w$ |
| **Outer Boundary Monotonicity** | No (erratic rays) | Yes (exponential decay patch) | **Yes (exact cell-bounded limit)** |
| **Scale Invariance** | None | Yes (via `auto_weigth`) | **Yes (intrinsic to $B - A$)** |
| **Void / Cavity Resolution** | Poor | Good (via `NEAREST`) | **Natively resolved via $d_{AB}$** |
| **Score Calibration** | Compressed $[0.35, 0.85]$ | Linear scaled $[0, 1]$ | **Naturally normalized $[0, 1]$** |
