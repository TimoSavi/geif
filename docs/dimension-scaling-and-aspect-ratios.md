# GEIF: Handling Disparate Feature Scales & Aspect Ratios

## 1. The Disparate Scale Problem

In real-world anomaly detection (and specifically in benchmarks like `complex2d.csv`), features have completely different physical units, variances, and numerical ranges:

* **East/West ($X$, e.g., Network Bytes / Latency in $\mu s$):**
  $$\text{Range: } [1000.0, 9000.0] \implies \text{Span}_X \approx 8000.0$$
* **North/South ($Y$, e.g., CPU Ratio / Packet Loss Rate):**
  $$\text{Range: } [0.2, 1.8] \implies \text{Span}_Y \approx 1.6$$
* **Aspect Ratio:**
  $$\frac{\text{Span}_X}{\text{Span}_Y} = \frac{8000.0}{1.6} = 5000 : 1$$

---

## 2. What Happens If Data Is Left Unscaled in Raw Euclidean Space?

If an algorithm naively applies raw Euclidean distance $\|x - y\|^2 = \sum_j (x_j - y_j)^2$:

### 2.1 The Voronoi Cut Degenerates into 1D Slices
When two training points $A$ and $B$ are drawn:
$$\Delta X = B_X - A_X \approx 4000.0$$
$$\Delta Y = B_Y - A_Y \approx 0.8$$

The raw normal vector is $n = (4000.0,\; 0.8)$.
The angle of this cut relative to the vertical axis is:
$$\theta = \arctan\left(\frac{0.8}{4000.0}\right) \approx 0.011^\circ$$

**Catastrophic Consequence:**
* Virtually 100% of all hyperplanes across all trees become **pure vertical slices**!
* The $Y$ coordinate is completely drowned out into floating-point rounding error ($4000^2 = 16,000,000$ vs $0.8^2 = 0.64$).
* The algorithm is completely blind to horizontal topological features (e.g., donut cavities, North/South spikes, and curved arcs).

### 2.2 Outer Space Distance Is Distorted by a Factor of 5000
If a point steps $1.0$ unit outside the North/South boundary (which is **62% outside the entire cluster span**):
$$\Delta Y^2 = 1.0^2 = 1.0$$
If a point steps $20.0$ units outside the East/West boundary (which is **only 0.25% outside the cluster span**):
$$\Delta X^2 = 20.0^2 = 400.0$$

A tiny deviation in $X$ would be scored as $400\times$ more anomalous than a massive excursion in $Y$!

---

## 3. The GEIF Solution: Dimension-Relative Metric (Intrinsic Normalization)

To be mathematically sound, distances and bisector hyperplanes must be measured in **dimension-relative units**:
$$\tilde{x}_j = \frac{x_j}{\text{span}_j} \quad \text{where } \text{span}_j = \max_{x \in X} x_j - \min_{x \in X} x_j$$

In this normalized metric space:
* Both $X$ and $Y$ have unit span $[0.0, 1.0]$.
* The donut cavity is a true circle.
* Bisector hyperplanes cut in all orientations with equal probability.

---

## 4. The Mathematical Breakthrough: Pre-Baking Scale into Hyperplane Normals

A naive implementation would require normalizing every incoming streaming query vector $x$ before traversing the trees, incurring memory copies and division overhead.

**GEIF solves this algebraically with ZERO run-time cost during inference!**

### 4.1 Algebraic Derivation
In normalized space, query point $x$ is closer to $A$ than $B$ if:
$$\sum_{j=1}^D \left(\frac{x_j - A_j}{\text{span}_j}\right)^2 < \sum_{j=1}^D \left(\frac{x_j - B_j}{\text{span}_j}\right)^2$$

Expanding each side:
$$\sum_{j=1}^D \frac{1}{\text{span}_j^2} \left(x_j^2 - 2 x_j A_j + A_j^2\right) < \sum_{j=1}^D \frac{1}{\text{span}_j^2} \left(x_j^2 - 2 x_j B_j + B_j^2\right)$$

Cancel $x_j^2$ from both sides:
$$\sum_{j=1}^D \frac{1}{\text{span}_j^2} \left(-2 x_j A_j + A_j^2\right) < \sum_{j=1}^D \frac{1}{\text{span}_j^2} \left(-2 x_j B_j + B_j^2\right)$$
$$\sum_{j=1}^D \frac{2 x_j (B_j - A_j)}{\text{span}_j^2} < \sum_{j=1}^D \frac{B_j^2 - A_j^2}{\text{span}_j^2} = \sum_{j=1}^D \frac{(B_j - A_j)(A_j + B_j)}{\text{span}_j^2}$$

Divide both sides by $2$:
$$\sum_{j=1}^D x_j \cdot \left[ \frac{B_j - A_j}{\text{span}_j^2} \right] < \sum_{j=1}^D \left[ \frac{B_j - A_j}{\text{span}_j^2} \right] \cdot \left(\frac{A_j + B_j}{2}\right)$$

### 4.2 The Pre-Baked Split Representation
Define the pre-weighted node parameters at train time:
$$n_j = \frac{B_j - A_j}{\text{span}_j^2}$$
$$p_{\text{scalar}} = \sum_{j=1}^D n_j \left(\frac{A_j + B_j}{2}\right)$$

Now look at the decision rule for any **raw, unscaled query vector $x$**:
$$\sum_{j=1}^D x_j n_j < p_{\text{scalar}} \implies \text{Branch Left}$$
$$\sum_{j=1}^D x_j n_j \ge p_{\text{scalar}} \implies \text{Branch Right}$$

### 4.3 Why This Is Optimal
1. **Zero Input Transformation:** Streaming input vectors $x$ are evaluated directly in their raw numbers (no dividing by spans, no copying arrays).
2. **Single SIMD Dot Product:** Inference remains exactly one dot product per node `dot(x, n) < p_scalar`.
3. **Perfect Aspect Ratio Invariance:** In `complex2d.csv`, $n_X$ is automatically scaled down by $8000^2$ while $n_Y$ is scaled down by $1.6^2$. Both dimensions carry identical statistical weight!

---

## 5. Outer Space Boundary on Disparate Scales

The same dimension-relative metric applies to the outer space distance $d_{\text{out}}$:

$$d_{\text{out}}^2(x) = \sum_{j=1}^D \left( \frac{\max(0,\; \min_j - x_j) + \max(0,\; x_j - \max_j)}{\text{span}_j} \right)^2$$

* **Effect:** Distance is measured in **units of cluster span** (e.g. 0.1 = 10% outside).
* **Geometry:** The outer boundary naturally conforms to the 5000:1 aspect ratio. It forms a rounded stadium elongated along $X$ and compressed along $Y$, treating a 10% departure in $Y$ with the exact same anomaly penalty as a 10% departure in $X$.
