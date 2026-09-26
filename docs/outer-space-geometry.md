# GEIF: Geometry of the Outer Space Border

## 1. The Core Questions

1. **What shape is the outer border? Is it a rectangle?**
2. **What actually happens when a data point lands outside? Does it step-jump or transition smoothly?**

---

## 2. Question 1: What Shape Is the Outer Border?

There are three architectural ways to define the outer boundary in $\mathbb{R}^D$:

```
   A. Axis-Aligned Box           B. Bounding Ellipsoid          C. Convex Voronoi Hull
   
   +-------------------+              .  -  ~  -  .                  / \
   |                   |          :                 :              /     \
   |      (Data)       |         :      (Data)       :            | (Data)|
   |                   |          :                 :              \     /
   +-------------------+              '  -  ~  -  '                  \ /
   
   2D: Rectangle                 2D: Ellipse                    2D: Convex Polygon
   D-dim: Hyper-rectangle        D-dim: Hyper-ellipsoid         D-dim: Polytope
```

### Option A: Axis-Aligned Hyper-Rectangle (Bounding Box)
- **Mathematical Form**:
  $$x_j \in [\min_j - \alpha \cdot \text{span}_j,\; \max_j + \alpha \cdot \text{span}_j] \quad \forall j \in \{1, \dots, D\}$$
- **Pros**:
  - Extremely cheap: exactly $2D$ scalar comparisons. Takes $\approx 2$ nanoseconds.
  - Zero memory overhead.
- **Cons**:
  - In 2D, the extreme corners form a 90° box corner.
  - Not rotationally invariant: rotating the dataset changes the box alignment.

---

### Option B: Bounding Ellipsoid (Data Variance / Mahalanobis)
- **Mathematical Form**:
  Centered at data mean $\mu$, oriented along the feature covariance $\Sigma$:
  $$d_M^2(x) = (x - \mu)^T \Sigma^{-1} (x - \mu) \le R^2$$
- **Pros**:
  - **Rotationally invariant & smooth**: No sharp 90° box corners.
  - Conforms naturally to correlated diagonal features (e.g., if feature 1 and feature 2 have strong covariance).
- **Cons**:
  - Requires computing or approximating the inverse covariance matrix $\Sigma^{-1}$ at train time ($O(D^2)$ to evaluate).

---

### Option C: The Hybrid "Rounded Hyper-Rectangle" (Euclidean Distance from Box)
- This is what we discovered during the `ceif` exterior decay experiments!
- Inside the data hull: distance outside $d_{\text{out}} = 0$.
- Outside the data box: measure the Euclidean distance to the nearest face/corner of the box:
  $$d_{\text{out}}^2(x) = \sum_{j=1}^D \max\left(0,\; \min_j - x_j\right)^2 + \max\left(0,\; x_j - \max_j\right)^2$$
- **The Geometry**:
  - Flat along the sides of the data.
  - **Naturally rounded at the corners** (due to Euclidean $\sqrt{\sum \Delta x^2}$)!
  - Produces a smooth, pill-shaped stadium envelope in 2D, completely eliminating sharp square corner artifacts!

---

## 3. Question 2: What Actually Happens When Data Hits Outside?

### Behavior 1: The Hard Cutoff ("The Electric Fence")

```text
Score
 1.0 |                   +------------------------- (Step jump to 1.0)
     |                   |
 0.8 |        . - ~ - '  | (Inside score was ~0.82)
     |      :            :
 0.0 +------+------------+------------------------- Distance
           Data        Border
```

```c
if (is_outside_envelope(x)) {
    return 1.000000; // Immediate hard clamp
}
```

#### What Happens:
- Points just 1 mm inside the boundary might score `0.82`.
- Points 1 mm outside the boundary jump instantaneously to `1.000000`.
- **The Problem**: On heatmaps and downstream automation, this creates an **artificial cliff / step artifact**. If an anomaly monitoring threshold is set to `0.95`, a point abruptly crosses from nominal to extreme anomaly at an arbitrary geometric line.

---

### Behavior 2: Smooth Monotonic Decay (Continuous Outer Metric) — *Recommended*

```text
Score
 1.0 |                                     . - - - - - - (Smooth 1.0 Asymptote)
     |                           . - ~ - '
 0.8 |                 . - ~ - '
 0.5 |       . - ~ - '
 0.0 +-------+------------------------------------------- Distance
           Data        Periphery              Deep Space
```

Instead of an instant step jump, entering outer space **smoothly attenuates the continuous metric depth $H$**:

$$H(x) = H_{\text{tree}}(x) \cdot \exp\left(-\frac{d_{\text{out}}(x)}{\text{span}}\right)$$
$$\text{Score}(x) = 1.0 - \frac{H(x)}{H_{\max}}$$

#### What Actually Happens:
1. **Inside the Data Cluster ($d_{\text{out}} = 0$)**:
   $\exp(0) = 1.0$. The score is 100% driven by the Voronoi tree traversal ($0.00 \to 0.50$).
2. **At the Cluster Perimeter ($d_{\text{out}} \approx 0.1 \times \text{span}$)**:
   Decay is minuscule ($\approx 0.90$). Score smoothly rises to $\approx 0.60 - 0.70$.
3. **In the Far Periphery ($d_{\text{out}} \approx 1.5 \times \text{span}$)**:
   Score smoothly reaches $\mathbf{0.900000}$.
4. **Deep Space ($d_{\text{out}} \ge 4.0 \times \text{span}$)**:
   $\exp(-4) \approx 0.018 \implies H \to 0 \implies \text{Score} \to \mathbf{0.999 \to 1.000000}$.

#### The Breakthrough:
- **Zero discontinuities**: The derivative is continuous everywhere ($\mathcal{C}^1$ smooth).
- **Monotonically increasing**: As you walk away from the data, your anomaly score *strictly increases* in all directions.
- **Zero starburst rays**: Infinite hyperplanes cannot create isolated low-score wedges because the radial decay acts as an isotropic spatial damper.
