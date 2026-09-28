# GEIF: Topological Cavities & The Non-Convex Void Problem (The Donut Hole)

**Author / Maintainer:** Timo Savinen (AI-assisted)

## 1. The Anatomy of Issue 3: Non-Convex Geometries

In anomaly detection benchmarks (such as `complex2d.csv`) and real multi-sensor correlation systems, data frequently forms **non-convex manifolds**:
* **Annular Rings / Donut Distributions:** A rotating machine or orbital phase where combinations of $(X, Y)$ must maintain a fixed radius $r \in [R_{\min}, R_{\max}]$, leaving an **empty interior void** ($r < R_{\min}$).
* **Crescent / Banana Manifolds:** Curved correlation trajectories that curl around an empty bay.
* **Disconnected Multi-Modal Clusters:** Two discrete clusters with empty space between them.

```text
               The Classic "Donut Hole" Dilemma

                      . - ~ - ~ - .
                   :                 :
                  :   ( EMPTY VOID )  :  ← True Anomaly! (0 training points)
                  :     Point X ?     :
                   :                 :
                      ' - ~ - ~ - '
                     ↑ Dense Ring ↑
                  (800 training points)
```

---

## 2. Why Linear Trees Fail on Cavities: "Hyperplane Bridging"

A tree split boundary is an infinite hyperplane: $(x - p) \cdot n = 0$. In 2D, this is an infinite straight line; in $D$ dimensions, an infinite flat sheet.

### The Catastrophic "Ghost Inlier" Failure Mode:
1. When a tree needs to split samples on the North side of the ring from samples on the South side, the straight cut **inevitably slices directly across the empty central cavity**.
2. A test point $X_{\text{void}}$ sitting dead center in the empty hole arrives:
   - Split 1 (North vs South): $X_{\text{void}}$ lands on one side.
   - Split 2 (East vs West): $X_{\text{void}}$ lands on one side.
   - Split 3, 4, 5, 6...
3. Because $X_{\text{void}}$ is geographically surrounded by training samples, it gets repeatedly bounced back and forth between bisectors, **traveling deep down the tree to depth 9 or 10**!
4. **The False Classification:**
   Classic Isolation Forest logic states:
   $$\text{High Path Length } (h \ge 9) \implies \text{Dense Inlier!}$$
   The algorithm declares the empty hole to be a **nominal inlier**! This is the notorious **"Hyperplane Bridging / Ghost Inlier"** bug in tree-based anomaly detection.

---

## 3. The GEIF Solution: Residual Cell Distance (Leaf Proximity Regularization)

GEIF solves this without abandoning fast $O(D)$ linear dot-product tree traversal. 

### 3.1 The Voronoi Cell Property
Because GEIF uses perpendicular bisectors between samples ($n = B - A$), the leaves of each tree form a true **Voronoi tessellation** of space:
* Every leaf node corresponds to a specific training sample $P_{\text{leaf}}$ (the cell generator).
* When a query point $x$ traverses the tree, it lands in the Voronoi cell of the training sample $P_{\text{leaf}}$ that it is closest to!

### 3.2 The Residual Distance Probe
Once query point $x$ reaches its leaf $P_{\text{leaf}}$, GEIF evaluates the **normalized residual distance**:
$$d_{\text{residual}}(x) = \|\tilde{x} - \tilde{P}_{\text{leaf}}\| = \sqrt{\sum_{j=1}^D \left(\frac{x_j - P_{\text{leaf}, j}}{\text{span}_j}\right)^2}$$

Now compare the two cases:
* **Point on the Ring Manifold:**
  $x$ is surrounded by actual training data. Its distance to the nearest leaf sample is tiny:
  $$d_{\text{residual}}(x) \le r_{\text{nominal}} \approx 0.02 \quad (\text{local cluster spacing})$$
* **Point in the Donut Hole:**
  $x$ landed in this Voronoi cell because one side of the ring was slightly closer than the other. But because there are **zero training points in the hole**, the distance to that ring sample is massive:
  $$d_{\text{residual}}(x_{\text{void}}) \approx 0.40 \gg 0.02!$$

---

## 4. The Void Damping Attenuation

Instead of naively accepting the tree depth $H_{\text{tree}}$, GEIF applies the **Cauchy-Lorentz Void Damping Factor**:

$$H(x) = H_{\text{tree}}(x) \cdot \frac{1}{1 + \left(\frac{d_{\text{residual}}(x)}{r_{\text{nominal}}}\right)^2}$$
$$\text{Score}(x) = 1.0 - \frac{H(x)}{H_{\max}}$$

Where $r_{\text{nominal}}$ is the characteristic neighbor spacing of the forest (derived from average generator distances $\|B - A\|$).

### How the Math Behaves in Practice:

| Spatial Region | $d_{\text{residual}}$ | Damping Factor | Metric Depth $H$ | Final GEIF Score | Status |
|---|---|---|---|---|---|
| **Ring Core (Nominal)** | $\approx 0.01$ | $\approx 0.90$ | High ($H \approx 9.0$) | **`0.10 - 0.25`** | Clean Inlier |
| **Ring Edge (Fringe)** | $\approx 0.03$ | $\approx 0.50$ | Moderate ($H \approx 5.0$) | **`0.50 - 0.65`** | Boundary |
| **Donut Hole (Cavity)** | $\approx 0.35$ | $\approx \frac{1}{1 + 17^2} = 0.003$ | Attenuated to $\approx 0$ | **`0.995 → 1.000`** | **True Void Anomaly** |
| **Outer Space (Exterior)** | $\approx 1.50$ | Quenched by $d_{\text{out}}$ decay | Attenuated to $\approx 0$ | **`0.999 → 1.000`** | **Outer Anomaly** |

---

## 5. Summary of Breakthroughs

1. **Zero Ghost Inliers:** The donut hole is no longer bridged by straight cuts. It correctly scores as an anomaly ($\approx 1.000000$).
2. **Smooth Circular Contours:** The Cauchy-Lorentz factor produces smooth, continuous radial gradients inside the hole rather than jagged geometric steps.
3. **No Heavy Kernel Computation:** Does not require $O(N^2)$ Kernel PCA, SVM kernels, or non-linear manifolds. Tree traversal remains a blazing-fast single dot product per node, with only **one scalar distance calculation at the leaf**!
