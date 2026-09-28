# GEIF: Duplicate Samples & Inseparable Node Leaves

**Author / Maintainer:** Timo Savinen (AI-assisted)

## 1. The Anatomy of Issue 2: Why Duplicates Break Bisectors

In real telemetry streams and discrete datasets, multiple rows often share the **exact same coordinate vector**:
* **Quantized Sensor Readings:** Temperature recorded in integer degrees (e.g., $21^\circ\text{C}, 21^\circ\text{C}, 21^\circ\text{C}$).
* **Categorical / Embedded States:** Identical state encodings (e.g., multiple identical `(user_id=0, status=OK)` telemetry packets).
* **High-Frequency Ping Bursts:** Identical packet sizes and port pairs.

### The Mathematical Degeneracy
In GEIF, hyperplanes are constructed directly from sample differences:
$$n = B - A$$

If a tree node randomly selects two samples $A$ and $B$ that have identical coordinates:
$$A = B \implies B - A = \mathbf{0} \implies n = \mathbf{0}$$

A normal vector of zeros $n = [0, 0, \dots, 0]$ cannot define a hyperplane. The split rule $x \cdot \mathbf{0} < p \cdot \mathbf{0}$ degenerates into $0 < 0$, which is mathematically undefined and cannot partition the sample set.

---

## 2. The Two Duplication Scenarios

When a tree node $v$ contains duplicate samples, it falls into one of two distinct categories:

### Scenario A: Mixed Duplicates (Node Contains Duplicates + Other Points)
* **Example:** Node has $50$ samples: $45$ copies of point $P_1 = (10, 20)$ and $5$ copies of point $P_2 = (15, 30)$.
* **The Hazard:** Naive random sampling $A, B \in S_v$ has an $\approx 81$% probability of picking $(P_1, P_1)$, causing $4$ out of $5$ split attempts to fail!
* **The Solution:** 
  1. Select candidate $A$ randomly.
  2. Attempt up to $K$ trials (default: $K=5$) to pick $B$ such that $\|B - A\| > \epsilon$.
  3. If random trials hit duplicates, perform a rapid linear scan across $S_v$ to find the first sample distinct from $A$.
* **The Geometric Result:** $n = P_2 - P_1 \ne \mathbf{0}$. The bisector cut cleanly isolates all $45$ copies of $P_1$ to the left branch and all $5$ copies of $P_2$ to the right branch in **a single cut**!

---

### Scenario B: Purely Inseparable Node (All Samples in Node Are Identical)
* **Example:** Node has $45$ samples, and every single one is $P = (10, 20)$.
* **The Hazard in Classic iForest:** Classic iForest would continue making meaningless cuts with Gaussian noise, creating $10$ to $15$ levels of dummy nodes until hitting `max_depth`. This wastes memory and burns CPU during inference.
* **The Solution in GEIF (Early Termination):**
  1. If no distinct sample exists ($A_j == B_j$ for all pairs), splitting **halts immediately**.
  2. The node is declared an **Inseparable Leaf Node**.
  3. The leaf stores:
     - `sample_count = k` (e.g., $45$)
     - `representative_point = P`
  4. Memory allocation stops immediately at depth $h = 3$ instead of depth $16$.

---

## 3. Scoring Inseparable Leaves: The Density Bonus

Does a cluster of $45$ identical points represent an anomaly or an inlier?

**Answer:** $45$ identical points represent an **extremely dense probability peak** (an ultra-inlier). 

### 3.1 When a Query Point Matches the Leaf Exactly ($x == P_{\text{leaf}}$)
In classic iForest, a leaf containing $k$ unresolved samples is credited with the average unbuilt subtree path length:
$$c(k) = 2 \ln(k - 1) + 2 \times 0.5772156649 - \frac{2(k - 1)}{k}$$

In GEIF, an exact match at an inseparable leaf receives an analytic continuous metric depth bonus:
$$H_{\text{leaf}} = H_{\text{current}} + \log_2(k) \cdot \overline{\delta}_{\text{forest}}$$

* **Result:** Because $H_{\text{leaf}}$ is large, the resulting anomaly score $\text{Score} = 1.0 - \frac{H}{H_{\max}}$ drops close to the **"Zero Kelvin"** nominal baseline. The model correctly identifies high-frequency duplicate telemetry as normal behavior.

---

### 3.2 When a Query Point Lands in the Leaf but Differs ($x \ne P_{\text{leaf}}$)
Suppose a query point $x = (10.5, 20)$ traverses the forest and falls into this Voronoi cell. It ended up here because it was closer to $P = (10, 20)$ than to any other training point, but it is **not** identical to $P$.

Because the training data at this cell had **zero variance** (all $k$ samples were strictly at $P$), any excursion represents a departure from that tight spike:

1. Measure normalized residual distance to the leaf:
   $$d_{\text{residual}}(x) = \sqrt{\sum_{j=1}^D \left(\frac{x_j - P_{\text{leaf}, j}}{\text{span}_j}\right)^2}$$
2. Attenuate the leaf metric depth:
   $$H(x) = H_{\text{leaf}} \cdot \frac{1}{1 + \left(\frac{d_{\text{residual}}(x)}{r_{\text{cell}}}\right)^2}$$
   - If $d_{\text{residual}} = 0 \implies H = H_{\text{leaf}}$ (Full inlier bonus).
   - If $d_{\text{residual}} > 0 \implies H$ drops smoothly, properly identifying the point as an off-center anomaly without step cliffs.

---

## 4. Engineering Impact & Benchmarks

| Metric | Classic EIF Behavior | GEIF Inseparable Leaf Handling |
|---|---|---|
| **Tree Depth on Duplicates** | Deep chains down to `max_depth` (16–20) | **Truncates immediately** (depth 2–4) |
| **Node Memory Overhead** | 30+ dummy nodes allocated per duplicate cluster | **1 compact leaf node** |
| **Inference Traversal Time** | 16–20 pointer dereferences per tree | **2–4 pointer dereferences** (up to $5\times$ faster) |
| **Mathematical Soundness** | Indeterminate splits ($0 < 0$) | **Guaranteed non-zero bisector** or clean early stop |
