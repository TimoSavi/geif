# GEIF: Deep Dive into Zero-Variance Dimensions & Constant Features

## 1. The Anatomy of Issue 1

In real-world telemetry, production logs, and IoT sensor streams, features are frequently **constant (zero variance)** during baseline training:

* **Error Flags:** An `error_code` column that was `0` for all $N$ training samples.
* **Network Protocol:** An `ip_proto` column that is always `6` (TCP) in a dedicated TCP monitoring pipeline.
* **Firmware / Schema Placeholders:** An unused sensor channel pinned at `0.000000`.
* **Calibrated Setpoints:** A target frequency or voltage pegged at nominal value (e.g., `50.000 Hz`).

### The Mathematical Hazards
In GEIF, dimension scaling relies on the feature span:
$$\text{span}_j = \max_{x \in X} x_j - \min_{x \in X} x_j = c_j - c_j = 0.0$$

A naive implementation encounters **catastrophic division by zero (`NaN`)** at three critical steps:
1. **Normal Vector Calculation:**
   $$n_j = \frac{B_j - A_j}{\text{span}_j^2} = \frac{0}{0} \implies \text{NaN}$$
2. **Outer Space Distance:**
   $$d_{\text{out}, j} = \frac{\Delta x_j}{\text{span}_j} = \frac{|x_j - c_j|}{0.0} \implies +\infty \text{ or NaN}$$
3. **Micro-Variance Instability:**
   If a feature is not strictly zero, but has tiny floating-point sensor jitter (e.g. $\text{span}_j = 10^{-10}$):
   $$\frac{1}{\text{span}_j^2} = 10^{20}$$
   This single near-constant feature would overwhelm all real features by $10^{20}\times$, causing numerical overflow!

---

## 2. Train-Time Architecture: Geometric Invariance

### 2.1 The Geometry of a Zero-Thickness Manifold
Geometrically, if all training samples have identical value $c_j$ on dimension $j$, the entire dataset lies on a flat hyper-plane orthogonal to axis $j$. The data manifold has **zero geometric thickness** in dimension $j$.

For any pair of training samples $A, B \in S_v$:
$$A_j = c_j, \quad B_j = c_j \implies B_j - A_j = 0.0$$

The bisector between $A$ and $B$ has no basis to partition space along dimension $j$.

### 2.2 The Solution: Dimension Health Mask
During dataset ingest (single pass $O(N)$):
1. Compute $\min_j$, $\max_j$, and $\text{span}_j = \max_j - \min_j$.
2. Compare against a stability threshold $\epsilon_{\text{span}} = 10^{-9}$:
   $$\text{is\_active}_j = \begin{cases} \text{true} & \text{if } \text{span}_j \ge \epsilon_{\text{span}} \\ \text{false} & \text{if } \text{span}_j < \epsilon_{\text{span}} \end{cases}$$
3. For inactive dimensions:
   $$n_j = 0.000000 \quad \text{always}$$

**Result at Train Time:**
- No division by zero.
- The Voronoi tree construction is 100% stable.
- Constant features consume zero influence in hyperplane cuts.

---

## 3. Inference-Time Architecture: The Vital Question

What happens at inference time when a new streaming point arrives?

### Case A: The Query Point Is Nominal ($x_j == c_j$)
* In tree traversal: $x_j \cdot n_j = c_j \cdot 0.0 = 0.0$.
* In outer space: $|x_j - c_j| = 0.0 \implies d_{\text{out}, j} = 0.0$.
* **Behavior:** The constant dimension is completely transparent. The point is scored purely on its active dimensions.

---

### Case B: The Query Point Deviates ($x_j \ne c_j$)
This is the core design choice. The training set *never* observed anything other than $c_j$. Now a query point has $x_j \ne c_j$:
* `error_code` went from `0` $\to$ `1`
* `ip_proto` went from `6` $\to$ `17` (UDP)
* `voltage` went from `230.0` $\to$ `230.005` (Sensor drift)

How should GEIF penalize this?

#### Design Option 1: The "Immediate Disqualification" (Strict Discrete)
* Any deviation on a zero-variance feature sets $d_{\text{out}} = +\infty$.
* Result: $H(x) = 0 \implies \text{Score} = \mathbf{1.000000}$.
* **Verdict:** Excellent for discrete flags and categorical IDs. However, for physical sensors, microscopic calibration noise (e.g. $230.000001\,\text{V}$) would unfairly trigger an instant $1.0$ alert.

#### Design Option 2: The Regularized Adaptive Span (Recommended)
Instead of dividing by $0$, GEIF assigns an **imputed effective span** for inactive dimensions:

$$\text{effective\_span}_j = \max\left(\text{span}_j,\; \alpha \cdot |\mu_j| + \beta \cdot \overline{\text{span}}_{\text{active}}\right)$$

Where:
* $\alpha = 0.01$ (1% relative tolerance to the nominal value).
* $\beta \cdot \overline{\text{span}}_{\text{active}}$ scales relative to other features if $\mu_j = 0$.

$$\Delta d_{\text{out}, j} = \frac{|x_j - c_j|}{\text{effective\_span}_j}$$

#### How This Behaves in Practice:
1. **Tiny Measurement Drift ($230.001\,\text{V}$ vs $230.0\,\text{V}$):**
   $$\Delta d = \frac{0.001}{2.30} \approx 0.0004 \implies \text{Virtually zero score impact.}$$
2. **True Outlier ($245.0\,\text{V}$ vs $230.0\,\text{V}$):**
   $$\Delta d = \frac{15.0}{2.30} \approx 6.5 \implies \exp(-6.5) \approx 0.0015 \implies \text{Score } \to \mathbf{0.999}.$$
3. **Discrete Flag ($0 \to 1$ on an error column):**
   $$\Delta d = \frac{1.0}{\beta \cdot \overline{\text{span}}} \gg 4.0 \implies \text{Score } \to \mathbf{1.000000}.$$

---

## 4. Edge Case: What If ALL Dimensions Are Constant?

Suppose the dataset consists of identical rows (or a single sample):
* Every dimension is inactive ($\text{is\_active}_j = \text{false} \; \forall j$).
* No splits are possible $\implies$ Each tree is a single root leaf node.
* **Inference Rule:**
  - If a query point exactly equals the constant vector ($x == c$): $\text{Score} = \mathbf{0.000000}$ (or baseline nominal).
  - If a query point differs on any coordinate ($x \ne c$): Outer distance kicks in $\implies \text{Score} \to \mathbf{1.000000}$.
* The system remains fully defined and never crashes.
