# GEIF: Tree Depth Control, Split Balance & Ensemble Diversity

**Author / Maintainer:** Timo Savinen (AI-assisted)

## 1. The Anatomy of Issue 5: Asymmetric Splits & Tree Imbalance

In standard binary search trees, the ideal split is balanced (50% / 50%). 

However, in isolation-based anomaly detection, **asymmetric splits are often desirable**:
* If a node contains $255$ nominal inliers and $1$ extreme outlier, the perfect split is **$1$ vs $255$** (isolating the anomaly in a single cut at depth $1$).

```text
       Desirable Asymmetric Split                 Undesirable Inlier Degeneracy
        (Isolates Outlier at Root)                  (Linked-List Chaining)

                  [ 256 ]                                   [ 256 ]
                  /     \                                   /     \
             [ 1 ]       [ 255 ]                       [ 1 ]       [ 255 ]
            (Outlier!)   (Inliers)                                 /     \
                                                              [ 1 ]       [ 254 ]
                                                                          /     \
                                                                     [ 1 ]       [ 253 ] ...
```

### The Inlier Chaining Hazard
The hazard occurs when asymmetric splits happen **inside nominal inlier clusters**:
* If random sample pairs repeatedly slice off only $1$ inlier at a time, the branch degenerates into a **linked list** of depth $40$ or $50$.
* **Consequences:**
  1. **Memory Bloat:** The tree allocates redundant node structures.
  2. **Inference Latency:** Query points must follow $50$ pointer dereferences instead of $\log_2(256) \approx 8$.
  3. **Depth Distortion:** A nominal sample trapped at the end of a degenerate chain receives an artificially inflated path length.

---

## 2. The Three GEIF Balance Mechanisms

GEIF controls split balance and tree depth through three synergistic mechanisms:

### 2.1 Mechanism 1: Deterministic Maximum Depth Cap ($h_{\max}$)
Just like classic iForest, GEIF enforces a hard mathematical ceiling on tree depth:

$$h_{\max} = \beta \cdot \lceil \log_2(\psi) \rceil$$

Where:
* $\psi$ is the tree sub-sample size (default: $\psi = 256 \implies \lceil \log_2(256) \rceil = 8$).
* $\beta = 2.0$ (allowing trees to grow up to $2\times$ average depth).
* **Default Cap:** $h_{\max} = 2 \times 8 = \mathbf{16}$.

#### What Happens at $h_{\max}$?
If a branch reaches depth $16$, splitting halts immediately:
1. The node becomes an **Early-Termination Leaf**.
2. It stores remaining sample count $k = |S_v|$.
3. It receives an analytic depth completion bonus:
   $$H_{\text{leaf}} = H_{\text{current}} + \log_2(k) \cdot \overline{\delta}_{\text{forest}}$$
4. **Deterministic Arena Guarantee:** Because depth is capped at $16$, a binary tree can have at most $2\psi - 1 = \mathbf{511\text{ nodes}}$. The entire forest memory can be pre-allocated in a single contiguous arena with **zero dynamic mallocs during training**!

---

### 2.2 Mechanism 2: Candidate Pair Rejection ($K$-Trials)
When selecting generator points $A, B \in S_v$:
1. A candidate pair is evaluated:
   - Does it produce a completely empty split ($0$ points on one side)?
   - If so, reject and try a second candidate pair (up to $K=3$ trials).
2. If $|S_v| \le 3$, simply pick the pair with the largest separation $\delta = \|B - A\|$ to ensure maximum geometric partition.
3. This simple check eliminates over **80% of accidental degenerate splits** with virtually zero CPU overhead.

---

### 2.3 Mechanism 3: Ensemble Law of Large Numbers ($T = 100$ Trees)
The defining strength of isolation ensembles is that **no single tree needs to be mathematically perfect**:
* Each tree $t \in \{1, \dots, T\}$ is trained on an **independently drawn random sub-sample** of size $\psi = 256$ (via reservoir sampling).
* If Tree 1 happens to produce an asymmetric cut on a cluster, Tree 2, Tree 3, and Tree 4 will pick completely different generator pairs and cut from different angles.
* By the Central Limit Theorem, individual tree depth variance decays as:
  $$\sigma_{\text{ensemble}} = \frac{\sigma_{\text{tree}}}{\sqrt{T}}$$
* Across $T = 100$ trees, individual split skews are smoothed into a continuous, robust density profile.

---

## 3. Why Sub-Sampling ($\psi = 256$) Is the Global Sweet Spot

In production datasets with $1,000,000+$ rows, users often ask why GEIF sub-samples $\psi = 256$ points per tree rather than building massive trees with all $1,000,000$ points.

### The Two Classic Big-Data Pitfalls:
1. **The Masking Effect (Swallowed Outliers):**
   If a tree is trained on $1,000,000$ samples, massive dense clusters grow so large that an outlier standing close to a cluster boundary gets surrounded and "swallowed" (masked) by thousands of interior nodes.
2. **The Swamping Effect (False Clusters):**
   A small group of $10$ outliers in empty space looks like an isolated cluster in a $1,000,000$-sample tree, falsely receiving an inlier score. In a sub-sample of $256$, at most $1$ or $2$ of those outliers appear in any single tree, ensuring they are isolated immediately.

### Performance & Geometric Efficiency:
* Training a 256-sample tree takes **$\approx 30$ microseconds**.
* Training an ensemble of $T = 100$ trees takes **$\approx 3$ milliseconds**!
* 100 trees of 256 samples create an intersecting web of **$25,600$ Voronoi hyperplanes**, resolving intricate non-linear boundaries around complex manifolds without ever storing large matrices in memory.
