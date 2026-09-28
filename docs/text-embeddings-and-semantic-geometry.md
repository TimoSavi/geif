# GEIF: Handling Text & Categorical Data via Semantic AI Embeddings

**Author / Maintainer:** Timo Savinen (AI-assisted)

## 1. The Core Dilemma: Arbitrary Text Encodings

When real-world telemetry or tabular data contains categorical strings (e.g., `dog`, `cat`, `spoon`, `fork` or HTTP error strings, device models, user-agents), traditional numeric conversions fail geometrically:

### 1.1 Label / Hash Encoding (Arbitrary Distances)
```text
"dog"   → 1
"cat"   → 2
"spoon" → 3
"fork"  → 4
```
* **The geometric error:**
  $$\text{dist}(\text{cat}, \text{spoon}) = |2 - 3| = 1.0$$
  $$\text{dist}(\text{dog}, \text{fork}) = |1 - 4| = 3.0$$
* A cat is scored as geometrically closer to a spoon than a dog is to a fork. The distance metric is pure noise imposed by alphabetical or hashing order.

### 1.2 One-Hot Encoding (Orthonormal Blindness)
```text
"dog"   → [1, 0, 0, 0]
"cat"   → [0, 1, 0, 0]
"spoon" → [0, 0, 1, 0]
"fork"  → [0, 0, 0, 1]
```
* **The geometric error:**
  $$\text{dist}(\text{dog}, \text{cat}) = \sqrt{2} \approx 1.414$$
  $$\text{dist}(\text{dog}, \text{fork}) = \sqrt{2} \approx 1.414$$
* Every distinct token is orthogonal. The model is completely blind to the fact that dogs and cats share the semantic manifold of "animals" while spoons and forks share the manifold of "cutlery".

---

## 2. The AI Solution: Semantic Vector Embeddings

Humans immediately recognize semantic relationships because human conceptual spaces are metric manifolds. Modern AI embedding models (e.g. Gemini `text-embedding-004`, OpenAI `text-embedding-3`, or local open-weights models like `all-MiniLM-L6-v2`) project strings into dense continuous vector spaces $\mathbb{R}^D$:

```text
              Animals Manifold                       Cutlery Manifold
          (dog, cat, puppy, wolf)                 (spoon, fork, knife)
                     *                                     +
                    /                                     /
                   /                                     /
           *------+------*                       +------+------+
          dog    |      cat                    spoon   |     fork
                 |                                     |
                 v                                     v
         Dense Cluster A                       Dense Cluster B
```

### 2.1 Geometric Separation in Embedding Space
In vector embedding space:
* $\text{CosineSim}(\mathbf{v}_{\text{dog}}, \mathbf{v}_{\text{cat}}) \approx 0.88 \implies \text{Distance is small}$.
* $\text{CosineSim}(\mathbf{v}_{\text{spoon}}, \mathbf{v}_{\text{fork}}) \approx 0.91 \implies \text{Distance is small}$.
* $\text{CosineSim}(\mathbf{v}_{\text{dog}}, \mathbf{v}_{\text{fork}}) \approx 0.12 \implies \text{Distance is large}$.

---

## 3. How GEIF Naturally Synergizes with Embeddings

Because GEIF is built on **Voronoi bisectors** ($n = B - A$), it is uniquely suited for embedding manifolds:

1. **Meaningful Bisectors:**
   When GEIF picks sample $A = \mathbf{v}_{\text{dog}}$ and sample $B = \mathbf{v}_{\text{spoon}}$, the normal vector $n = B - A$ points directly along the **semantic concept axis** separating Animals from Cutlery!
2. **Topological Outliers in Outer Space:**
   If a system normally processes pet supplies (`dog`, `cat`, `food`, `collar`), and suddenly an unexpected query arrives (`fork`, or an exploit string `'; DROP TABLE;`), its embedding lands far outside the nominal manifold in outer space. GEIF's continuous exponential decay instantly scores it as an extreme anomaly ($0.999 \to 1.000$).

---

## 4. Architectural Integration Options

How should this be integrated into the toolchain without polluting the core C17 engine with heavy cloud SDKs or HTTP clients?

### Pattern A: The UNIX Pipeline Preprocessor (Recommended for Separation of Concerns)
Follows the UNIX philosophy: a dedicated embedding preprocessor feeds standard numerical CSV into `geif`:

```bash
# Streaming log with text -> embed column -> score with geif
cat events.csv | geif-embed --column category | geif -a -
```

* **Pros:** Keeps `libgeif` 100% pure C17, dependency-free, and blazingly fast.
* **Flexibility:** The embedding utility can switch between local CPU models and cloud APIs without recompiling GEIF.

---

### Pattern B: Unique-Category Dictionary Caching (High-Throughput Hybrid)
For streaming logs with millions of rows, calling an AI API per row is cost-prohibitive and slow (network latency $\approx 50\,\text{ms}$ per request).

However, text categories usually have a **finite vocabulary** (e.g. 50 distinct status words, or 500 product types):

```
                   Incoming CSV Stream
                           │
                 Extract Text Token (e.g. "dog")
                           │
                  In Local Hash Map?
                  ├── YES ──► Retrieve cached vector (10 nanoseconds)
                  └── NO  ──► Call AI Embedding API once, store in hash map
                           │
                 Emit Dense Float Vector to GEIF
```

* **Result:** You only call the AI API **once per unique word** (e.g. 50 API calls for a 1,000,000 row file).
* **Throughput:** Operates at native C streaming speeds for all subsequent rows.

---

### Pattern C: Local Lightweight Embedding (Zero Network Latency, Zero API Cost)
Instead of calling a cloud API over the internet:

1. **Static Vector Table (FastText / GloVe):**
   A memory-mapped binary table of word vectors in C.
   - Takes $\approx 10\,\text{ns}$ per word.
   - Zero dependencies, completely offline.
2. **Quantized Local Transformer (`all-MiniLM-L6-v2` via ONNX or llama.cpp/ggml):**
   - 22 MB model file running on CPU via SIMD/AVX2.
   - Generates 384-dimensional embeddings for arbitrary sentences and phrases at $10,000\,\text{tokens/sec}$.

---

## 5. Dimension Scaling & Matryoshka Embeddings in GEIF

Standard AI embeddings are often high-dimensional (e.g. 768 or 1536 dimensions). High dimensions increase memory usage and dot-product compute.

### 5.1 Matryoshka Embeddings (Truncation to 32D or 64D)
Modern embedding models (like Google's Gemini `text-embedding-004` and OpenAI `text-embedding-3`) are trained with **Matryoshka Representation Learning (MRL)**:
* You can take only the **first 32 or 64 dimensions** of the vector!
* The first 64 dimensions retain over **97%** of the full semantic clustering accuracy.
* Evaluating a 32D or 64D vector in GEIF is blazing fast (takes 1–2 AVX2 SIMD instructions per node).

### 5.2 Auto-Balancing with Numerical Features
When combining text embeddings ($D_{\text{text}} = 32$) with raw numerical telemetry ($D_{\text{num}} = 3$, e.g. latency, error count, payload size):
* GEIF's **pre-baked dimension normalization** ($n_j = \frac{B_j - A_j}{\text{span}_j^2}$) automatically balances the embedding dimensions with the physical telemetry dimensions.
* Neither the text embedding nor the numerical features drown each other out!
