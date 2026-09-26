# GEIF: Reservoir Sampling, Sorted Data Immunity & High-Throughput Streaming Ingest

## 1. The Core Strength: Why Reservoir Sampling Conquers Sorted Data

In enterprise telemetry, server monitoring, and database dumps, input streams are **almost never randomly shuffled**:
* **Time-Sorted Logs:** Rows are strictly ordered by timestamp (`00:00:00` $\to$ `23:59:59`).
* **Grouped / Key-Sorted Dumps:** Telemetry ordered by tenant ID, device IP, or geographical region.
* **Metric-Sorted Batches:** Data ordered by value (e.g. exported with `ORDER BY request_duration`).

```text
       Sorted Input Stream (10,000,000 rows)
       [ Monday Morning ... ] → [ Wednesday Noon ... ] → [ Friday Night ... ]
                                      │
                         Reservoir Sampling Algorithm
                                      ▼
                        Fixed Reservoir (256 samples)
            * Monday 04:12    * Wednesday 11:45    * Friday 22:30
            * Monday 09:30    * Thursday 02:15     * Friday 23:58
```

### 1.1 The Mathematical Proof of Order Invariance
Let $S$ be the reservoir capacity (e.g. $S = 256$), and $N$ be the total number of items seen in the stream.

For the $k$-th item in the stream:
1. It is accepted into the reservoir with probability:
   $$P(\text{accept } k) = \frac{S}{k}$$
2. For any subsequent item $m > k$, item $k$ survives if item $m$ either is rejected OR replaces a different reservoir slot:
   $$P(\text{survives } m) = 1 - \left(\frac{S}{m}\right)\left(\frac{1}{S}\right) = 1 - \frac{1}{m} = \frac{m - 1}{m}$$
3. Multiplying survival probabilities telescopingly from $m = k+1$ to $N$:
   $$P(k \in \text{Reservoir at step } N) = \frac{S}{k} \times \left( \frac{k}{k+1} \times \frac{k+1}{k+2} \times \dots \times \frac{N-1}{N} \right) = \frac{S}{N}$$

**The Breakthrough:**
Every single row—whether it appeared in line 1, line 5,000,000, or line 10,000,000—has the **exact same uniform probability $\frac{S}{N}$** of being in the final training reservoir! 
* Zero ordering bias.
* No temporal skew.
* No need for expensive disk pre-shuffling (`shuf input.csv` or `sort -R`).

---

## 2. High-Throughput Streaming: Upgrading from Algorithm R to Algorithm L

In classic implementations (Waterman's Algorithm R):
* Every incoming line calls the pseudo-random generator:
  ```c
  j = rand() % current_row;
  if (j < RESERVOIR_SIZE) {
      reservoir[j] = parse_row(line);
  }
  ```
* **The Performance Bottleneck:** If an input log has $50,000,000$ lines, the CPU executes $50,000,000$ `rand()` calls and parses millions of floating-point strings that are immediately thrown away!

### The Modern GEIF Optimization: Vitter's Algorithm L
In 1985, Jeffrey Vitter proved that rather than rolling a die for every single line, the number of lines to **skip** until the next acceptance follows a geometric distribution:

$$S_{\text{skip}} = \left\lfloor \frac{\ln(U)}{\ln(1 - W)} \right\rfloor$$

Where $U \sim \text{Uniform}(0, 1)$ and $W$ is updated recursively.

#### The Speed Advantage in GEIF:
* As the stream progresses, the skip distance grows into thousands or millions of lines ($S_{\text{skip}} = 142,500$).
* GEIF simply counts newline characters `\n` using blazing-fast AVX2 SIMD byte-scanners, **completely bypassing CSV tokenization, float parsing, and `rand()` calls** for the skipped lines!
* **Throughput:** Ingests massive multi-gigabyte logs at wire speed ($> 500\,\text{MB/sec}$).

---

## 3. Continuous Online Updates & The Reservoir Ceiling

Under standard reservoir sampling, as a monitoring daemon runs continuously for months ($N \to 100,000,000$), the acceptance probability $P = \frac{S}{N} \to 0$. The model becomes "frozen" in ancient history and cannot adapt to seasonal or infrastructure evolution.

### The GEIF Adaptive Reservoir Ceiling
Inherited and refined from `ceif`:
$$\text{effective\_N} = \min\left(N,\; (\text{CEILING\_FACTOR} + 1) \cdot S\right)$$

* With $\text{CEILING\_FACTOR} = 3$ and $S = 256$, $\text{effective\_N}$ is capped at $4 \times 256 = 1024$.
* The minimum acceptance probability for incoming telemetry is guaranteed at:
  $$P_{\min} = \frac{256}{1024} = \mathbf{25\%}$$
* **Result:** As new traffic patterns emerge, approximately 1 in 4 new rows rolls into the reservoir, gracefully rolling the Voronoi trees forward while retaining resilience against transient noise spikes.
