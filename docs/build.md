# GEIF: Build System & Compilation Guide

**Author / Maintainer:** Timo Savinen (AI-assisted)

This guide documents the prerequisites, compilation targets, and testing workflows for **GEIF (Geometric Extended Isolation Forest)**.

---

## 1. Prerequisites & Dependencies

GEIF is implemented in strict ISO C17 and depends only on standard POSIX system libraries and `json-c`:

| Requirement | Minimum Version | Purpose | Installation (Debian/Ubuntu) | Installation (RHEL/Fedora) |
| :--- | :---: | :--- | :--- | :--- |
| **C Compiler** | GCC 9+ or Clang 10+ | ISO C17 compiler | `sudo apt install build-essential` | `sudo dnf install gcc make` |
| **Make** | GNU Make 4.0+ | Build automation | Included in build-essential | Included in make |
| **json-c** | 0.13+ | JSON model I/O | `sudo apt install libjson-c-dev` | `sudo dnf install json-c-devel` |
| **gnuplot** (optional) | 5.0+ | Heatmap rendering | `sudo apt install gnuplot` | `sudo dnf install gnuplot` |

---

## 2. Compilation Targets

The root `Makefile` provides standardized build targets:

```bash
# Compile core libraries, CLI executable, and migration tool
make

# Run the comprehensive test suite (unit + integration tests)
make test

# Install binaries and libraries to /usr/local (or PREFIX=<path>)
sudo make install

# Clean all build artifacts, object files, and binaries
make clean
```

### Generated Artifacts:
- `lib/libgeif.a`: Static C17 library for static linking.
- `lib/libgeif.so`: Dynamically linked shared library.
- `bin/geif`: Primary command-line interface executable.
- `bin/ceif2geif`: Model migration utility for converting legacy models.

---

## 3. Compiler Flags & Optimization Settings

By default, the Makefile uses aggressive performance optimizations:

```makefile
CC       = gcc
CFLAGS   = -std=c17 -O3 -march=native -Wall -Wextra -Wpedantic -fPIC -D_GNU_SOURCE
INCLUDES = -Iinclude -Isrc/lib -Isrc/cli
LDFLAGS  = -lm -ljson-c
```

### Compiler Features Enabled:
- `-std=c17`: Strict ISO C17 conformance.
- `-O3 -march=native`: Loop vectorization, inlining, and native CPU instruction set extensions (AVX2, FMA).
- `-Wall -Wextra -Wpedantic`: Rigorous compiler warning checks.
- `-fPIC`: Position-independent code generation for shared library creation.

---

## 4. Verification & Automated Testing

GEIF includes 16 automated test suites (3 C unit test binaries and 13 CLI integration/benchmark suites) verifying mathematical accuracy, CLI interfaces, and performance:

```bash
make test
```

### Test Suites Executed:
1. `test_voronoi`: Mathematical unit tests for perpendicular bisectors and 5000:1 aspect ratio invariance.
2. `test_stadium`: Mathematical unit tests for continuous Euclidean stadium outer space decay.
3. `test_negative`: Negative testing for corrupted inputs, empty files, and zero-variance columns.
4. `test_cli_lifecycle.sh`: End-to-end training, saving, loading, and scoring pipeline tests.
5. `test_cli_categories.sh`: Multi-category routing, label extraction, and category filtering.
6. `test_cli_recalibration.sh`: Outlier pruning (`-k`), percentage thresholds (`0%`–`200%`), and scale recalibration.
7. `test_cli_rcfile.sh`: Cascading configuration file parser (`~/.geifrc`, `-g`).
8. `test_cli_grid.sh`: Synthetic test grid generation and population drift visualization.
9. `test_cli_ceif2geif.sh`: Legacy model migration and transparent ingestion.
10. `test_cli_algorithms.sh`: Multi-algorithm engine test suite (`bubble`, `voronoi`, `exemplar`, `ceif`).
11. `test_cli_csv.sh`: CSV dialect, custom delimiter, header skipping, and locale formatting tests.
12. `test_cli_columns.sh`: Column inclusion (`-U`), ignore (`-I`), label (`-L`), and category (`-C`) routing tests.
13. `test_cli_attribution.sh`: Single-dimension anomaly attribution (`%e`) and expansion (`-j`) tests.
14. `test_cli_decay.sh`: Exponential sample age decay (`-D`) and rolling reservoir update tests.
15. `test_cli_aggregate.sh`: Category value aggregation (`-A`, `--aggregate`) tests.
16. `test_ref_bubble.sh`: Hyperspherical Bubble reference benchmark suite and cavity verification (plus `test_prod_cron_patterns.sh` via `make test-prod`).

---

## Development & Attribution

**GEIF** was designed and engineered by **Timo Savinen** with AI pair-programming assistance (Google Antigravity / Gemini) under human architectural direction.
