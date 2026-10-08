# GEIF Test & Benchmark Datasets

This directory contains benchmark and integration testing datasets used by the GEIF test suite (`make test`, `make test-prod`).

---

## License & Attribution

The datasets in this directory (`paddydataset.csv` and `winequality-red.csv`) are licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

> **License Notice:**
> This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
> This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.
> 
> Full license deed: https://creativecommons.org/licenses/by/4.0/

---

## Datasets Overview

### 1. Wine Quality Dataset (`winequality-red.csv`)
- **Dimensions:** 11 continuous physicochemical features + 1 quality score column (semicolon-delimited `;`).
- **Rows:** 1,599 observations.
- **Usage in GEIF:** Used for single-column category routing (`-C 12`), single-dimension attribution (`%e`), scale recalibration, and continuous metric depth testing.
- **Citation:**
  > P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis.  
  > *Modeling wine preferences by data mining from physicochemical properties.*  
  > Decision Support Systems, Elsevier, 47(4):547-553, 2009.  
  > UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/186/wine+quality

### 2. Paddy Crop Agricultural Dataset (`paddydataset.csv`)
- **Dimensions:** 45 mixed categorical, meteorological, soil, and agricultural feature columns (comma-delimited `,`).
- **Rows:** 2,790 observations across multiple administrative blocks (`Agriblock`), varieties (`Variety`), and soil types (`Soil Types`).
- **Usage in GEIF:** Used for multi-column categorical ensemble routing (`-C 2-4`), category regex filtering (`-F`), label isolation (`-L 1`), ignoring fields (`-I 35-39`) and high-dimensional streaming ingest.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
