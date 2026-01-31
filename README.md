# M5GP 2.0
**Extensions and Enhancements to a Constructive Feature Induction System based on Genetic Programming**

---

## Overview

**M5GP 2.0** is an open-source implementation of an advanced **Multidimensional Genetic Programming (GP)** framework designed for **symbolic regression** and **constructive feature induction**, with a strong focus on **interpretability, scalability, and GPU acceleration**.

M5GP 2.0 **derives conceptually from the original M5GP framework**, but it is released as an **independent repository**, reflecting a substantial redesign and consolidation of the algorithm, its evolutionary operators, numerical stability mechanisms, and GPU execution model.

> **Note**  
> Although M5GP 2.0 builds upon the ideas introduced in M5GP (v1), **this repository is fully independent**, with a clean history and a restructured codebase targeting large-scale, reproducible experimentation.

---

## Key Concepts

M5GP 2.0 follows the **constructive feature induction paradigm**, where Genetic Programming is used to evolve **symbolic transformations of the input space**, and the resulting features are subsequently combined using **linear models** (e.g., Linear Regression or Ridge Regression).

This hybrid symbolic–linear approach enables:
- Compact and interpretable symbolic models
- Competitive predictive performance
- Efficient large-scale execution on GPUs

---

## Main Differences vs. M5GP (v1)

### Algorithmic Enhancements
- Adaptive **UMAD mutation operator** guided by function-level performance feedback
- Expanded symbolic language with high-arity aggregation operators:
  - `SUM`, `PRD`, `AVG`, `DSV`
- Explicit numerical-stability mechanisms:
  - Input/output normalization
  - Bounded random constants
  - Inclusion of fundamental constants (π and e)
- **Ridge Regression (L2 regularization)** for improved robustness under multicollinearity

### GPU and Systems-Level Improvements
- Redesigned GPU execution model using **Numba CUDA JIT kernels**
- Optimized GPU memory management, batching, and buffer reuse
- Standardized **float32 arithmetic** for numerical consistency
- Reduced CPU–GPU data transfers and improved parallel efficiency

### Experimental Scope
- Rigorous evaluation using **SRBench**
- Reproducible configurations and scripts
- Support for large-scale symbolic regression benchmarks

---

## Software Stack

```
Python, NumPy, SciKit-Learn, Numba, CUDA, RAPIDS cuML, SRBench, DIGEN
```

---

## Requirements

### Software
- Python ≥ 3.8
- Conda ≥ 23.x
- RAPIDS cuML
- Numba
- scikit-learn
- scikit-cuda
- PyTorch (optional)

### Hardware
- NVIDIA GPU with CUDA support
- CUDA Toolkit compatible with RAPIDS version

---

## Environment Setup (Recommended)

```bash
conda create -n rapids-25.04 \
  -c rapidsai -c conda-forge -c nvidia \
  rapids=25.04 python=3.12 cudatoolkit=11.5
conda activate rapids-25.04
```

```bash
pip install scikit-cuda
conda install -c conda-forge scikit-learn
conda install pytorch::pytorch
```

---

## Installation

```bash
git clone https://github.com/armandocardenasf/m5gp-2.0.git
cd m5gp-2.0
```

---

## Execution

### Basic execution
```bash
python m5gp.py
```

### Test execution
```bash
python m5gp_test.py
```

---

## Benchmarks and Reproducibility

The repository provides scripts and configurations for:
- **SRBench** (symbolic regression)
- **DIGEN** (classification benchmarks)

All experiments reported in the associated paper can be reproduced using the provided configurations.

---

## Documentation

The codebase follows a **Doxygen-style documentation structure**, enabling rapid onboarding and direct use after minimal reading.

---

## References

If you use this software, please cite:

Cárdenas Florido, L., Trujillo, L., et al.  
*M5GP 2.0: Extensions and Enhancements to a Constructive Feature Induction System based on Genetic Programming*  
(Manuscript under review, 2026)

Original M5GP reference:
- Cárdenas Florido, L. et al.  
  *M5GP: Parallel Multidimensional Genetic Programming for Symbolic Regression*  
  Mathematical and Computational Applications, MDPI, 2024.  
  https://doi.org/10.3390/mca29020025

---

## Institutions

- Tecnológico Nacional de México / IT de Ensenada
- Tecnológico Nacional de México / IT de Tijuana
- Tecnológico Nacional de México / IT de La Paz

---

## Funding

This research is supported by **TecNM (Mexico)**.

---

## License

To be defined before the first stable public release.

