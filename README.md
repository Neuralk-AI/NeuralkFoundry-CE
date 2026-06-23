<div align="center">

[![PyPI version](https://badge.fury.io/py/neuralk-foundry-ce.svg)](https://badge.fury.io/py/neuralk-foundry-ce)
[![Dashboard](https://img.shields.io/badge/dashboard-neuralk.ai-red)](https://dashboard.neuralk-ai.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.neuralk-ai.com)
[![Website](https://img.shields.io/badge/website-neuralk.ai-%2345b69c)](https://www.neuralk-ai.com)

</div>

<div align="center">
  
 [![Neuralk Foundry](https://raw.githubusercontent.com/Neuralk-AI/NeuralkFoundry-CE/main/assets/foundry_cover.png)](https://dashboard.neuralk-ai.com)

</div>

<h3 align="center">A Modular Machine Learning Framework for Industrial Tasks</h3>


<p align="center">
  <a href="https://dashboard.neuralk-ai.com"><strong>[Dashboard]</strong></a>
  <a href="./tutorials"><strong>[Examples & Tutorials]</strong></a>
</p>

---

## 🎉 Welcome to Neuralk Foundry

**Neuralk Foundry** is a lightweight yet powerful framework for building modular machine learning pipelines — particularly well-suited for industrial tasks and representation learning. Whether you're prototyping or scaling up, Foundry helps you build, combine, and orchestrate steps cleanly and efficiently.

Foundry is also the engine behind [**TabBench**](https://github.com/Neuralk-AI/TabBench), Neuralk's internal benchmark for evaluating ML models on real-world tabular datasets.

### Why Foundry?

Most ML frameworks fall into one of two camps:

* **Rigid benchmarks and academic pipelines**: great for simple supervised learning tasks, but brittle or limited when adapting to more complex use cases.
* **Heavyweight MLOps frameworks** (e.g., ZenML, Metaflow): offer full orchestration but at the cost of steep setup and reduced flexibility.

**Foundry sits in between.** It gives you just the right level of structure to scale from prototype to production — without locking you into opinionated tooling.

---

## 🚀 Key Features

**Composable Workflows**
: Define steps in terms of their inputs and outputs — no black boxes.

**Supports Heterogeneous Tasks**
: Classification, regression, ranking, record linkage, and more.

**Customizable & Extensible**
: Plug in your own logic or replace any step with a variant.

**Built-in Caching & Logging**
: Avoid recomputation and keep track of metrics automatically. Workflows
  garbage-collect overlapping outputs across steps so caches stay small,
  while preserving resume semantics through a ``postponed`` marker.

**Workflow Explorer UI**
: Inspect and debug workflows through an interactive, visual interface.

**Reproducibility by Design**
: Strong separation between configuration, code, and data.

---

## 🧠 How Things Are Organized

Foundry is a modular framework. Its codebase is split into submodules that reflect each phase of the ML pipeline:

```
neuralk_foundry_ce/
├── datasets/               # Dataset loading utilities
├── sample_selection/
│   ├── splitter/           # Data splitting strategies (e.g., stratified shuffle)
│   └── blocking/           # Candidate pair selection (e.g., for deduplication)
├── feature_engineering/
│   ├── preprocessing/      # Traditional preprocessing for tabular data
│   ├── vectorizer/         # Text and other unstructured data vectorization
│   └── blocking/           # Pair processing modules for matching/merging
├── models/
│   ├── classifier/         # Classification models
│   ├── regressor/          # Regression models
│   ├── embedder/           # Embedding/representation learning
│   └── clustering/         # Clustering and unsupervised methods
├── workflow/               # Core execution engine: Step, Workflow, etc.
└── utils/                  # Helper functions and shared infrastructure
```

Each component (e.g., a model or preprocessing step) inherits from a base `Step` class and declares:

* Its expected **inputs**
* The **outputs** it produces
* Any configurable **parameters**

Steps can then be connected into a `Workflow`, either manually or through a task-specific template (e.g., `Classification`).

---

## ⚙️ Quick-Start Installation

Install the package from PyPI:

```bash
pip install neuralk_foundry_ce
```

## 🔬 Development Installation

### Clone the Repository

```bash
git clone https://github.com/Neuralk-AI/NeuralkFoundry-CE
cd NeuralkFoundry-CE
```

### Create a Dedicated Environment (recommended)

Neuralk Foundry relies on a variety of external machine learning libraries. As a result, managing package versions can be delicate. To avoid compatibility issues, we strongly recommend installing Foundry in a dedicated virtual environment (e.g., using conda or venv).

```bash
conda create -n foundry python=3.11
conda activate foundry
```

### Install the Package

```bash
pip install -e .
```

## Examples and tutorials

* [**Getting Started with Neuralk Foundry**](tutorials/1%20-%20Getting%20Started%20with%20Neuralk%20Foundry.ipynb)
  A gentle introduction to the framework and how to run your first workflow.

* [**Three Levels of Workflows**](tutorials/2%20-%20Three%20levels%20of%20workflows.ipynb)
  Understand how Foundry supports simple pipelines, reusable workflows, and specialized task flows.

* [**Use a Custom Model**](tutorials/3%20-%20Use%20a%20custom%20model.ipynb)
  Learn how to plug in and use your own ML model within a Foundry pipeline.

## Citing Foundry

If you incorporate any part of this repository into your work, please reference it using the following citation:

```bibtex
@article{neuralk2025foundry,
         title={Foundry: A Modular Machine Learning Framework for Industrial Tasks}, 
         author={Neuralk-AI},
         year={2025},
         publisher = {GitHub},
         howpublished = {\url{https://github.com/Neuralk-AI/NeuralkFoundry-CE}},
}
```

# Contact

If you have any questions or wish to propose new features please feel free to open an issue or contact us at alex@neuralk-ai.com.  

For collaborations please contact us at antoine@neuralk-ai.com.  
