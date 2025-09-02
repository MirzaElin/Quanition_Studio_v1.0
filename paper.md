---
title: "Quanition Studio v1.0: A desktop toolkit for quantum‑inspired decision modeling and economics simulations"
tags:
  - quantum cognition
  - decision theory
  - economics
  - finance
  - PySide6
  - Python
authors:
  - name: Mirza Niaz Zaman Elin
    affiliation: "1"
affiliations:
  - name: AMAL Youth & Family Centre, St. John's, NL, Canada
    index: 1
date: 2025-09-02
---

# Summary

**Quanition Studio** is a Python/PySide6 desktop application for researchers and students working at the intersection of
decision science, quantum‑inspired cognition, and economics. The software bundles six methods—(i) Hilbert‑space fitting of
conditional judgments, (ii) Quantum Decision Theory (QDT), (iii) Contextuality‑by‑Default (CbD) checks, (iv) Eisert‑type
quantum games, (v) binomial option pricing, and (vi) a simple monetary sandbox—behind an easy‑to‑use GUI. Results can be
exported as HTML reports. 

The repository provides a standard Python package, tests, and documentation to meet the Journal of Open Source Software (JOSS)
submission criteria. While the application is referred to as **Quanition Studio** in the interface, the public repository may
use the slug `quanition-studio` for packaging/installation convenience.

# Statement of need

Researchers in behavioral economics and cognitive science increasingly explore quantum‑inspired models to account for order effects,
interference, and context dependence in judgments and choices [@BusemeyerBruza2012; @YukalovSornette2016]. However, practitioners
often face a fragmented tooling landscape: small scripts are scattered across papers, GUI tools are rare, and reproducing figures
or exploring alternative parameterizations requires substantial effort. **Quanition Studio** addresses this gap by offering a single,
installable desktop application that (a) reads common data formats (CSV/XLSX), (b) runs multiple quantum‑inspired and classical
models out‑of‑the‑box, and (c) produces exportable, publication‑ready tables and figures without requiring users to assemble a
bespoke environment.

Typical user groups include: (1) cognition researchers analyzing conditional probabilities or attraction effects; (2) decision and
game theorists teaching Eisert‑type quantum games; (3) finance instructors demonstrating binomial option pricing; and (4) applied
social scientists who benefit from quick “what‑if” explorations of simple monetary dynamics.

# Functionality

The application exposes six modules and an “Auto mode” that detects what can be run directly from an imported dataset.

- **Hilbert module.** Fits conditional judgments in a two‑dimensional Hilbert space and sweeps over phase φ to match \(p(A\mid B)\),
  returning best‑fit φ and implied θ.
- **QDT module.** Computes utility factors via softmax and combines them with attraction factors, yielding normalized final choice
  probabilities [@BusemeyerBruza2012].
- **CbD module.** Evaluates the S_odd versus (2+ICC) threshold to flag contextuality in a 2×2 system [@DzhafarovKujala2016].
- **Quantum Game module.** Implements an Eisert‑style 2‑player, 2‑strategy quantized game and reports profiles that maximize
  joint payoffs [@Eisert1999].
- **QFinance module.** Prices European call options using the Cox–Ross–Rubinstein binomial tree [@CRR1979].
- **QMoney module.** A lightweight deterministic sandbox for household budget/wealth trajectories under simple behavioral rules.

The GUI (Qt/PySide6) supports CSV/XLSX import, interactive tables/plots, and one‑click **HTML** or **DOCX** report export.
A Learning Center summarizes the theoretical ideas behind each method, and built‑in documentation tabs (EULA/Privacy/Manual)
facilitate distribution in teaching/research settings.

# Design and implementation

Quanition Studio is implemented in Python 3.9+ with a Qt (PySide6) front‑end. Each method is encapsulated in a small module exposing
a single `run(cfg: dict) -> RunResult` entry point (returning a short summary string, figures, tables, and details). The GUI
orchestrates: (i) **Auto mode** that infers runnable analyses from column headers, and (ii) **Manual mode** where users provide a JSON
“spec” for a specific method. The reporter builds self‑contained HTML/DOCX artifacts for archiving or sharing.

# Quality control

The repository includes unit tests (pytest) that exercise each module on small fixtures: (1) softmax normalization in QDT, (2) detection
fields in CbD, (3) spectrum computation in quantum games, (4) price positivity in the CRR model, (5) trajectory length checks in the
monetary sandbox, and (6) a package import smoke test. These tests are designed to run headlessly in CI.

# Examples

The app ships with a one‑click sample‑data generator (Hilbert/QDT/CbD/QFinance/QMoney/QGame). Users can open a sample, run Auto mode,
inspect the result tables/plots, and export a report. Programmatic examples are provided in the README for scripting workflows.

# Availability

- **Source code:** https://github.com/USER/quanecon-studio (public repository; replace `USER` with your handle)
- **License:** MIT
- **Python:** 3.9+
- **OS:** Windows, macOS, Linux (Qt support required)
- **Dependencies:** PySide6 (Qt for Python) [@PySide6], optional `python-docx` for DOCX export

# Conflict of Interest Statement
The author declares that there are no financial or non-financial conflicts of interest relevant to this work.


# References
