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
    orcid: 0000-0001-9577-7821
    affiliation: "1"
affiliations:
  - name: "AMAL Youth and Family Centre, St. John's, NL, Canada"
    index: 1
date: 2025-09-10
bibliography: paper.bib
---

# Summary

Quanition Studio v1.0.0 is a small, reusable Python library—plus a command-line interface—that exposes headless implementations of several compact analyses originally bundled in a GUI: a Hilbert-space fit for conditional probabilities, a Quantum Decision Theory (QDT) utility-plus-attraction model, a Contextuality-by-Default (CbD) check, a lightweight Eisert-style quantum game explorer, a Cox–Ross–Rubinstein binomial option pricer, and a toy monetary sandbox. The package is designed for transparent, reproducible research: it provides an importable API and a simple CLI, includes examples for quick trials, and ships with unit tests and GitHub Actions CI. By separating computation from any user interface and keeping dependencies light, QuanEcon Core enables scripting in notebooks and pipelines, classroom demonstrations, and small research prototypes without coupling to a desktop app.

# Statement of need

Researchers, students, and instructors who work with quantum-like models often face a gap between what the literature explains and what is easily runnable in a notebook or a simple script. Core ideas such as Quantum Decision Theory, Contextuality-by-Default, Hilbert space interference models, and basic quantum game formalisms are well established conceptually [@busemeyer2012; @yukalov2016; @dzhafarov2016; @eisert1999], but practical, minimal examples are scattered across papers [@popp2023; @russo2021; @gray2018; @gupt2019; @roy2023], tied to heavyweight graphical tools, or reimplemented ad hoc without tests. This fragmentation raises the cost of reproducing small figures, preparing teaching materials, and running quick experiments in the familiar NumPy/pandas workflow [@harris2020; @mckinney2010].

Quanition Studio v1.0.0 addresses this need by providing a lightweight, headless library with a very small and consistent Python API. The goal is not to introduce new theory, but to package a few recurring computations in a form that is easy to import, easy to read, and easy to test. The library includes: a Hilbert space phase-fit for conditional probabilities; a Quantum Decision Theory routine that combines utility terms with attraction adjustments; a Contextuality-by-Default checker based on standard summary indices; a compact quantum-game explorer following the Eisert–Wilkens–Lewenstein setup with three strategies per player; a Cox–Ross–Rubinstein binomial option pricer as a finance baseline [@cox1979]; and a small money/household sandbox. Each module is intentionally narrow so that users can call one function with plain arrays or Series and get a result suitable for notebooks, slides, or automated tests.

Compared with general-purpose scientific libraries, Quanition Studio v1.0.0 supplies domain-specific glue that users typically rewrite: input shape checks, normalization and consistency tests, numerically sensible defaults, and tiny convenience helpers that make short scripts reliable. Compared with GUI-first tools, it cleanly separates computation from presentation, which enables headless execution in continuous integration and in batch runs, while remaining straightforward to wrap with a classroom GUI if needed.

The intended audience includes instructors preparing short demonstrations, researchers who want unit-tested baselines for replication and ablation studies, and students learning by reproducing canonical results without installing large desktop stacks. By shipping examples, tests, and continuous integration alongside a compact API, Quanition Studio v1.0.0 lowers the activation energy for reproducible teaching and small research prototypes in this space.

# Quality control

Unit tests cover numerical sanity for each module (probability normalization, thresholds, payoff search, positive option price, trajectory length). GitHub Actions runs tests on each push/PR. CSV demos are included for quick trials.

# State of the field

The package is a pragmatic, educational collection rather than a novel contribution. It draws on standard references in quantum cognition [@busemeyer2012; @yukalov2016], contextuality [@dzhafarov2016], quantum games [@eisert1999], and option pricing [@cox1979], and relies on the scientific Python stack [@harris2020; @mckinney2010].


# Conflict of interest

The author declares no competing interests.
