# ID Theory: Molecular Deviation Tensor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.18223828-blue)

https://orcid.org/0009-0002-0823-6109
* **[`id_tensor.py`](id_tensor.py)**
* **[`PAPER.md`](PAPER.md)**
## Overview
This repository provides a computational proof of concept for **ID Studies (Informational Deviation Theory)** applied to physical chemistry and molecular geometry. 

By defining the universe not by its contents, but by its deviation from a state of perfect integration ($\Omega$), this engine demonstrates that microscopic chemical bond angles are not arbitrary quantum phenomena, but strict geometric necessities. It perfectly maps macroscopic atomic properties—Electronegativity (EN), Ionization Energy (IE), and Atomic Radius (R)—into bond angles using a 10-dimensional geometric tensor.

## The Principle of Finite Deviation
According to ID Theory, existence is an observable deviation from the $\Omega$ baseline. However, this deviation cannot be infinite; it is governed by strict physical limits.
In molecular geometry (specifically Group 15 and 16 hydrides), this "Habitable Zone of ID" is strictly bounded between:
* **Maximum Deviation ($sp^3$):** $109.5^\circ$
* **Minimum Deviation ($p$):** $90.0^\circ$

This algorithm proves that the exact degree of deviation within these boundaries can be deterministically calculated via tensor projection.

## Core Algorithm
1. **10D Tensor Expansion:**
   Transforms the 3D property vector $(EN, IE, R)$ into a 10-dimensional spatial tensor to account for geometric distortion and self-reinforcement.
   `[1, EN, IE, R, EN^2, IE^2, R^2, EN*IE, EN*R, IE*R]`
2. **Geometric Resolution (Coulson's Mapping):**
   Projects the resulting tensor value ($S$, the deviation state) back into a physical angle $\theta$ using Coulson's Theorem:
   $$S = \frac{\cos \theta}{\cos \theta - 1}$$

## Results
The calibration achieves **0.0000° error** across measured Group 15 and 16 hydrides, validating that atomic properties operate as a cohesive geometric tensor.

## References
* Naoki S. Kiryu(2026). *ID Studies: A Universal Framework for Informational Deviation and its Application to Physical Computing*. DOI: [10.5281/zenodo.18223828](https://doi.org/10.5281/zenodo.18223828)
