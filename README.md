# TwoDG.jl

[![Build Status](https://github.com/xkykai/TwoDG.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/xkykai/TwoDG.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance Julia framework for solving 2D partial differential equations using advanced finite element methods.

## Cool Visuals

<p align="center">
  <img src="figures/eulerchannel_machnumber.gif" height="300" />
  <br>
  <em>Compressible flow through a channel with a bump computed with 2D Euler equations showing evolution of Mach number</em>
</p>

<p align="center">
  <img src="figures/cp_trefftz_10.png" height="350" />
  <img src="figures/hdg_convdiff_ustar_size_0.2_k_1_c_10_10_p_4.png" height="350" />
  <br>
  <em>Pressure coefficient of a potential flow solution (left) and convection-diffusion solution on an unstructured mesh with Hybridizable Discontinuous Galerkin (HDG) (right)</em>
</p>


## Overview

TwoDG.jl provides a unified implementation of three powerful discretization schemes for 2D PDEs:

- **Continuous Galerkin (CG)** - Traditional continuous finite elements
- **Discontinuous Galerkin (DG)** - High-order explicit time-stepping methods
- **Hybridizable Discontinuous Galerkin (HDG)** - Efficient implicit solvers with static condensation and parallel capabilities

Whether you're studying wave propagation, compressible flows, or convection-diffusion phenomena, TwoDG.jl offers the numerical tools to tackle these problems with arbitrary polynomial orders and sophisticated error analysis.

## Key Features

- **Multiple PDE Types**: Poisson, convection-diffusion, wave equations, and Euler equations
- **High-Order Accuracy**: Arbitrary polynomial order support (p-refinement) with Koornwinder orthogonal basis
- **Efficient HDG**: Static condensation dramatically reduces system size compared to standard DG
- **Parallel HDG Solver**: Multi-threaded assembly and solving for large-scale problems
- **Rich Mesh Support**: Built-in generators for squares, circles, L-shapes, NACA airfoils, and more
- **Comprehensive Examples**: Convergence studies, wave scattering, channel flows, and transport problems
- **Performance Optimized**: Inline functions and careful pre-allocation for computational efficiency

## What Can You Do With It?

- **Run convergence studies** to verify optimal rates across different polynomial orders
- **Compare discretization methods** (CG vs DG vs HDG) on the same problems
- **Simulate wave scattering** on complex geometries with absorption boundaries
- **Solve compressible flow** problems including shock waves in channels
- **Analyze convection-diffusion** transport with various stabilization parameters
- **Develop new numerical methods** using the extensible master element framework

## Quick Example

```julia
using TwoDG

# Generate a mesh for your domain
mesh = squaremesh_rsquare(1.0, 5)  # 1x1 square, 5 elements per side

# Solve a Poisson problem with HDG
# (Check out examples in src/Apps/ for complete scripts)
```

## Getting Started

Explore the example scripts in `src/Apps/` to see the solvers in action:
- `runhdg_poisson.jl` - Poisson equation convergence studies
- `runwavescattering.jl` - Wave scattering on circular domains
- `runeulerchannel.jl` - Compressible Euler equations with shocks
- `runconvection.jl` - Pure convection with DG explicit time-stepping

Perfect for researchers in numerical analysis, students learning finite element methods, or anyone needing a flexible high-order PDE solver in Julia.