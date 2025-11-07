# HHL Algorithm Simulation with PennyLane

A detailed explanation of the HHL (Harrow-Hassidim-Lloyd) algorithm for solving linear systems, along with simulation code implemented in PennyLane.

## Introduction to HHL

The HHL algorithm (named after authors Harrow, Hassidim and Lloyd who first proposed the algorithm) is designed to "solve" a linear system of equations
$$Ax = b$$

With $A$ an $N\times N$ matrix and $x,b\in\mathbb{C}^N$. Technically speaking $A$ must be Hermitian, but at the cost of size (needing additional qubits) one can make any matrix Hermitian simply by forming a matrix $A^\prime\in M_{2N\times 2N}(\mathbb{C})$ whose upper right quadrant is $A^\dagger$ and lower left quadrant is $A$. This means $b\in \mathbb{C}^{N}$ becomes $b^\prime = (b,\vec{0})\in \mathbb{C}^{2N}$ and the solution $x^\prime = (x,\vec{0}) \in \mathbb{C}^{2N}$.

Note: $\vec{0} \in \mathbb{C}^N$ is the zero vector.

## High Level Overview

The essential idea of HHL is to perform a series of linear operations on $\ket{b}$ which represent the inverse of $A$. This is done primarily through quantum phase estimation, exploiting the fact: if $Ax = b$ then, if we work in the eigenbasis of $A$,
$$x = \sum_{j = 1}^{N} \dfrac{\beta_j}{\lambda_j}u_j$$
where $u_j$ are the eigenvectors of $A$ and $\lambda_j$ the associated eigenvalues.

### Algorithm Steps

1. **Prepare two registers:** $\ket{b}$, the register storing the components of $b$ (in any basis, assume standard), and the estimation register $\ket{0}^{\otimes N}$.

2. **Apply Hadamard gates** to the estimation register to form the computational basis.

3. **Form the operator** representing $\exp(iAt)$ (the "time translation" operator for a system defined by $A$). We now have a unitary operator. The process for getting this operator in real hardware is not clear.

4. **Find the eigenphases**, which are one-to-one with the eigenvalues of $A$: if $\lambda_j$ is an eigenvalue of $A$, then $e^{i\lambda_jt}$ is the corresponding eigenvalue of $\exp(iAt)$.

5. **Apply the inversion:** The transformation $\ket{\lambda_j}\mapsto \lambda_j^{-1}$ is non-unitary and therefore has some probability of failure. We define the *filter function* $f: \mathbb{Z} \rightarrow \mathbb{R}$ which does the following,
   $$f(s) = \dfrac{1}{4\pi\kappa s}$$
   in the well-behaved space of the linear system.

6. **Uncompute the phase estimation registers** and we're left with a state $\ket{x}$ representing the solution.

## Installation

### Prerequisites
- Python 3.7 or higher
- (Optional) LaTeX distribution for plot rendering (TeX Live, MiKTeX, or MacTeX)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/robbiemosher/hhl_pennylane_simulation.git
cd hhl_pennylane_simulation
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

Download the simulation code from the [latest release](https://github.com/robbiemosher/hhl_pennylane_simulation/releases/latest) and run:

```bash
python HHL_Final_Sim.py
```

### Code Structure

The simulation performs the following:
- Implements the HHL algorithm using PennyLane's quantum computing framework
- Analyzes algorithm performance across different time constants
- Generates metrics including fidelity, success probability, and norm differences
- Creates publication-quality plots with LaTeX formatting
- Saves results as both PDF (vector) and PNG (raster) formats

### Key Functions

- `hhl_circuit()` - Implements the core HHL quantum circuit
- `data_collection_and_plotting_function()` - Main function for running simulations and generating plots
- `create_sweep_plots()` - Generates time sweep analysis plots
- `plot_eigenvalues_vs_metrics()` - Analyzes performance vs eigenvalue distribution

### Customization

You can modify the simulation parameters in the main function call at the bottom of the script:
```python
data_collection_and_plotting_function([15], 100, [4], 1)
```
Parameters:
- Matrix sizes to test
- Number of time points in sweep
- Precision qubit counts
- Number of test matrices

## Output

The simulation generates:
- **Data files** (`.npy` format): Numerical results from simulations
- **Plots** (PDF and PNG): Visualizations of fidelity, success probability, and norm differences
- **Summary text file**: Information about the simulation run

## Thesis

For a comprehensive and accessible explanation of the HHL algorithm and the development process for this code, see my undergraduate thesis:

**[Download Thesis (PDF)](https://github.com/robbiemosher/hhl_pennylane_simulation/releases/latest)** 

Also available at: [SMU Library](https://library2.smu.ca/handle/01/32187)

## License & Attribution

Anyone can use the information or code contained in this repository. If you have taken my code and made improvements, or have any suggestions, please reach out.

**If you are using this code in an academic or professional environment, please cite:**
- This repository: `https://github.com/robbiemosher/hhl_pennylane_simulation`
- Or my thesis: [available at SMU Library](https://library2.smu.ca/handle/01/32187)

## Contact

For questions, improvements, or suggestions, feel free to open an issue or reach out.

## Acknowledgments

This work was completed as part of my undergraduate thesis at Saint Mary's University, Halifax, Nova Scotia.
