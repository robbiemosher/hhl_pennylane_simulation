# Introduction to HHL
The HHL algorithm (named after authors Harrow, Hassidim and Lloyd who first proposed the algorithm) is designed to "solve" a linear system of equations
$$Ax = b$$. 
With $A$ an $N\times N$ matrix and $x,b\in\mathbb{C}^N$. Technically speaking $A$ must be Hermitian, but at the cost of size (needing additional qubits) one can make any matrix Hermitian simply forming a matrix $A^\prime\in M_{2N\times 2N}(\mathbb{C})$ whose upper right quadrant is $A^\dagger$ and lower left quadrant is $A$. This means $b\in C^{N}$ becomes $b^\prime = (b,\vec{0})\in C^{2N}$ and the solution $x^\prime = (x,\vec{0}) \in C^{2N}$.

Note $\vec{0} \in C^N$ is the zero vector.

**High Level Overview**

The essential idea of HHL is to perform a series of linear operations on $\ket{b}$ which represent represents the inverse of $A$, this is done primarily through quantum phase estimation, exploiting the fact: if $Ax = b$ then, if we work in the eigenbasis of $A$,

$$x = \sum_{i = 1}^{N} \dfrac{\beta_j}{\lambda_j}u_j$$

where $u_j$ are the eigenvectors of $A$ and $\lambda_j$ the associated eigenvalues. So the steps of the algorithm are as follows:

1. Prepare two registers: $\ket{b}$, the register storing the components of $b$ (in any basis, assume standard), and the estimation register $\ket{0}^{\otimes N}$.

2. Apply Hadamard gates to the estimation register to form the computational basis.

3. Form the operator representing $\exp(iAt)$ (the "time translation" operator for a system defined by $A$). We now have a \emph{unitary} operator. The process for getting this operator in real hardware is not clear.
4. Now that we have a unitary operator, we can find the eigenphases, which are one-to-one with the eigenvalues of $A$: if $\lambda_j$ is an eigenvalue of $A$, then $e^{i\lambda_jt}$ is the corresponding eigenvalue of $\exp(iAt)$.
5. The final issue is the transformation $\ket{\lambda_j}\mapsto \lambda_j^{-1}$ which is non-unitary and therefore has some probability of failure; we define the *filter function* $f: Z \rightarrow R$ which does the following,
   $$f(s) = \dfrac{1}{4\pi\kappa s}$$
in the well behaved space of the linear system.
6. We then simply uncompute the phase estimation registers and we're left with a state $\ket{x}$ representing the solution.

**DISCLAIMER**
Anyone can use the information or code contained on this page. If you have taken my code and made some improvements, or have any suggestions please reach out.

If you are using my code in an academic or professional environment please include this page OR my paper as a reference.

My thesis (undergraduate) is very accesible, and outlines my procedure for developing the code. It is free to download from https://library2.smu.ca/handle/01/32187.
