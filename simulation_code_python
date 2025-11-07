import numpy as np
import pennylane as qml
import scipy.linalg
import matplotlib.pyplot as plt
import os
import time
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern'  # This is the default LaTeX font
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'
#For later
save_path = os.path.dirname(os.path.abspath(__file__))

def create_plot(x_data, y_data, filename_prefix='plot', 
                x_label=r'$x$ (arb. units)', 
                y_label=r'$y$ (arb. units)',
                title=r'Plot with \LaTeX\ Formatting',
                best_time=None):
    """
    Create and save a plot with LaTeX styling.
    All artifacts (legends, text boxes) removed, only keeping LaTeX formatting.
    Includes vertical line for best time if provided.
    
    Parameters:
    -----------
    x_data : array-like
        x-axis data
    y_data : array-like
        y-axis data
    filename_prefix : str
        Prefix for the saved files
    x_label : str
        Label for x-axis (LaTeX formatting supported)
    y_label : str
        Label for y-axis (LaTeX formatting supported)
    title : str
        Plot title (LaTeX formatting supported)
    best_time : float, optional
        If provided, adds a vertical line at this time value
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the data - no legend or labels
    ax.plot(x_data, y_data, 'b-', linewidth=2)
    
    # Add vertical line for best time if provided
    if best_time is not None:
        ax.axvline(x=best_time, color='purple', linestyle='--')
    
    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save as PDF (vector) and PNG (raster)
    plt.savefig(f'{filename_prefix}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename_prefix}.png', dpi=300, bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close(fig)

def create_sweep_plots(t_space, fidelity_array, success_probability_array, norm_difference_array, best_time_constant, save_prefix):
    """
    Create and save plots for the HHL time sweep data.
    All artifacts (legends, text boxes) removed, only keeping LaTeX formatting.
    Includes vertical line for best time on all plots.
    
    Parameters:
    -----------
    t_space : array-like
        Time constants used in the simulation
    fidelity_array : array-like
        Fidelity values for each time constant
    success_probability_array : array-like
        Success probability values for each time constant
    norm_difference_array : array-like
        Norm difference values for each time constant
    best_time_constant : float
        The optimal time constant found
    save_prefix : str
        Prefix for saved files (e.g., 'matrix1_q3_sine')
    """
    # Fidelity plot
    create_plot(
        t_space, fidelity_array,
        filename_prefix=f'{save_prefix}_fidelity',
        x_label=r'Time Constant ($t_0$)',
        y_label=r'Fidelity ($\mathcal{F}$)',
        title=r'Fidelity vs Time Constant',
        best_time=best_time_constant
    )
    
    # Success probability plot
    create_plot(
        t_space, success_probability_array,
        filename_prefix=f'{save_prefix}_success',
        x_label=r'Time Constant ($t_0$)',
        y_label=r'Success Probability ($P$)',
        title=r'Success Probability vs Time Constant',
        best_time=best_time_constant
    )
    
    # Norm difference plot
    create_plot(
        t_space, norm_difference_array,
        filename_prefix=f'{save_prefix}_norm_diff',
        x_label=r'Time Constant ($t_0$)',
        y_label=r'Norm Difference ($|\Psi - \Phi|$)',
        title=r'Norm Difference vs Time Constant',
        best_time=best_time_constant
    )
    
    # Combined metrics plot - without legend
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all metrics on the same axis - no legends
    ax.plot(t_space, fidelity_array, 'o-', color='blue', markersize=4)
    ax.plot(t_space, success_probability_array, 's-', color='green', markersize=4)
    ax.plot(t_space, norm_difference_array, '^-', color='red', markersize=4)
    
    # Add vertical line for the best time constant - no label
    ax.axvline(x=best_time_constant, color='purple', linestyle='--')
    
    # Set labels and title
    ax.set_xlabel(r'Time Constant ($t_0$)')
    ax.set_ylabel(r'Metric Value')
    ax.set_title(r'Algorithm Metrics vs Time Constant')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_combined.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_prefix}_combined.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_eigenvalues_vs_metrics(matrices_list, best_eig_data, precision_qubit_range, use_sine_values=[True, False]):
    """
    Create plots comparing eigenvalues to metrics like fidelity and success probability.
    All artifacts (legends, text boxes) removed, only keeping LaTeX formatting.
    No best time line for eigenvector plots.
    
    Parameters:
    -----------
    matrices_list : list
        List of matrices used in the simulation
    best_eig_data : numpy.ndarray
        Array containing the best results for each eigen-analysis
    precision_qubit_range : list
        List of precision qubit counts used
    use_sine_values : list
        List of use_sine values used (typically [True, False])
    """
    for sin_idx, use_sine in enumerate(use_sine_values):
        sine_label = "sine" if use_sine else "hadamard"
        
        for q, qubits in enumerate(precision_qubit_range):
            for m, matrix in enumerate(matrices_list):
                # Get eigenvalues for this matrix
                eig_vals, _ = np.linalg.eigh(matrix)
                
                # Extract data for valid eigenvalues (up to matrix size)
                valid_indices = range(matrix.shape[0])
                
                # Extract metrics for these eigenvalues
                best_fidelities = best_eig_data[sin_idx, q, m, valid_indices, 1]
                best_probs = best_eig_data[sin_idx, q, m, valid_indices, 3]
                
                # Plot fidelity vs eigenvalues
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(eig_vals, best_fidelities, 'o-', linewidth=2, color='blue')
                
                # Set labels and title
                ax.set_xlabel(r'Eigenvalue ($\lambda_i$)')
                ax.set_ylabel(r'Fidelity ($\mathcal{F}$)')
                ax.set_title(r'Fidelity vs Eigenvalues')
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Save the plot
                filename = f'eigen_fidelity_m{m}_q{qubits}_{sine_label}'
                plt.tight_layout()
                plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Plot success probability vs eigenvalues
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(eig_vals, best_probs, 's-', linewidth=2, color='green')
                
                # Set labels and title
                ax.set_xlabel(r'Eigenvalue ($\lambda_i$)')
                ax.set_ylabel(r'Success Probability ($P$)')
                ax.set_title(r'Success Probability vs Eigenvalues')
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Save the plot
                filename = f'eigen_success_m{m}_q{qubits}_{sine_label}'
                plt.tight_layout()
                plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
                plt.close(fig)

def plot_condition_number_analysis(matrices_list, best_b_data, t_sweep_b_data, precision_qubit_range, t_space, b_vectors, use_sine_values=[True, False]):
    """
    Create plots analyzing the relationship between condition number and HHL performance.
    All artifacts (legends, text boxes) removed, only keeping LaTeX formatting.
    
    Parameters:
    -----------
    matrices_list : list
        List of matrices used in the simulation
    best_b_data : numpy.ndarray
        Array containing the best results for each b-vector analysis
    t_sweep_b_data : numpy.ndarray
        Array containing the time sweep data for each b-vector
    precision_qubit_range : list
        List of precision qubit counts used
    t_space : numpy.ndarray
        Array of time constants used
    b_vectors : list or array
        List of b-vectors used
    use_sine_values : list
        List of use_sine values used (typically [True, False])
    """
    # Calculate condition numbers for all matrices
    condition_numbers = np.array([np.linalg.cond(matrix) for matrix in matrices_list])
    
    for sin_idx, use_sine in enumerate(use_sine_values):
        sine_label = "sine" if use_sine else "hadamard"
        
        for q, qubits in enumerate(precision_qubit_range):
            # Initialize arrays to store aggregated metrics for each matrix
            avg_fidelities = np.zeros(len(matrices_list))
            avg_success_probs = np.zeros(len(matrices_list))
            
            # For each matrix, calculate average metrics across compatible b-vectors
            for m, matrix in enumerate(matrices_list):
                valid_metrics = []
                valid_probs = []
                
                # Collect metrics for all compatible b-vectors
                for b, vector in enumerate(b_vectors):
                    # Skip incompatible dimensions
                    if matrix.shape[1] != len(vector):
                        continue
                        
                    fidelity = best_b_data[sin_idx, q, m, b, 1]
                    success_prob = best_b_data[sin_idx, q, m, b, 3]
                    
                    # Only add non-NaN values
                    if not np.isnan(fidelity) and not np.isnan(success_prob):
                        valid_metrics.append(fidelity)
                        valid_probs.append(success_prob)
                
                # Calculate averages if we have valid data
                if valid_metrics:
                    avg_fidelities[m] = np.mean(valid_metrics)
                    avg_success_probs[m] = np.mean(valid_probs)
                else:
                    avg_fidelities[m] = np.nan
                    avg_success_probs[m] = np.nan
            
            # Plot condition number vs average fidelity
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Remove NaN entries
            valid_indices = ~np.isnan(avg_fidelities)
            if np.any(valid_indices):
                # Sort by condition number for line plotting
                sorted_indices = np.argsort(condition_numbers[valid_indices])
                sorted_cond = condition_numbers[valid_indices][sorted_indices]
                sorted_fidelity = avg_fidelities[valid_indices][sorted_indices]
                
                # Plot using log scale for condition number
                ax.semilogx(sorted_cond, sorted_fidelity, 'o-', linewidth=2, color='blue')
                
                # Set labels and title
                ax.set_xlabel(r'Condition Number ($\kappa$)')
                ax.set_ylabel(r'Average Fidelity ($\bar{\mathcal{F}}$)')
                ax.set_title(r'Fidelity vs Condition Number')
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Save the plot
                filename = f'cond_fidelity_q{qubits}_{sine_label}'
                plt.tight_layout()
                plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                # Plot condition number vs average success probability
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Sort by condition number for line plotting
                sorted_prob = avg_success_probs[valid_indices][sorted_indices]
                
                # Plot using log scale for condition number
                ax.semilogx(sorted_cond, sorted_prob, 's-', linewidth=2, color='green')
                
                # Set labels and title
                ax.set_xlabel(r'Condition Number ($\kappa$)')
                ax.set_ylabel(r'Average Success Probability ($P$)')
                ax.set_title(r'Success Probability vs Condition Number')
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Save the plot
                filename = f'cond_success_q{qubits}_{sine_label}'
                plt.tight_layout()
                plt.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight')
                plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
                plt.close(fig)

def generate_all_plots(matrices_list, best_eig_data, best_b_data, t_sweep_eig_data, 
                       t_sweep_b_data, precision_qubit_range, t_space, b_vectors):
    """
    Generate all plots for the HHL simulation results.
    
    This function calls the other plotting functions to create a comprehensive set of visualizations.
    """
    print("Generating eigenvalue analysis plots...")
    plot_eigenvalues_vs_metrics(matrices_list, best_eig_data, precision_qubit_range)
    
    print("Generating condition number analysis plots...")
    plot_condition_number_analysis(matrices_list, best_b_data, t_sweep_b_data, 
                                 precision_qubit_range, t_space, b_vectors)
    
    print("Generating time sweep plots for each matrix and eigenvector...")
    for sin_idx, use_sine in enumerate([True, False]):
        sine_label = "sine" if use_sine else "hadamard"
        
        for q, qubits in enumerate(precision_qubit_range):
            for m, matrix in enumerate(matrices_list):
                # Generate plots for each eigenvector
                for e in range(matrix.shape[0]):
                    # Extract sweep data for this eigenvector
                    f_array = t_sweep_eig_data[sin_idx, q, m, e, 0, :]
                    p_array = t_sweep_eig_data[sin_idx, q, m, e, 1, :]
                    diff_array = t_sweep_eig_data[sin_idx, q, m, e, 2, :]
                    
                    # Get best values
                    best_t = best_eig_data[sin_idx, q, m, e, 0]
                    
                    # Generate plots
                    save_prefix = f'eigen_m{m}_q{qubits}_e{e}_{sine_label}'
                    create_sweep_plots(t_space, f_array, p_array, diff_array, best_t, save_prefix)
                
                # Generate plots for each compatible b-vector
                for b, vector in enumerate(b_vectors):
                    # Skip incompatible dimensions
                    if matrix.shape[1] != len(vector):
                        continue
                    
                    # Extract sweep data for this b-vector
                    f_array = t_sweep_b_data[sin_idx, q, m, b, 0, :]
                    diff_array = t_sweep_b_data[sin_idx, q, m, b, 1, :]
                    p_array = t_sweep_b_data[sin_idx, q, m, b, 2, :]
                    
                    # Get best values
                    best_t = best_b_data[sin_idx, q, m, b, 0]
                    
                    # Generate plots
                    save_prefix = f'bvec_m{m}_q{qubits}_b{b}_{sine_label}'
                    create_sweep_plots(t_space, f_array, p_array, diff_array, best_t, save_prefix)
    
    print("All plots generated successfully.")


"""
    Change the representation of a vector from one basis to another.
    
    Parameters:
    -----------
    vector : numpy.ndarray
        The vector in the old basis coordinates
    old_basis : numpy.ndarray
        The old basis vectors as columns in a matrix
    new_basis : numpy.ndarray
        The new basis vectors as columns in a matrix
    
    Returns:
    --------
    numpy.ndarray
        The vector represented in the new basis coordinates
    """


def change_basis(vector, old_basis, new_basis):

    # Check input dimensions
    if old_basis.shape[0] != new_basis.shape[0]:
        raise ValueError("Basis vectors must have the same dimension")
    
    if old_basis.shape[1] != new_basis.shape[1]:
        raise ValueError("Number of basis vectors must be the same in both bases")
    
    if vector.shape[0] != old_basis.shape[0]:
        raise ValueError("Vector dimension must match basis dimension")
    
    # Step 1: Find the transformation matrix from old_basis to standard basis
    transformation_to_standard = np.linalg.inv(old_basis)
    
    # Step 2: Find the transformation matrix from standard basis to new_basis
    transformation_to_new = new_basis
    
    # Step 3: Combine the transformations
    change_of_basis_matrix = np.matmul(transformation_to_new, transformation_to_standard)
    
    # Step 4: Apply the transformation to the vector
    vector_in_new_basis = np.matmul(np.linalg.inv(change_of_basis_matrix), vector)
    
    return vector_in_new_basis

"""
exact_matrix

Params:
    
    matrix : matrix of coefficients for linear system
    b : inhomogeneous term in linear system
    
Returns:
    eig_vals : array of matrix eigen values
    eig_vecs : NxN array of matrix eigen vectos
    n_qubits : number of qubits required to encode system
    theoretical_solution: the ideal solution to the system in the eigen
                          basis
"""

def exact_matrix(matrix,b):
    #Get eigen values and vectors
    eig_vals, eig_vecs = np.linalg.eigh(matrix)
    #Check if matrix is hermitian
    is_hermitian = np.allclose(matrix, matrix.conj().T)
    print(f"Is matrix Hermitian? {is_hermitian}")
    #Get solution to system (x in Ax = b) and change to eigen basis of matrix
    theoretical_solution = change_basis(np.linalg.solve(matrix, b),np.eye(len(b)),eig_vecs)
    #Normalize theoretical solution
    theoretical_solution = theoretical_solution / np.linalg.norm(theoretical_solution)
    
    #return everything
    return eig_vals,eig_vecs, theoretical_solution    

def inversion_filter(estimated_eig, filter_constant, regularization=0):
    # Avoid division by zero and handle small eigenvalues
    if abs(estimated_eig) < 1e-10:
        return 0
    
    # Apply regularization to avoid extreme values
    denominator = estimated_eig**2 + regularization
    regularized_filter = filter_constant * estimated_eig / denominator
    
    # Ensure the filter value is bounded
    return min(1.0, abs(regularized_filter))

def run_hhl_circuit(matrix, b, time_constant,precision_qubits,use_sine_amplitudes \
                    = True):
    
    #How many qubits to encode matrix?
    n_qubits = int(np.log2(len(matrix)))
    
    #Normalize b
    b = b / np.linalg.norm(b)
    
    #Sine amplitudes for phase estimation qubits
    sine_amplitudes = np.array([np.sin( np.pi * (j + 0.5) / 2**(precision_qubits)) for j in range(2**(precision_qubits))])
    sine_amplitudes = sine_amplitudes / np.linalg.norm(sine_amplitudes)
    #define quantum device
    dev = qml.device('default.qubit', n_qubits + precision_qubits + 1)
    
    #Define the hhl circuit a Qnode
    @qml.qnode(dev)
    def hhl_circuit():
        # Name registers
        pe_wires = list(range(precision_qubits))
        system_wires = list(range(precision_qubits, precision_qubits + n_qubits))
        ancilla_wire = precision_qubits + n_qubits
        
        # Prepare |b⟩ state in system register
        qml.MottonenStatePreparation(b, wires=system_wires)
    
        # Apply H to ancilla
        qml.Hadamard(wires=ancilla_wire)
    
        # Phase estimation part
        if use_sine_amplitudes == True:
            qml.MottonenStatePreparation(sine_amplitudes, wires = pe_wires)
        else:
            for i in pe_wires:
                qml.Hadamard(wires=i)
    
        # Controlled unitaries for phase estimation
        # For Hermitian matrices, exponentiation is well-behaved
        for i, wire in enumerate(pe_wires):
            power = 2**i
            scaled_matrix = matrix * time_constant * power
            evolution_op = scipy.linalg.expm(1j * scaled_matrix)
            qml.ctrl(qml.QubitUnitary, control=wire)(
                evolution_op, wires=system_wires)
    
        # Apply inverse QFT to PE register
        qml.adjoint(qml.QFT)(wires=pe_wires)
        
        # Apply controlled rotations based on eigenvalue estimates
        # We need to loop through all possible values in the PE register
        for i in range(2**precision_qubits):
            # Correct binary conversion
            binary = format(i, f'0{precision_qubits}b')
            
            # Calculate estimated eigenvalue from phase
            phase = i / (2**precision_qubits)
            estimated_eig = phase * (2 * np.pi) / time_constant
            
            # Define filter constant and calculate filter value
            filter_constant = 2 * np.pi / (time_constant * 2**(precision_qubits))
            f_lambda = inversion_filter(estimated_eig, filter_constant)
            
            # Calculate rotation angle based on filtered eigenvalue
            # Clamp the f_lambda value to prevent invalid rotations
            theta = 2 * np.arcsin(min(1.0, max(0.0, f_lambda)))
            
            # Set up control pattern for this specific eigenvalue estimate
            for j, bit in enumerate(binary):
                if bit == '0':
                    qml.PauliX(wires=pe_wires[j])
            
            # Apply controlled rotation
            qml.ctrl(qml.RY, control=pe_wires)(theta, wires=ancilla_wire)
            
            # Restore control qubits
            for j, bit in enumerate(binary):
                if bit == '0':
                    qml.PauliX(wires=pe_wires[j])
        
        # Uncompute phase estimation
        qml.QFT(wires=pe_wires)  # Note: not adjoint here as we're uncomputing
        
        # Apply inverse controlled unitaries with proper time scaling
        for i in range(precision_qubits-1, -1, -1):
            wire = pe_wires[i]
            power = 2**i
            # Use negative time for uncomputation
            scaled_matrix = -matrix * time_constant * power
            evolution_op = scipy.linalg.expm(1j * scaled_matrix)
            qml.ctrl(qml.QubitUnitary, control=wire)(
                evolution_op, wires=system_wires)
        
        # Apply H gates to PE register to complete uncomputation
        for i in pe_wires:
            qml.Hadamard(wires=i)
        
        # Return full state
        return qml.state()
        
        #Run the circuit
    final_state = hhl_circuit()
    #Return the registers as np.arrays
    return final_state

def analyze_hhl_results(final_state, theoretical_solution, precision_qubits, n_qubits):
    """
    Analyze the results of the HHL circuit by computing metrics comparing
    the HHL solution to the theoretical solution.
    
    Parameters:
    -----------
    final_state : numpy.ndarray
        The full quantum state returned by the HHL circuit
    theoretical_solution : numpy.ndarray
        The theoretical solution to the linear system in the eigenbasis
    precision_qubits : int
        Number of qubits used for phase estimation
    n_qubits : int
        Number of qubits used to encode the matrix system
        
    Returns:
    --------
    hhl_solution : numpy.ndarray
        Extracted and normalized HHL solution vector
    success_probability : float
        Probability of measuring the ancilla in |1⟩ state
    fidelity : float
        Fidelity between the normalized HHL solution and theoretical solution
    norm_difference : float
        L2 norm of the difference between normalized HHL solution and theoretical solution
    """
    #Normalize theoretical solution
    theoretical_solution = theoretical_solution /\
        np.linalg.norm(theoretical_solution)

    # Calculate dimensions
    dim_pe = 2**precision_qubits
    dim_system = 2**n_qubits
    
    # Reshape the state for easier analysis
    reshaped_state = final_state.reshape(dim_pe, dim_system, 2)
    
    # Extract the state with ancilla = |1⟩
    ancilla_one_state = reshaped_state[:, :, 1]
    
    # Compute probability of success (measuring ancilla in |1⟩ state)
    success_probability = np.sum(np.abs(ancilla_one_state)**2)
    
    # Extract solution vector (sum over PE register to get system state when ancilla is |1⟩)
    hhl_solution = np.sum(ancilla_one_state, axis=0)
    
    # Normalize the solution
    if np.linalg.norm(hhl_solution) > 0:
        hhl_solution = hhl_solution / np.linalg.norm(hhl_solution)
    
    # Compute fidelity between normalized HHL solution and theoretical solution
    # Fidelity = |⟨ψ_theoretical|ψ_hhl⟩|²
    fidelity = np.abs(np.vdot(theoretical_solution, hhl_solution))**2
    
    # Compute norm difference
    norm_difference = np.linalg.norm(hhl_solution - theoretical_solution)
    
    # Return metrics directly
    return hhl_solution, success_probability, fidelity, norm_difference

def time_sweep(matrix, b, t_space, precision_qubits,use_sine):
    
    """
    Run HHL algorithm for a range of time constants and analyze/plot the results.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        The matrix representing the linear system Ax = b
    b : numpy.ndarray
        The vector representing the right-hand side of the linear system
    t_space : numpy.ndarray
        Array of time constants to test
    precision_qubits : int
        Number of qubits used for phase estimation
    """
    
    
    # Get exact information
    eig_vals, eig_vecs, theoretical_solution = exact_matrix(matrix, b)
    
    # Get n_qubits
    n_qubits = int(np.log2(len(matrix)))
    
    # Arrays to store metrics
    fidelity_array = np.empty(len(t_space))
    success_probability_array = np.empty(len(t_space))
    norm_difference_array = np.empty(len(t_space))
    
    # Run the circuit for each value in t_space
    for k, t in enumerate(t_space):
        # Run circuit for this t
        hhl_state = run_hhl_circuit(matrix, b, t, precision_qubits,use_sine)
        
        # Analyze the solution state
        hhl_solution, success_probability, fidelity, norm_difference = analyze_hhl_results(
            hhl_state, theoretical_solution, precision_qubits, n_qubits)
        
        # Store metrics
        fidelity_array[k] = fidelity
        success_probability_array[k] = success_probability
        norm_difference_array[k] = norm_difference
        
        #print(f"For t = {t:.4f}, fidelity = {fidelity:.4f}, success prob = {success_probability:.4f}")

    
    # Find best time constant based on fidelity (not norm_difference)
    best_fidelity = np.max(fidelity_array)
    best_index = np.argmax(fidelity_array)
    best_time_constant = t_space[best_index]
    best_difference = norm_difference_array[best_index]
    best_probability =success_probability_array[best_index]
    # Use the actual time value, not the index
    
    #print("\nOptimal Results:")
    #print(f"Best Time Constant: {best_time_constant:.4f}")
    #print(f"Best Fidelity: {best_fidelity:.4f}")
    #print(f"Success Probability: {success_probability_array[best_index]:.4f}")
    #print(f"Norm Difference: {norm_difference_array[best_index]:.4f}")
    """
    # 1. Fidelity Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_space, fidelity_array, 'o-', label='Fidelity')
    plt.axvline(x=best_time_constant, color='r', linestyle='--', label=f'Best t = {best_time_constant:.4f}')
    plt.xlabel('Time Constant (t)')
    plt.ylabel('Fidelity')
    plt.title('HHL Algorithm: Fidelity vs Time Constant')
    plt.grid(True)
    plt.legend()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'fidelity_vs_time.png'), dpi=300)
    plt.show()
    
    # 2. Norm Difference Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_space, norm_difference_array, 'o-', color='green', label='Norm Difference')
    plt.axvline(x=best_time_constant, color='r', linestyle='--', label=f'Best t = {best_time_constant:.4f}')
    plt.xlabel('Time Constant (t)')
    plt.ylabel('Norm Difference')
    plt.title('HHL Algorithm: Norm Difference vs Time Constant')
    plt.grid(True)
    plt.legend()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'norm_difference_vs_time.png'), dpi=300)
    plt.show()
    
    # 3. Success Probability Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_space, success_probability_array, 'o-', color='purple', label='Success Probability')
    plt.axvline(x=best_time_constant, color='r', linestyle='--', label=f'Best t = {best_time_constant:.4f}')
    plt.xlabel('Time Constant (t)')
    plt.ylabel('Success Probability')
    plt.title('HHL Algorithm: Success Probability vs Time Constant')
    plt.grid(True)
    plt.legend()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'success_probability_vs_time.png'), dpi=300)
    plt.show()
    
    # 4. Combined Plot
    plt.figure(figsize=(12, 8))
    
    # Check if values need to be scaled for better visualization
    max_fidelity = np.max(fidelity_array)
    max_norm_diff = np.max(norm_difference_array)
    max_success = np.max(success_probability_array)
    
    # If values are all in different ranges, use multiple y-axes
    if max_fidelity > 10*max_success or max_norm_diff > 10*max_success:
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        color1 = 'tab:blue'
        ax1.set_xlabel('Time Constant (t)')
        ax1.set_ylabel('Fidelity', color=color1)
        ax1.plot(t_space, fidelity_array, 'o-', color=color1, label='Fidelity')
        ax1.tick_params(axis='y', labelcolor=color1)
        
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Norm Difference', color=color2)
        ax2.plot(t_space, norm_difference_array, 's-', color=color2, label='Norm Difference')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        color3 = 'tab:purple'
        ax3.set_ylabel('Success Probability', color=color3)
        ax3.plot(t_space, success_probability_array, '^-', color=color3, label='Success Probability')
        ax3.tick_params(axis='y', labelcolor=color3)
        
        # Add vertical line for best t
        ax1.axvline(x=best_time_constant, color='r', linestyle='--', label=f'Best t = {best_time_constant:.4f}')
        
        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')
        
        plt.title('HHL Algorithm: Performance Metrics vs Time Constant')
    else:
        # If values are comparable, use a single y-axis
        plt.plot(t_space, fidelity_array, 'o-', label='Fidelity')
        plt.plot(t_space, norm_difference_array, 's-', color='green', label='Norm Difference')
        plt.plot(t_space, success_probability_array, '^-', color='purple', label='Success Probability')
        plt.axvline(x=best_time_constant, color='r', linestyle='--', label=f'Best t = {best_time_constant:.4f}')
        plt.xlabel('Time Constant (t)')
        plt.ylabel('Metric Value')
        plt.title('HHL Algorithm: Performance Metrics vs Time Constant')
        plt.grid(True)
        plt.legend()
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'combined_metrics_vs_time.png'), dpi=300)
    plt.show()
    
    # 5. Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize the metrics for better visualization
    norm_fidelity = fidelity_array / np.max(fidelity_array) if np.max(fidelity_array) > 0 else fidelity_array
    norm_success_prob = success_probability_array / np.max(success_probability_array) if np.max(success_probability_array) > 0 else success_probability_array
    
    # Create a meshgrid
    x = t_space
    y = np.arange(2)  # 0 for fidelity, 1 for success probability
    X, Y = np.meshgrid(x, y)
    
    # Create Z values (combining both metrics)
    Z = np.zeros((2, len(t_space)))
    Z[0, :] = norm_fidelity
    Z[1, :] = norm_success_prob
    
    # Try to create custom colormap, fall back to viridis if not available
    try:
        colors = [(0, 0, 0.8), (0, 0.8, 0), (0.8, 0.8, 0), (0.8, 0, 0)]  # blue, green, yellow, red
        cmap_name = 'custom_cmap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    except:
        cm = plt.cm.viridis
    
    # Create heatmap
    c = ax.pcolormesh(X, Y, Z, cmap=cm, vmin=0, vmax=1)
    
    # Set y-axis ticks and labels
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Fidelity', 'Success Probability'])
    
    # Add colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Normalized Value')
    
    # Add title and labels
    ax.set_title('HHL Algorithm: Fidelity and Success Probability vs Time Constant')
    ax.set_xlabel('Time Constant (t)')
    
    # Mark the best time constant
    ax.axvline(x=best_time_constant, color='white', linestyle='--', 
               label=f'Best t = {best_time_constant:.4f}')
    ax.legend(loc='upper right')
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'heatmap_metrics_vs_time.png'), dpi=300)
    plt.show()
"""    
    return best_fidelity, best_difference, best_probability,\
        best_time_constant, fidelity_array, success_probability_array,\
        norm_difference_array   
"""
#Define a matrix
matrix = np.array([
    [2, 2+2j, 0, 0],
  [2-2j, 3, -3j, 0],
   [0, 3j, 4,-1j],
    [0, 0, 1j, 5]
], dtype=complex)

#Get eigen vectors in case we need them
eig_vals, eig_vecs = np.linalg.eigh(matrix)
#
#Define inhomogeneous term
k = 0
b = 1 * eig_vecs[:,0] + eig_vecs[:,1] + eig_vecs[:,2] + eig_vecs[:,3]

#Calculate theoretical best value for t
min_gap = min(np.diff(np.sort(eig_vals)))
t_optimal = 2 * np.pi / min_gap

#Run parameter sweep
t_space = np.linspace(t_optimal - t_optimal / 2, t_optimal + t_optimal / 2, 100)  # Using fewer points for faster testing
fidelity_0, success_prob, norm_diff, best_t = time_sweep(matrix, b, t_space, precision_qubits)

print(eig_vals[k])     

fidelity_eig = np.empty(len(eig_vals))
k = 0
for vals in eig_vals:
    b = eig_vecs[:,k]
    fidelity, success_prob, norm_diff, best_t = time_sweep(matrix, b, t_space, precision_qubits)
    fidelity_eig[k] = fidelity
    k+=1
plt.figure(figsize=(10, 6))
plt.plot(eig_vals, fidelity_eig, 'o-', label=r"Fidelity vs Eigenvalue")
plt.axhline(y=fidelity_0, color='r', linestyle='--', label=r"Best Fidelity for Equal Weights on Eigenvectors")
plt.axhline(y=np.sum(fidelity_eig), color='g', linestyle='--', label=r"Theoretical Upper Bound")
plt.axhline(y=np.min(fidelity_eig), color='b', linestyle='--', label=r"Theoretical Lower Bound")
plt.xlabel(r"Eigenvalue $\lambda_i$")
plt.ylabel(r"Fidelity $F(\psi, \psi_{\text{ideal}})$")
plt.title(r"Fidelity-Eigenvalue Relationship")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('fidelity_plot.pdf')  # Save as PDF for best quality
plt.show()
print(eig_vals)
"""


def generate_sparse_invertible_hermitian_matrices(sizes, n):
    """
    Generate n randomly sparse, invertible, Hermitian matrices for each size in the sizes array,
    with evenly spaced condition numbers.
    
    Parameters:
    sizes (list): List of matrix sizes (each matrix will be size x size)
    n (int): Number of matrices to generate for each size
    
    Returns:
    dict: Dictionary with sizes as keys and lists of matrices as values
    """
    matrices = {}
    
    for size in sizes:
        matrices[size] = []
        
        # Define minimum and maximum condition numbers
        min_cond = 1.5
        max_cond = 100.0
        
        # Create evenly spaced condition numbers in log space
        target_condition_numbers = np.logspace(
            np.log10(min_cond), 
            np.log10(max_cond), 
            n
        )
        
        for i in range(n):
            # Create invertible Hermitian matrix with desired condition number
            matrix = create_hermitian_matrix_with_condition_number(size, target_condition_numbers[i])
            
            # Add sparsity (randomly set elements to zero)
            max_sparse_elements = size * size // 2  # Maximum number of zeros
            # Ensure we don't set too many zeros which could make the matrix non-invertible
            num_sparse_elements = np.random.randint(0, max_sparse_elements + 1)
            
            # Keep trying until we get a sparse matrix that is still invertible
            sparse_matrix = add_sparsity_to_hermitian(matrix, num_sparse_elements)
            attempt = 0
            while not is_invertible(sparse_matrix) and attempt < 10:
                sparse_matrix = add_sparsity_to_hermitian(matrix, num_sparse_elements)
                attempt += 1
            
            # If we couldn't create a sparse invertible matrix, fall back to the original
            if not is_invertible(sparse_matrix):
                sparse_matrix = matrix
                
            matrices[size].append(sparse_matrix)
    
    return matrices

def create_hermitian_matrix_with_condition_number(size, condition_number):
    """
    Create a random Hermitian matrix with a specific condition number.
    
    Parameters:
    size (int): Size of the square matrix
    condition_number (float): Desired condition number
    
    Returns:
    numpy.ndarray: Hermitian matrix with the desired condition number
    """
    # Generate random eigenvalues with the desired condition number
    # The largest eigenvalue will be 1, and the smallest will be 1/condition_number
    smallest_eigenvalue = 1.0 / condition_number
    
    # Generate linearly spaced eigenvalues between smallest_eigenvalue and 1
    eigenvalues = np.linspace(smallest_eigenvalue, 1.0, size)
    
    # Shuffle the eigenvalues for more randomness
    np.random.shuffle(eigenvalues)
    
    # Create a diagonal matrix with the eigenvalues
    D = np.diag(eigenvalues)
    
    # Generate a random unitary matrix using QR decomposition
    Q, _ = np.linalg.qr(np.random.randn(size, size) + 1j * np.random.randn(size, size))
    
    # Create the Hermitian matrix with the desired eigenvalues
    A = Q @ D @ Q.conj().T
    
    # Ensure matrix is Hermitian (fix any numerical issues)
    A = (A + A.conj().T) / 2
    
    return A

def add_sparsity_to_hermitian(matrix, num_zeros):
    """
    Add sparsity to a Hermitian matrix by randomly setting elements to zero
    while preserving the Hermitian property.
    
    Parameters:
    matrix (numpy.ndarray): Input Hermitian matrix
    num_zeros (int): Number of elements to set to zero (actual number may be less due to Hermitian constraint)
    
    Returns:
    numpy.ndarray: Sparse Hermitian matrix
    """
    sparse_matrix = matrix.copy()
    size = sparse_matrix.shape[0]
    
    # For Hermitian matrices, we only need to consider the upper triangular part
    # (including diagonal)
    upper_indices = [(i, j) for i in range(size) for j in range(i, size)]
    
    # Adjust number of zeros if it's more than available upper triangle elements
    num_zeros = min(num_zeros, len(upper_indices))
    
    # Randomly choose indices to set to zero
    zero_indices = np.random.choice(len(upper_indices), num_zeros, replace=False)
    
    for idx in zero_indices:
        i, j = upper_indices[idx]
        sparse_matrix[i, j] = 0
        
        # If it's not a diagonal element, set the symmetric element to 0 as well
        if i != j:
            sparse_matrix[j, i] = 0
    
    return sparse_matrix

def is_invertible(matrix):
    """
    Check if a matrix is invertible.
    
    Parameters:
    matrix (numpy.ndarray): Input matrix
    
    Returns:
    bool: True if matrix is invertible, False otherwise
    """
    # For numerical stability, check if the smallest eigenvalue is above a threshold
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.min(np.abs(eigenvalues)) > 1e-10

def get_condition_number(matrix):
    """
    Calculate the condition number of a matrix.
    
    Parameters:
    matrix (numpy.ndarray): Input matrix
    
    Returns:
    float: Condition number of the matrix
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))

def is_hermitian(matrix, tol=1e-10):
    """
    Check if a matrix is Hermitian within a tolerance.
    
    Parameters:
    matrix (numpy.ndarray): Input matrix
    tol (float): Tolerance for numerical errors
    
    Returns:
    bool: True if matrix is Hermitian, False otherwise
    """
    diff = matrix - matrix.conj().T
    return np.max(np.abs(diff)) < tol

# Example usage:
def matrix_generator(num_matrices,sizes_array):
    # Example: Generate 3 matrices for each size in [2, 4, 8]
    
    result = generate_sparse_invertible_hermitian_matrices(sizes_array, num_matrices)
    
    # Display the results
    for size, matrix_list in result.items():
        print(f"Size {size}x{size} matrices:")
        for i, matrix in enumerate(matrix_list):
            print(f"Matrix {i+1}:")
            print(matrix)
            print(f"Condition number: {get_condition_number(matrix):.2f}")
            print(f"Sparsity: {np.count_nonzero(matrix == 0)} zeros")
            print(f"Invertible: {is_invertible(matrix)}")
            print(f"Hermitian: {is_hermitian(matrix)}")
            print()
    return result


def data_collection_and_plotting_function(precision_qubit_range, t_points, matrix_sizes, num_matrices):
    def min_difference_pairwise(arr):
        # Get pairwise differences without sorting
        # (this calculates all possible pairs, not just adjacent elements)
        differences = np.abs(arr[:, None] - arr[None, :])
        # Remove zeros (differences of elements with themselves)
        differences = differences[differences > 0]
        # Return the minimum difference
        return np.min(differences)
    print("Starting data collection and analysis...")
    
    #Get dictionary of random matrices
    matrices_dict = matrix_generator(num_matrices, matrix_sizes)
    
    #Convert the dictionary to a list of matrices (not a numpy array)
    all_matrices = []
    for size, matrices in matrices_dict.items():
        all_matrices.extend(matrices)
    
    # Keep matrices as a list, not a numpy array of objects
    matrices_list = all_matrices
    
    #Create a set of b vectors written in eigen basis - keep as LIST not array
    b_vectors = []
    for size in matrix_sizes:
        for h in range(size):
            b_vector = np.full(size, 0.25)
            b_vector[h] = 0.75
            b_vectors.append(b_vector)
        b_vectors.append(np.ones(size))
    
    print(f"Created {len(b_vectors)} b-vectors for testing")
    
    #Create arrays for data storage
    #Arrays for eigen space analysis
    best_eig_data = np.empty((2, len(precision_qubit_range), len(matrices_list), int(np.max(matrix_sizes)), 4))
    t_sweep_eig_data = np.empty((2, len(precision_qubit_range), len(matrices_list), int(np.max(matrix_sizes)), 3, t_points))
    
    #Arrays for each combined system
    best_b_data = np.empty((2, len(precision_qubit_range), len(matrices_list), len(b_vectors), 4))
    t_sweep_b_data = np.empty((2, len(precision_qubit_range), len(matrices_list), len(b_vectors), 3, t_points))
    
    #Create array of condition numbers
    condition_numbers = np.empty(len(matrices_list))
    
    #Compute the condition numbers - process each matrix individually
    for M, matrix in enumerate(matrices_list):
        try:
            condition_numbers[M] = np.linalg.cond(matrix)
        except np.linalg.LinAlgError:
            condition_numbers[M] = np.nan  # Handle singular matrices
    
    print("Starting HHL simulations...")
    
    #Start Analysis
    #Two different runs for use_sine == True and False
    for sin_idx, pe_sine in enumerate([True, False]):
        print(f"Processing run with sine amplitudes = {pe_sine}")
        
        #Now loop over qubits
        for q, qubits in enumerate(precision_qubit_range):
            print(f"  Processing qubit count = {qubits}")
            
            #Now loop through matrices
            for m, matrix in enumerate(matrices_list):
                print(f"    Processing matrix {m+1}/{len(matrices_list)}")
                
                #For eigen analysis
                eig_vals, eig_vecs = np.linalg.eigh(matrix)
                best_t_M = 1000 * 2 * np.pi / min_difference_pairwise(eig_vals)
                print(f"    Matrix eigenvalues: {eig_vals}")
                
                for e in range(matrix.shape[0]):  # Only process eigenvectors for this matrix
                    vector = eig_vecs[:, e]  # Column e is eigenvector e
                    print(f"      Processing eigenvector {e+1}/{matrix.shape[0]}")
                    best_t_e = 1000 * 2 * np.pi / eig_vals[e] 
                    t_space = np.linspace(best_t_e - 0.10 * best_t_e, best_t_e + 0.10 * best_t_e,t_points)    
                    f_max, diff_opt, prob_opt, t_opt, f_array, p_array, diff_array = time_sweep(
                        matrix, vector, t_space, qubits, use_sine=pe_sine
                    )
                    
                    print(f"      Best t: {t_opt:.4f}, Fidelity: {f_max:.4f}, Prob: {prob_opt:.4f}")
                            
                    #Add data to data arrays
                    best_eig_data[sin_idx, q, m, e, 0] = t_opt
                    best_eig_data[sin_idx, q, m, e, 1] = f_max
                    best_eig_data[sin_idx, q, m, e, 2] = diff_opt
                    best_eig_data[sin_idx, q, m, e, 3] = prob_opt
                    
                    t_sweep_eig_data[sin_idx, q, m, e, 0, :] = f_array
                    t_sweep_eig_data[sin_idx, q, m, e, 1, :] = p_array
                    t_sweep_eig_data[sin_idx, q, m, e, 2, :] = diff_array
                    
                # Fill remaining slots with NaN if any
                for e in range(matrix.shape[0], int(np.max(matrix_sizes))):
                    best_eig_data[sin_idx, q, m, e, :] = np.nan
                    t_sweep_eig_data[sin_idx, q, m, e, :, :] = np.nan
                    
                #Now loop through b_vectors
                for b, vector in enumerate(b_vectors):
                    #Don't do anything for matrix vector pairs of different sizes
                    if matrix.shape[1] != len(vector):
                        # Fill with NaN for incompatible dimensions
                        best_b_data[sin_idx, q, m, b, :] = np.nan
                        t_sweep_b_data[sin_idx, q, m, b, :, :] = np.nan
                        continue
                    else:
                        print(f"      Processing b-vector {b+1} (compatible with matrix)")
                        #Otherwise expand vector in standard basis
                        input_vec = np.zeros(len(vector), dtype=complex)
                        for k in range(len(vector)):
                            input_vec += vector[k] * eig_vecs[:, k]  # Column k is eigenvector k

                        #Get data from time sweep
                        t_space = np.linspace(best_t_M - best_t_M * 0.1, best_t_M * 1.1,t_points)
                        f_max, diff_opt, prob_opt, t_opt, f_array, p_array, diff_array = time_sweep(
                            matrix, input_vec, t_space, qubits, use_sine=pe_sine
                        )
                                
                        #Append data to storage arrays
                        t_sweep_b_data[sin_idx, q, m, b, 0, :] = f_array
                        t_sweep_b_data[sin_idx, q, m, b, 1, :] = diff_array
                        t_sweep_b_data[sin_idx, q, m, b, 2, :] = p_array
                        
                        best_b_data[sin_idx, q, m, b, 0] = t_opt
                        best_b_data[sin_idx, q, m, b, 1] = f_max
                        best_b_data[sin_idx, q, m, b, 2] = diff_opt
                        best_b_data[sin_idx, q, m, b, 3] = prob_opt
                
    print("Data collection complete. Saving data files...")
    
    # Now save the data with proper formats
    # For binary data with complex numbers, use np.save instead of np.savetxt
    np.save("best_eig_data.npy", best_eig_data)
    np.save("t_sweep_eig_data.npy", t_sweep_eig_data)
    np.save("best_b_data.npy", best_b_data)
    np.save("t_sweep_b_data.npy", t_sweep_b_data)
    np.save("condition_numbers.npy", condition_numbers)
    
    # Save matrices and vectors (these are likely complex so use pickle or numpy binary format)
    import pickle
    with open('matrices.pkl', 'wb') as f:
        pickle.dump(matrices_list, f)
    
    # Save b_vectors as a pickle file since they have inconsistent sizes
    with open('b_vectors.pkl', 'wb') as f:
        pickle.dump(b_vectors, f)
    
    # Add some details about what was saved
    with open('data_collection_info.txt', 'w') as f:
        f.write("Data collection information:\n")
        f.write(f"Precision qubit range: {precision_qubit_range}\n")
        f.write(f"Matrix sizes: {matrix_sizes}\n")
        f.write(f"Number of matrices: {num_matrices}\n")
        f.write(f"Time space range: {t_space[0]} to {t_space[-1]} with {len(t_space)} points\n")
        f.write(f"Number of b_vectors: {len(b_vectors)}\n")
        
    print("Data saved successfully. Beginning visualization...")
    
    # Generate plots
    generate_all_plots(matrices_list, best_eig_data, best_b_data, t_sweep_eig_data, 
                       t_sweep_b_data, precision_qubit_range, t_space, b_vectors)
    
    print("All analysis and visualization complete.")
    
    # Return important data structures in case needed for further analysis
    return best_eig_data, best_b_data, t_sweep_eig_data, t_sweep_b_data, matrices_list, b_vectors  
    
t_1 = time.time()   
data_collection_and_plotting_function([15], 100, [4], 1)
t_2 = time.time()
print('This took t = ',t_2-t_1,' seconds')
    
"""
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def plot_fidelity_and_norm(matrix, b, pe_qubits, t_space):

    #Calculate and plot fidelity and norm difference versus time

    # Disable LaTeX to avoid Unicode issues
    plt.rcParams['text.usetex'] = False
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif'],
    })
    
    # Calculate eigenvalues, eigenvectors, and exact solution
    eig_vals, eig_vec, exact_solution = exact_matrix(matrix, b)
    
    # Change basis for b
    b_in_eig = change_basis(b, np.eye(len(b)), eig_vec.T)
    b_in_eig = b_in_eig / np.linalg.norm(b_in_eig)
    
    # Compute the T value
    T = 2**pe_qubits
    
    def alpha(k, j, t_0):
        delta = eig_vals[j] * t_0 - 2 * np.pi * k
        # Handle potential numerical instability
        if abs(delta) < 1e-10:
            return 1.0
        
        # Handle potential division by zero
        try:
            alpha = np.exp(1j * delta * (1-1/T)) * np.sqrt(2) * (np.cos(delta/2) / T) * ((2 * np.cos(delta/(2*T)) * np.sin(np.pi/(2*T)))
                                                            / (np.sin((delta+np.pi)/(2*T)) * np.sin((delta-np.pi)/(2*T))))
            # Check if result is valid
            if np.isnan(alpha) or np.isinf(alpha):
                return 0.0
            return alpha
        except:
            return 0.0
    
    def get_coefficients(t_0):
        coefficients = np.zeros(len(b), dtype=complex)
        for j in range(len(b)):
            coeff_sum = 0
            for k in range(T):
                alpha_val = alpha(k, j, t_0)
                for t in range(T):
                    estimated_eig = 2 * np.pi * k / t_0
                    filter_val = inversion_filter(estimated_eig, 1e-4)
                    exp_term = np.exp((1j * t / T) * (2 * np.pi * k - eig_vals[j] * t_0))
                    coeff_sum += filter_val * alpha_val * b_in_eig[j] * exp_term
            coefficients[j] = coeff_sum
        coefficients = coefficients / np.linalg.norm(coefficients)
        fidelity = abs(np.dot(coefficients, exact_solution))
        norm_diff = np.linalg.norm(coefficients - exact_solution)
        return coefficients, fidelity, norm_diff
    
    # Initialize arrays to store fidelity and norm difference
    fidelities = np.zeros(len(t_space))
    norm_diffs = np.zeros(len(t_space))
    
    # Calculate values for each time point
    print("Calculating fidelity and norm difference...")
    for i, t in enumerate(t_space):
        # Show progress periodically
        if i % 100 == 0:
            print(f"Progress: {i}/{len(t_space)} points")
        
        _, fidelity, norm_diff = get_coefficients(t)
        fidelities[i] = fidelity
        norm_diffs[i] = norm_diff
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot fidelity
    ax1.plot(t_space, fidelities, 'b-', linewidth=1.5)
    ax1.set_ylabel('Fidelity')
    ax1.set_title('Fidelity vs Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Find peaks in fidelity
    peaks, _ = find_peaks(fidelities, height=np.mean(fidelities), distance=10)
    if len(peaks) > 0:
        ax1.plot(t_space[peaks], fidelities[peaks], 'ro', markersize=6)
        
        # Calculate average period
        if len(peaks) >= 2:
            periods = np.diff(t_space[peaks])
            avg_period = np.mean(periods)
            # Add text about period
            ax1.text(0.02, 0.95, f"Avg. Period: {avg_period:.4f}", transform=ax1.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot norm difference
    ax2.plot(t_space, norm_diffs, 'g-', linewidth=1.5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Norm Difference')
    ax2.set_title('Norm Difference vs Time')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Find minima in norm difference
    # Invert the data to find minima as peaks
    inv_norm_diffs = -norm_diffs
    minima, _ = find_peaks(inv_norm_diffs, height=-np.mean(norm_diffs), distance=10)
    if len(minima) > 0:
        ax2.plot(t_space[minima], norm_diffs[minima], 'ro', markersize=6)
        
        # Calculate average period
        if len(minima) >= 2:
            periods = np.diff(t_space[minima])
            avg_period = np.mean(periods)
            # Add text about period
            ax2.text(0.02, 0.95, f"Avg. Period: {avg_period:.4f}", transform=ax2.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Maximum Fidelity: {np.max(fidelities):.4f} at t = {t_space[np.argmax(fidelities)]:.4f}")
    print(f"Minimum Norm Difference: {np.min(norm_diffs):.4f} at t = {t_space[np.argmin(norm_diffs)]:.4f}")
    
    # Find optimal time points
    best_fidelity_idx = np.argmax(fidelities)
    best_norm_idx = np.argmin(norm_diffs)
    
    print("\nOptimal Time Points:")
    print(f"Best Fidelity: t = {t_space[best_fidelity_idx]:.4f}")
    print(f"Best Norm: t = {t_space[best_norm_idx]:.4f}")
    
    # Check if these times are close to theoretical periods
    theoretical_periods = [2*np.pi/abs(val) for val in eig_vals]
    print("\nTheoretical Periods from Eigenvalues:")
    for i, period in enumerate(theoretical_periods):
        print(f"Eigenvalue {i} (λ = {eig_vals[i]}): Period = {period:.4f}")
    
    return fidelities, norm_diffs, t_space

# Define the matrix and vector
matrix = np.array([[1.543, 9.7584-4.843j], [9.7584+4.843j, 8.4738]])
b = np.array([4.39843, 7.8948])

# Check if matrix is Hermitian
is_hermitian = np.allclose(matrix, matrix.conj().T)
print(f"Is matrix Hermitian? {is_hermitian}")

# Set time range (adjust as needed to see the patterns clearly)
t_space = np.linspace(0.1, 3, 500)

# Run with reduced pe_qubits for faster execution
fidelities, norm_diffs, times = plot_fidelity_and_norm(matrix, b, 5, t_space)
"""
