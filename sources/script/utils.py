import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def gsigmoid(V, A, B, C, D):
    """
    Generalized sigmoid function.

    Computes a sigmoid function with adjustable parameters.

    Parameters
    ----------
    V : float or np.ndarray
        The input value(s) for the function.
    A : float
        The vertical shift (baseline value).
    B : float
        The amplitude of the sigmoid.
    C : float
        The steepness of the curve (scaling factor).
    D : float
        The horizontal shift of the sigmoid.

    Returns
    -------
    float or np.ndarray
        The result of the sigmoid function applied to `V`.

    Notes
    -----
    The generalized sigmoid function is given by:
        gsigmoid(V, A, B, C, D) = A + B / (1 + exp((V + D) / C))
    """
    return A + B / (1 + np.exp((V + D) / C))


def d_gsigmoid(V, A, B, C, D):
    """
    Derivative of the generalized sigmoid function.

    Computes the derivative of the sigmoid function with respect to `V`.

    Parameters
    ----------
    V : float or np.ndarray
        The input value(s) for the function.
    A : float
        The vertical shift (baseline value) of the sigmoid.
    B : float
        The amplitude of the sigmoid.
    C : float
        The steepness of the curve (scaling factor).
    D : float
        The horizontal shift of the sigmoid.

    Returns
    -------
    float or np.ndarray
        The derivative of the sigmoid function at `V`.

    Notes
    -----
    The derivative is given by:
        d_gsigmoid(V, A, B, C, D) = -B * exp((V + D) / C) /
                                    (C * (1 + exp((V + D) / C))^2)
    """
    return -B * np.exp((V + D) / C) / (C * (1 + np.exp((V + D) / C)) ** 2)


def gamma_uniform_mean_std_matching(uniform_a, uniform_b):
    """
    Match a gamma distribution's parameters to a uniform distribution.

    Solves for the shape (k) and scale (theta) parameters of a gamma
    distribution that matches the mean and variance of a uniform distribution
    over the interval [a, b].

    Parameters
    ----------
    uniform_a : float
        The lower bound of the uniform distribution.
    uniform_b : float
        The upper bound of the uniform distribution.

    Returns
    -------
    k : float
        The shape parameter of the gamma distribution.
    theta : float
        The scale parameter of the gamma distribution.

    Notes
    -----
    The uniform distribution has:
        - Mean: (a + b) / 2
        - Variance: (b - a)^2 / 12

    The gamma distribution is parameterized as:
        p(x) = x^(k-1) * exp(-x/theta) / (theta^k * Gamma(k)),
    where the mean is k * theta and the variance is k * theta^2.

    The parameters are solved as:
        k = 3 * (a + b)^2 / (b - a)^2
        theta = (b - a)^2 / (6 * (a + b))
    """
    a = uniform_a
    b = uniform_b
    
    p = a + b
    q_sq = (b - a) ** 2
    k = 3 * p ** 2 / q_sq
    theta = q_sq / (6 * p)
    return k, theta


# == simulation utils functions ==

def simulate_population_multiprocessing(simulation_function, population, u0, T_final, dt, params, max_workers=8, verbose=False):
    """
    Simulate a population using multiprocessing over a fixed time duration.

    Parameters
    ----------
    simulation_function : callable
        The function to simulate a single individual's dynamics.
    population : list
        The population to simulate, represented as a list of individuals.
    u0 : array-like
        The initial state of the simulation.
    T_final : float
        The final time of the simulation.
    dt : float
        The time step for the simulation.
    params : dict
        Additional parameters required by the simulation function.
    max_workers : int, optional
        The maximum number of worker processes for multiprocessing. Defaults to 8.
    verbose : bool, optional
        If True, display progress using `tqdm`. Defaults to False.

    Returns
    -------
    list
        A list of simulation results, one for each individual in the population.

    Notes
    -----
    This function divides the population among multiple processes and evaluates
    the simulation function for each individual in parallel.
    """
    return simulate_population_t_eval_multiprocessing(simulation_function, population, u0, np.arange(0, T_final, dt), params, max_workers, verbose)



def simulate_population_t_eval_multiprocessing(simulation_function, population, u0, t_eval, params, max_workers=8, verbose=False, use_tqdm=True):
    """
    Simulate a population using multiprocessing over specified evaluation times.

    Parameters
    ----------
    simulation_function : callable
        The function to simulate a single individual's dynamics.
    population : list
        The population to simulate, represented as a list of individuals.
    u0 : array-like
        The initial state of the simulation.
    t_eval : array-like
        An array of time points at which to evaluate the simulation.
    params : dict
        Additional parameters required by the simulation function.
    max_workers : int, optional
        The maximum number of worker processes for multiprocessing. Defaults to 8.
    verbose : bool, optional
        If True, display progress. Defaults to False.
    use_tqdm : bool, optional
        If True and verbose is True, use tqdm for progress bar. Otherwise, use print.

    Returns
    -------
    list
        A list of simulation results, one for each individual in the population.
    """
    traces = []
    total = len(population)
    tasks = [(u0, individual, t_eval, params) for individual in population]

    if verbose and use_tqdm:
        # Use tqdm progress bar
        raise ValueError("tqdm has been removed from the code.")
    else:
        # Use print-based progress reporting
        results = [None] * total
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(simulation_function, task): i
                for i, task in enumerate(tasks)
            }

            completed = 0
            next_print = 10  # percentage to print at
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1

                if verbose:
                    progress = (completed / total) * 100
                    if progress >= next_print:
                        print(f"PROGRESS: {completed}/{total}: {progress:.1f}%", flush=True)
                        next_print += 10

    traces.extend(results)
    return traces


def simulate_population(simulation_function, population, u0, T_final, dt, params, verbose=False):
    """
    Simulate a population sequentially over a fixed time duration.

    Parameters
    ----------
    simulation_function : callable
        The function to simulate a single individual's dynamics.
    population : list
        The population to simulate, represented as a list of individuals.
    u0 : array-like
        The initial state of the simulation.
    T_final : float
        The final time of the simulation.
    dt : float
        The time step for the simulation.
    params : dict
        Additional parameters required by the simulation function.
    verbose : bool, optional
        If True, display progress using `tqdm`. Defaults to False.

    Returns
    -------
    list
        A list of simulation results, one for each individual in the population.

    Notes
    -----
    This function simulates each individual in the population sequentially.
    """
    return simulate_population_t_eval(simulation_function, population, u0, np.arange(0, T_final, dt), params, verbose)


def simulate_population_t_eval(simulation, population, u0, t_eval, params, verbose=False):
    """
    Simulate a population sequentially over specified evaluation times.

    Parameters
    ----------
    simulation : callable
        The function to simulate a single individual's dynamics.
    population : list
        The population to simulate, represented as a list of individuals.
    u0 : array-like
        The initial state of the simulation.
    t_eval : array-like
        An array of time points at which to evaluate the simulation.
    params : dict
        Additional parameters required by the simulation function.
    verbose : bool, optional
        If True, display progress using `tqdm`. Defaults to False.

    Returns
    -------
    list
        A list of simulation results, one for each individual in the population.

    Notes
    -----
    This function iterates over the population and evaluates the simulation
    function for each individual sequentially.
    """
    traces = []
    for i in range(len(population)):
        if verbose:
            print(f"Simulating individual {i + 1}/{len(population)}", flush=True)
        
        individual = population[i]
        trace = simulation([u0, individual, t_eval, params])
        traces.append(trace)
    return traces


# == DICs computation related utils functions ==

def w_factor(V, tau_x, tau_1, tau_2, default=1):
    """
    Compute the weighting factor based on dynamic tau values.

    Parameters
    ----------
    V : array-like
        The input variable (e.g., voltage or some other parameter).
    tau_x : callable
        A function that computes tau_x(V).
    tau_1 : callable
        A function that computes tau_1(V).
    tau_2 : callable
        A function that computes tau_2(V).
    default : float, optional
        The default value to assign when no conditions are met. Defaults to 1.

    Returns
    -------
    numpy.ndarray
        An array of weighting factors corresponding to each value in `V`.

    Notes
    -----
    - The weighting factor is calculated based on logarithmic differences
      between tau_x(V), tau_1(V), and tau_2(V).
    - If tau_x(V) > tau_2(V), the weighting factor is set to 0.
    - If tau_x(V) is between tau_1(V) and tau_2(V), the weighting factor is
      calculated proportionally.
    """
    V = np.asarray(V)
    result = np.ones_like(V) * default
    mask_1 = (tau_x(V) > tau_1(V)) & (tau_x(V) <= tau_2(V))
    mask_2 = tau_x(V) > tau_2(V)
    result[mask_1] = (np.log(tau_2(V[mask_1])) - np.log(tau_x(V[mask_1]))) / (np.log(tau_2(V[mask_1])) - np.log(tau_1(V[mask_1])))
    result[mask_2] = 0
    return result


def w_factor_constant_tau(V, tau_x, tau_1, tau_2, default=1):
    """
    Compute the weighting factor based on constant tau_x and dynamic tau_1 and tau_2.

    Parameters
    ----------
    V : array-like
        The input variable (e.g., voltage or some other parameter).
    tau_x : float
        A constant value for tau_x.
    tau_1 : callable
        A function that computes tau_1(V).
    tau_2 : callable
        A function that computes tau_2(V).
    default : float, optional
        The default value to assign when no conditions are met. Defaults to 1.

    Returns
    -------
    numpy.ndarray
        An array of weighting factors corresponding to each value in `V`.

    Notes
    -----
    - The weighting factor is calculated based on logarithmic differences
      between tau_x, tau_1(V), and tau_2(V).
    - If tau_x > tau_2(V), the weighting factor is set to 0.
    - If tau_x is between tau_1(V) and tau_2(V), the weighting factor is
      calculated proportionally.
    """
    V = np.asarray(V)
    result = np.ones_like(V) * default
    mask_1 = (tau_x > tau_1(V)) & (tau_x <= tau_2(V))
    mask_2 = tau_x > tau_2(V)
    result[mask_1] = (np.log(tau_2(V[mask_1])) - np.log(tau_x)) / (np.log(tau_2(V[mask_1])) - np.log(tau_1(V[mask_1])))
    result[mask_2] = 0
    return result


def get_w_factors(V, tau_x, tau_f, tau_s, tau_u):
    """
    Compute two weighting factors using dynamic tau values.

    Parameters
    ----------
    V : array-like
        The input variable (e.g., voltage or some other parameter).
    tau_x : callable
        A function that computes tau_x(V).
    tau_f : callable
        A function that computes tau_f(V).
    tau_s : callable
        A function that computes tau_s(V).
    tau_u : callable
        A function that computes tau_u(V).

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two arrays:
        - The first array is the weighting factor for (tau_f, tau_s).
        - The second array is the weighting factor for (tau_s, tau_u).
    """
    return w_factor(V, tau_x, tau_f, tau_s, default=1), w_factor(V, tau_x, tau_s, tau_u, default=1)


def get_w_factors_constant_tau(V, tau_x, tau_f, tau_s, tau_u):
    """
    Compute two weighting factors using a constant tau_x and dynamic tau values.

    Parameters
    ----------
    V : array-like
        The input variable (e.g., voltage or some other parameter).
    tau_x : float
        A constant value for tau_x.
    tau_f : callable
        A function that computes tau_f(V).
    tau_s : callable
        A function that computes tau_s(V).
    tau_u : callable
        A function that computes tau_u(V).

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two arrays:
        - The first array is the weighting factor for (tau_f, tau_s).
        - The second array is the weighting factor for (tau_s, tau_u).
    """
    return w_factor_constant_tau(V, tau_x, tau_f, tau_s), w_factor_constant_tau(V, tau_x, tau_s, tau_u)

def get_spiking_times(t, V, spike_high_threshold=10, spike_low_threshold=0):
    """
    Extract the spiking times from a voltage trace based on threshold crossings.

    Parameters
    ----------
    t : array-like
        Time points corresponding to the voltage trace `V`.
    V : array-like
        Voltage trace to analyze for spiking events.
    spike_high_threshold : float, optional
        The voltage threshold above which a spike is considered to start. Defaults to 10.
    spike_low_threshold : float, optional
        The voltage threshold below which a spike is considered to end. Defaults to 0.

    Returns
    -------
    tuple of (array-like, array-like)
        - `valid_starts` : The indices of `t` where spikes start (crossing above `spike_high_threshold`).
        - `spike_times` : The corresponding time points in `t` for the spike starts.

    Notes
    -----
    - This function assumes that the voltage trace is continuous and does NOT handle batch processing.
    - A spike is defined as a region where the voltage exceeds `spike_high_threshold` and eventually falls below `spike_low_threshold`.
    - Only spike starts that have a corresponding spike end are considered valid.
    - If no spikes are detected, the function returns two empty arrays.
    """
    above_threshold = V > spike_high_threshold
    below_threshold = V < spike_low_threshold

    spike_starts = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
    spike_ends = np.where(np.diff(below_threshold.astype(int)) == 1)[0] + 1
    
    # Only consider starts that have a corresponding end after them
    if len(spike_starts) == 0 or len(spike_ends) == 0:
        return np.array([]), np.array([])
    
    valid_starts = spike_starts[spike_starts < spike_ends[-1]]

    spike_times = t[valid_starts]
    
    return valid_starts, spike_times