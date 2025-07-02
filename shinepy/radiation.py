from typing import Literal

import numpy as np
import scipy.interpolate as ip
import srwlib as sp  # a Python module, sp for 'srw' and 'python package'
from numba import njit, prange


def compute(
    *,
    beam: sp.SRWLPartBeam,
    b_field: sp.SRWLMagFldC,
    mesh: sp.SRWLRadMesh,
    comp_method: Literal['manual', 'auto-undulator', 'auto-wiggler'],
    comp_rel_prec: float,
    n_ptl: int,
    rad_charac: Literal['intensity'],
    output_file: str,
    rand_method: Literal['pseudo-random', 'halton'],
    beam_convo: bool,
):

    if comp_method == 'manual':
        sr_meth = 0
    elif comp_method == 'auto-undulator':
        sr_meth = 1
    elif comp_method == 'auto-wiggler':
        sr_meth = 2
    else:
        raise RuntimeError(
            "Invalid computation method provided."
            "Choices are: 'manual', 'auto-undulator', 'auto-wiggler')"
        )

    if rad_charac == 'intensity':
        _char = 0
    else:
        raise RuntimeError("Invalid radiation characteristic provided. Choices are: 'intensity'")

    if rand_method == 'pseudo-random':
        _rand_meth = 1
    elif rand_method == 'halton':
        _rand_meth = 2
    else:
        raise RuntimeError(
            "Invalid random number generator provided. Choices are: pseudo-random, halton"
        )

    if beam_convo:
        me_approx = 1
    else:
        me_approx = 0

    sp.srwl_wfr_emit_prop_multi_e(
        _e_beam=beam,
        _mag=b_field,
        _mesh=mesh,
        _sr_meth=sr_meth,
        _sr_rel_prec=comp_rel_prec,
        _n_part_tot=n_ptl,
        _char=_char,
        _file_path=output_file,
        _rand_meth=_rand_meth,
        _me_approx=me_approx,
    )


def convolve_angular_divergence(
    I_2d,
    x_grid,
    y_grid,
    sigma_xp,
    sigma_yp,
    distance_to_center,
    undulator_length,
    n_samples=10000,
    seed=None,
):
    """Convolve 2D intensity I(x, y) with angular divergence of the electron beam.

    This version accounts for radiation being emitted continuously along the undulator
    rather than just at its center. Each angular offset shifts the intensity pattern
    depending on the emission point.

    Args:
        I_2d (ndarray): 2D intensity array with shape (nx, ny), representing |E|^2.
        x_grid (ndarray): 1D array of horizontal spatial coordinates [m].
        y_grid (ndarray): 1D array of vertical spatial coordinates [m].
        sigma_xp (float): RMS angular divergence in the horizontal direction [rad].
        sigma_yp (float): RMS angular divergence in the vertical direction [rad].
        distance_to_center (float): Distance from undulator center (z=0) to detector [m].
        undulator_length (float): Total length of the undulator [m].
        n_samples (int, optional): Number of Monte Carlo samples. Defaults to 10000.
        seed (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
        ndarray: Convolved intensity array of shape (nx, ny).
    """
    if seed is not None:
        np.random.seed(seed)

    print(f"Starting angular divergence convolution with {n_samples} samples...")
    print(f"Detector distance from center: {distance_to_center:.2f} m")
    print(f"Undulator length: {undulator_length:.2f} m")
    print(f"Angular divergences: σx' = {sigma_xp*1e6:.1f} μrad, σy' = {sigma_yp*1e6:.1f} μrad")

    # Create interpolator for 2D intensity
    interpolator = ip.RegularGridInterpolator(
        (x_grid, y_grid), I_2d, method='linear', bounds_error=False, fill_value=0.0
    )

    I_convolved = np.zeros_like(I_2d)

    # Generate random emission positions along undulator (centered at z=0)
    z_emit = np.random.uniform(
        low=-0.5 * undulator_length, high=+0.5 * undulator_length, size=n_samples
    )

    # Distance from emission point to detector
    distances = distance_to_center - z_emit

    # Angular offsets
    xp_offsets = np.random.normal(0, sigma_xp, n_samples)
    yp_offsets = np.random.normal(0, sigma_yp, n_samples)

    # Spatial shifts at detector from emission point
    x_shifts = distances * xp_offsets
    y_shifts = distances * yp_offsets

    print(
        f"Max spatial shifts: x = ±{np.max(np.abs(x_shifts))*1e6:.1f} μm, "
        f"y = ±{np.max(np.abs(y_shifts))*1e6:.1f} μm"
    )

    progress_points = [int(n_samples * p) for p in [0.1, 0.25, 0.5, 0.75, 0.9]]
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

    for i in range(n_samples):
        if i in progress_points:
            print(f"Progress: {100*i/n_samples:.0f}%")

        dx, dy = x_shifts[i], y_shifts[i]
        X_shifted = X - dx
        Y_shifted = Y - dy

        coords = np.stack([X_shifted.ravel(), Y_shifted.ravel()], axis=1)
        I_shifted = interpolator(coords).reshape(I_2d.shape)
        I_convolved += I_shifted

    I_convolved /= n_samples
    print("Angular divergence convolution completed!")
    return I_convolved


def convolve_beam_size(I_2d, x_grid, y_grid, sigma_x, sigma_y, n_samples=10000, seed=None):
    """Convolve 2D intensity I(x, y) with finite electron beam size.

    The electron beam has a spatial spread at the source, which shifts the radiation
    pattern. This function applies a Monte Carlo convolution to simulate the effect.

    Args:
        I_2d (ndarray): 2D intensity array with shape (nx, ny), representing |E|^2.
        x_grid (ndarray): 1D array of horizontal spatial coordinates [m].
        y_grid (ndarray): 1D array of vertical spatial coordinates [m].
        sigma_x (float): RMS beam size in the horizontal direction [m].
        sigma_y (float): RMS beam size in the vertical direction [m].
        n_samples (int, optional): Number of Monte Carlo samples. Defaults to 10000.
        seed (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
        ndarray: Convolved intensity array of shape (nx, ny).
    """

    if seed is not None:
        np.random.seed(seed)

    print(f"Starting beam size convolution with {n_samples} samples...")
    print(f"Beam sizes: σx = {sigma_x*1e6:.1f} μm, σy = {sigma_y*1e6:.1f} μm")
    print(I_2d.shape, x_grid.shape, y_grid.shape)

    # Create interpolator for 2D intensity
    interpolator = ip.RegularGridInterpolator(
        (x_grid, y_grid), I_2d, method='linear', bounds_error=False, fill_value=0.0
    )

    # Initialize convolved intensity array
    I_convolved = np.zeros_like(I_2d)

    # Generate random electron positions
    x_offsets = np.random.normal(0, sigma_x, n_samples)
    y_offsets = np.random.normal(0, sigma_y, n_samples)

    # Progress tracking
    progress_points = [int(n_samples * p) for p in [0.1, 0.25, 0.5, 0.75, 0.9]]

    # Create coordinate meshgrid once
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

    for i in range(n_samples):
        if i in progress_points:
            print(f"Progress: {100*i/n_samples:.0f}%")

        # Current electron position offset
        dx, dy = x_offsets[i], y_offsets[i]

        # Shifted coordinates
        X_shifted = X - dx
        Y_shifted = Y - dy

        # Stack coordinates for interpolation
        coords = np.stack([X_shifted.ravel(), Y_shifted.ravel()], axis=1)

        # Interpolate intensity at shifted coordinates
        I_shifted = interpolator(coords).reshape(I_2d.shape)

        # Add to convolved intensity
        I_convolved += I_shifted

    # Average over all samples
    I_convolved /= n_samples

    print("Beam size convolution completed!")
    return I_convolved


# def convolve_beam_size_and_divergence(
#     I_2d,
#     x_grid,
#     y_grid,
#     sigma_x,
#     sigma_y,
#     sigma_xp,
#     sigma_yp,
#     distance_to_center,
#     undulator_length,
#     n_samples=20_000,
#     seed=None,
# ):
#     """Convolve 2D intensity I(x, y) with both beam size and angular divergence.

#     Simulates the combined effect of transverse source size and angular spread
#     using a single Monte Carlo loop.

#     Args:
#         I_2d (ndarray): 2D intensity array with shape (nx, ny), representing |E|^2.
#         x_grid (ndarray): 1D array of horizontal spatial coordinates [m].
#         y_grid (ndarray): 1D array of vertical spatial coordinates [m].
#         sigma_x (float): RMS beam size in horizontal direction [m].
#         sigma_y (float): RMS beam size in vertical direction [m].
#         sigma_xp (float): RMS angular divergence in horizontal direction [rad].
#         sigma_yp (float): RMS angular divergence in vertical direction [rad].
#         distance_to_center (float): Distance from undulator center (z=0) to detector [m].
#         undulator_length (float): Total length of the undulator [m].
#         n_samples (int, optional): Number of Monte Carlo samples. Defaults to 10000.
#         seed (int, optional): Seed for reproducibility.

#     Returns:
#         ndarray: Convolved intensity array of shape (nx, ny).
#     """
#     if seed is not None:
#         np.random.seed(seed)

#     print(f"Starting combined convolution with {n_samples} samples...")
#     print(f"Detector distance from center: {distance_to_center:.2f} m")
#     print(f"Undulator length: {undulator_length:.2f} m")
#     print(f"Beam size: σx = {sigma_x*1e6:.1f} μm, σy = {sigma_y*1e6:.1f} μm")
#     print(f"Angular divergence: σx' = {sigma_xp*1e6:.1f} μrad, σy' = {sigma_yp*1e6:.1f} μrad")

#     interpolator = ip.RegularGridInterpolator(
#         (x_grid, y_grid), I_2d, method='cubic', bounds_error=False, fill_value=0.0
#     )

#     I_convolved = np.zeros_like(I_2d)

#     # Sample emission position (z), beam position (x/y), and angles (x'/y')
#     z_emit = np.random.uniform(
#         low=-0.5 * undulator_length, high=+0.5 * undulator_length, size=n_samples
#     )
#     distances = distance_to_center - z_emit

#     x0 = np.random.normal(0, sigma_x, n_samples)
#     y0 = np.random.normal(0, sigma_y, n_samples)
#     xp = np.random.normal(0, sigma_xp, n_samples)
#     yp = np.random.normal(0, sigma_yp, n_samples)

#     # Total shift: initial position + angle × distance
#     x_shifts = x0 + distances * xp
#     y_shifts = y0 + distances * yp

#     print(
#         f"Max total shifts: x = ±{np.max(np.abs(x_shifts))*1e6:.1f} μm, "
#         f"y = ±{np.max(np.abs(y_shifts))*1e6:.1f} μm"
#     )

#     X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
#     progress_points = [int(n_samples * p) for p in [0.1, 0.25, 0.5, 0.75, 0.9]]

#     for i in range(n_samples):
#         if i in progress_points:
#             print(f"Progress: {100*i/n_samples:.0f}%")

#         X_shifted = X - x_shifts[i]
#         Y_shifted = Y - y_shifts[i]
#         coords = np.stack([X_shifted.ravel(), Y_shifted.ravel()], axis=1)

#         I_shifted = interpolator(coords).reshape(I_2d.shape)
#         I_convolved += I_shifted

#     I_convolved /= n_samples
#     print("Combined convolution completed!")
#     return I_convolved


def convolve_beam_size_and_divergence(
    I_2d,
    x_grid,
    y_grid,
    sigma_x,
    sigma_y,
    sigma_xp,
    sigma_yp,
    distance_to_center,
    undulator_length,
    n_samples=20_000,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    x_min, x_max = x_grid[0], x_grid[-1]
    y_min, y_max = y_grid[0], y_grid[-1]
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    nx, ny = len(x_grid), len(y_grid)
    I_convolved = np.zeros((nx, ny), dtype=np.float64)

    # Sample Monte Carlo variables
    z_emit = np.random.uniform(-0.5 * undulator_length, 0.5 * undulator_length, size=n_samples)
    distances = distance_to_center - z_emit
    x0 = np.random.normal(0, sigma_x, size=n_samples)
    y0 = np.random.normal(0, sigma_y, size=n_samples)
    xp = np.random.normal(0, sigma_xp, size=n_samples)
    yp = np.random.normal(0, sigma_yp, size=n_samples)

    x_shifts = x0 + distances * xp
    y_shifts = y0 + distances * yp

    I_convolved = _convolve_loop(
        I_2d, x_grid, y_grid, x_shifts, y_shifts, dx, dy, x_min, y_min, nx, ny
    )

    I_convolved /= n_samples

    return I_convolved


@njit(parallel=True)
def _convolve_loop(I_2d, x_grid, y_grid, x_shifts, y_shifts, dx, dy, x_min, y_min, nx, ny):
    I_accum = np.zeros((nx, ny))
    for i in prange(len(x_shifts)):
        x_shift = x_shifts[i]
        y_shift = y_shifts[i]

        for ix in range(nx):
            for iy in range(ny):
                x = x_grid[ix] - x_shift
                y = y_grid[iy] - y_shift

                # Get fractional index in grid
                fx = (x - x_min) / dx
                fy = (y - y_min) / dy

                ix0 = int(np.floor(fx))
                iy0 = int(np.floor(fy))

                if ix0 < 0 or iy0 < 0 or ix0 >= nx - 1 or iy0 >= ny - 1:
                    continue  # outside bounds

                dx1 = fx - ix0
                dy1 = fy - iy0

                # Bilinear interpolation
                I_interp = (
                    (1 - dx1) * (1 - dy1) * I_2d[ix0, iy0]
                    + dx1 * (1 - dy1) * I_2d[ix0 + 1, iy0]
                    + (1 - dx1) * dy1 * I_2d[ix0, iy0 + 1]
                    + dx1 * dy1 * I_2d[ix0 + 1, iy0 + 1]
                )

                I_accum[ix, iy] += I_interp

    return I_accum
