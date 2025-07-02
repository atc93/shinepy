"""Compute synchrotron radiation of an electron beam inside a planar undulator using SRW library"""

# --- Imports ---

# Standard library imports
import sys
import tomllib

# Third-party imports
import cbpm_analysis_core.visualize.helper as hp
import matplotlib.pyplot as plt
import numpy as np

# Local module imports
import beam
import helper
import mesh
import radiation
import undulator
import wavefront
from config import *


def main():

    # Load configuration from TOML config file
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    # Compute variables using configuration file
    gamma = config['beam']['ptl_energy'] / config['beam']['ptl_mass']
    z_init = config['undulator']['position']['z'] - config['undulator']['length']
    und_num_per = int(config['undulator']['length'] / config['undulator']['period'])

    # Retrieve single photon energy to compute from command line argument
    ph_energy = float(sys.argv[1])

    # Define output file name
    if ph_energy >= 10_000:
        output_file_name = str(ph_energy / 1e3) + '_keV'
    elif ph_energy > 0 and ph_energy < 10_000:
        output_file_name = '0' + str(ph_energy / 1e3) + '_keV'

    # Create point-like electron beam with energy spread
    e_beam = beam.create(
        beam_current=config['beam']['avg_current'],
        gamma=gamma,
        x_init=config['beam']['init']['x'],
        y_init=config['beam']['init']['y'],
        z_init=z_init,
        xp_init=config['beam']['init']['xp'],
        yp_init=config['beam']['init']['yp'],
        sigma_x=1e-16,
        sigma_y=1e-16,
        sigma_xp=1e-16,
        sigma_yp=1e-16,
        sigma_z=config['beam']['sigma']['z'],
    )

    # Create planar undulator
    und_mag = undulator.create_planar(
        und_num_per=und_num_per,
        und_per=config['undulator']['period'],
        und_k=config['undulator']['k'],
        und_phase=config['undulator']['phase'],
        und_field_dir=config['undulator']['field_dir'],
        und_x_cent=config['undulator']['position']['x'],
        und_y_cent=config['undulator']['position']['y'],
        und_z_cent=config['undulator']['position']['z'],
    )

    # --- Create mesh ---
    det_mesh = mesh.create(
        engy_min=ph_energy,
        engy_max=ph_energy,
        engy_n=1,
        mesh_x_min=config['simulation']['mesh']['x']['min'],
        mesh_x_max=config['simulation']['mesh']['x']['max'],
        mesh_x_n=config['simulation']['mesh']['x']['n'],
        mesh_y_min=config['simulation']['mesh']['y']['min'],
        mesh_y_max=config['simulation']['mesh']['y']['max'],
        mesh_y_n=config['simulation']['mesh']['y']['n'],
        det_z=config['simulation']['detector_z'],
    )

    # --- Create wavefront ---
    # wfr = wavefront.create(mesh=det_mesh, beam=e_beam)

    # --- Compute synchrotron radiation ---
    radiation.compute(
        beam=e_beam,
        b_field=und_mag,
        mesh=det_mesh,
        comp_method='auto-undulator',
        comp_rel_prec=0.01,
        n_ptl=config['beam']['ptl_n'],
        rad_charac='intensity',
        output_file=output_file_name + '.dat',
        rand_method='halton',
        beam_convo=False,
    )

    # --- Extract results for further processing ---
    profile, x_axis, y_axis, energy = helper.read_single_energy_profile(output_file_name + '.dat')

    np.savez(output_file_name + '_single', arr=profile)

    x_grid = np.linspace(x_axis[0], x_axis[1], x_axis[2])
    y_grid = np.linspace(y_axis[0], y_axis[1], y_axis[2])

    # --- Apply beam size and angular divergence convolution
    profile = radiation.convolve_beam_size_and_divergence(
        profile,
        x_grid,
        y_grid,
        config['beam']['sigma']['x'],
        config['beam']['sigma']['y'],
        config['beam']['sigma']['xp'],
        config['beam']['sigma']['yp'],
        config['simulation']['detector_z'],
        config['undulator']['length'],
        n_samples=1_000_000,
    )

    # --- Save result to compressed NumPy array ---
    np.savez(output_file_name, arr=profile)


if __name__ == "__main__":
    main()

# # --- Define undulator properties ---
# und_l: float = 1.55  # length [m]
# und_per: float = 28.4 * 1e-3  # period [m]
# und_num_per: int = int(und_l / und_per)  # number of undulator periods
# und_k: float = 2.5245  # strength (dimensionless)
# und_phase: float = 0.0  # initial phase [rad]
# und_field_dir: str = 'v'  # direction of the magnetic field: 'v' for vertical, 'h' for horizontal
# und_x_cent: float = 0  # horizontal position of the center of the undulator [m]
# und_y_cent: float = 0  # vertical position of the center of the undulator [m]
# und_z_cent: float = 0  # longitudinal position of the center of the undulator [m]

# # --- Define beam properties ---
# avg_current: float = 3e-3  # CESR one bunch current at 145 mA in 9x5
# electron_mass: float = 510.99890221e3  # 511 keV mass
# electron_energy: float = 6e9  # 6 GeV energy
# gamma: float = electron_energy / electron_mass
# sigma_z: float = 0  # 8.208e-4  # relative std dev energy (longitudinal) spread
# sigma_x: float = 0.555e-3  # horizontal std dev size [m]
# sigma_xp: float = 0.05014e-3  # horizontal std dev angular divergence [rad]
# sigma_y: float = 0.02904e-3  # vertical std dev size [m]
# sigma_yp: float = 0.01025e-3  # vertical std dev angular divergence [rad]

# x_init: float = 0  # initial horizontal position of the beam [m]
# y_init: float = 0  # initial vertical position of the beam [m]
# z_init: float = und_z_cent - und_l  # initial longitudinal position of the beam [m]
# xp_init: float = 0  # initial horizontal momentum (angle)
# yp_init: float = 0  # initial vertical momentum (angle)

# # --- Define simulation parameters ---
# beam_comp = 'custom_conv'  # Choices are: 'mc', 'custom_conv' and 'srw_conv'
# engy_min: float = 2875.673742644  # minimum photon energy to compute [eV]
# engy_max: float = 2875.673742644  # maximum photon energy to compute [eV]
# engy_n: int = 1  # number of energy steps between min and max energies
# mesh_x_min: float = -2500e-6  # minimum horizontal coordinate of the detector mesh screen [m]
# mesh_x_max: float = 2500e-6  # maximum horizontal coordinate of the detector mesh screen [m]
# mesh_x_n: int = 500  # number of mesh points on the horizontal mesh screen
# mesh_y_min: float = -2500e-6  # minimum vertical coordinate of the detector mesh screen [m]
# mesh_y_max: float = 2500e-6  # maximum vertical coordinate of the detector mesh screen [m]
# mesh_y_n: int = 500  # number of mesh points on the vertical mesh screen
# det_z: float = 10  # longitudinal position of the detector z [m]
# output_file_name = './single_e_10m.dat'  # name (path) of the file containing computation results
