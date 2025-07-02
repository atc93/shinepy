import array

import numpy as np
import srwlib as sp  # a Python module, sp for 'srw' and 'python package'
from matplotlib import pyplot


def create_planar(
    *,  # only keyword arguments allowed
    und_num_per: int,
    und_per: float,
    und_k: float,
    und_phase: float,
    und_field_dir: str,
    und_x_cent: float,
    und_y_cent: float,
    und_z_cent: float,
) -> sp.SRWLMagFldC:

    # compute B field using standard K formula approximated to
    # K≈0.934*B_0[T]*λ_u[cm], see for instance Weidemann 21.4
    # magnetifc field [T] given the undulator period in [cm]
    b_field_intensity = und_k / (0.934 * und_per * 100)

    # instantiate undulator magnetic field SRWLMagFldU via an inherited instance of the undulator
    # SRWLMagFldH class. SRWLMagFldH arguments are:
    #  harmonic number: default is 1
    #  field plane (vertical 'v' or horizontal 'h'),
    #  magnetic field amplitude [T]
    #  initial phase [rad]
    #  longitudinal symmetry: 1 is symmetric, -1 is antisymmetric
    #  transverse dependence coefficient: default is 1
    # other arguments are undulator period and number of periods
    und_field = sp.SRWLMagFldU(
        [sp.SRWLMagFldH(1, und_field_dir, b_field_intensity, und_phase, 1, 1)],
        und_per,
        und_num_per,
    )

    # instantiate a container class to store the magnetic field. First 4 arguments are:
    #  magnetic field array (SRWLMagFldU) as a list, array or tuple
    #  horizontal center position of magnetic field element
    #  vertical center position of magnetic field element
    #  longitudinal center position of magnetic field element
    x_c = array.array('d', [und_x_cent])
    y_c = array.array('d', [und_y_cent])
    z_c = array.array('d', [und_z_cent])

    und_cont = sp.SRWLMagFldC([und_field], x_c, y_c, z_c)

    return und_cont
