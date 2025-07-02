import srwlib as sp  # a Python module, sp for 'srw' and 'python package'


def create(
    *,
    beam_current: float,
    gamma: float,
    x_init: float,
    y_init: float,
    z_init: float,
    xp_init: float,
    yp_init: float,
    sigma_x: float,
    sigma_y: float,
    sigma_xp: float,
    sigma_yp: float,
    sigma_z: float,
) -> sp.SRWLPartBeam:

    beam = sp.SRWLPartBeam()

    beam.Iavg = beam_current  # average beam current [A]

    beam.partStatMom1.x = x_init  # initial horizontal position [m]
    beam.partStatMom1.y = y_init  # initial vertical position [m]
    beam.partStatMom1.z = z_init  # initial longitudinal position [m]
    beam.partStatMom1.xp = xp_init  # type: ignore # initial horizontal angle [rad]
    beam.partStatMom1.yp = yp_init  # initial vertical angle [rad]
    beam.partStatMom1.gamma = gamma  # relativistic gamma = E / mc2

    # Specify the transverse beam size and divergence, energy (longitudinal) spread
    beam.arStatMom2 = [
        sigma_x**2,
        0,
        sigma_xp**2,
        sigma_y**2,
        0,
        sigma_yp**2,
        0,
        0,
        0,
        0,
        sigma_z**2,
        0,
        0,
    ]

    return beam
