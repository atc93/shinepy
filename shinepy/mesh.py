import srwlib as sp  # a Python module, sp for 'srw' and 'python package'


def create(
    engy_min: float,
    engy_max: float,
    engy_n: int,
    mesh_x_min: float,
    mesh_x_max: float,
    mesh_x_n: int,
    mesh_y_min: float,
    mesh_y_max: float,
    mesh_y_n: int,
    det_z: float,
):
    """_summary_

    Args:
        engy_min (float): minimum photon energy to compute [eV]
        engy_max (float): maximum photon energy to compute [eV]
        engy_n (int): number of energy steps between min and max energies
        mesh_x_min (float): minimum horizontal coordinate of the detector mesh screen [m]
        mesh_x_max (float): maximum horizontal coordinate of the detector mesh screen [m]
        mesh_x_n (int): number of mesh points on the horizontal mesh screen
        mesh_y_min (float): minimum vertical coordinate of the detector mesh screen [m]
        mesh_y_max (float): maximum vertical coordinate of the detector mesh screen [m]
        mesh_y_n (int): number of mesh points on the vertical mesh screen
        det_z (float): longitudinal position of the detector mesh screen [m]

    Returns:
        _type_: _description_
    """
    mesh = sp.SRWLRadMesh(
        _eStart=engy_min,  # type: ignore
        _eFin=engy_max,  # type: ignore
        _ne=engy_n,  # type: ignore
        _xStart=mesh_x_min,  # type: ignore
        _xFin=mesh_x_max,  # type: ignore
        _nx=mesh_x_n,  # type: ignore
        _yStart=mesh_y_min,  # type: ignore
        _yFin=mesh_y_max,  # type: ignore
        _ny=mesh_y_n,  # type: ignore
        _zStart=det_z,  # type: ignore
    )

    return mesh
