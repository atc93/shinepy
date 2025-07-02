import srwlib as sp  # a Python module, sp for 'srw' and 'python package'


def create(*, mesh: sp.SRWLRadMesh, beam: sp.SRWLPartBeam) -> sp.SRWLWfr:
    wfr = sp.SRWLWfr()
    wfr.mesh = mesh
    wfr.partBeam = beam
    wfr.allocate(mesh.ne, mesh.nx, mesh.ny)
    wfr.presCA = 0  # physical representation: 0 for coordinates, 1 for angles
    wfr.presFT = 0  # domain representation: 0 for frequency, 1 for time

    return wfr
