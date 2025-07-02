import numpy as np


def read_single_energy_profile(filename):
    """
    Reads an SRW ASCII file with intensity data (characteristic 0) and returns a 2D array of
    intensity. Assumes data format: one component, with shape (ny, nx).
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Separate header and data
    header_lines = [line.strip() for line in lines if line.strip().startswith('#')]
    data_lines = [
        line.strip() for line in lines if not line.strip().startswith('#') and line.strip() != ''
    ]

    def extract_value(line):
        return float(line.split('#')[1])

    x_start = extract_value(header_lines[4])
    x_end = extract_value(header_lines[5])
    nx = int(extract_value(header_lines[6]))

    y_start = extract_value(header_lines[7])
    y_end = extract_value(header_lines[8])
    ny = int(extract_value(header_lines[9]))

    profile = np.array([float(val) for val in data_lines])

    # Reshape profile: assume shape (ny, nx)
    profile = profile.reshape((ny, nx)).T  # raw data ix (y, x), transpose to (x, y)

    # energy = filename.split('test')[1].split('.dat')[0]

    energy = 0

    return profile, (y_start, y_end, ny), (x_start, x_end, nx), energy
