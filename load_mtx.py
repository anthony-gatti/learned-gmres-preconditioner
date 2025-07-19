import numpy as np
from scipy.sparse import coo_matrix

def load_mtx(filename):
    with open(filename, 'r') as f:
        # skip header/comments
        while True:
            line = f.readline()
            if line == '':
                raise ValueError("Reached end of file without finding size line.")
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            break

        try:
            rows, cols, nnz = map(int, line.split())
        except ValueError:
            raise ValueError(f"Failed to parse matrix dimensions: '{line}'")

        data = []
        row_idx = []
        col_idx = []

        for _ in range(nnz):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file while reading matrix data.")
            parts = line.strip().split()
            if len(parts) != 4:
                raise ValueError(f"Unexpected line format: {parts}")
            i, j = int(parts[0]) - 1, int(parts[1]) - 1
            real, imag = float(parts[2]), float(parts[3])
            row_idx.append(i)
            col_idx.append(j)
            data.append(complex(real, imag))

    return coo_matrix((data, (row_idx, col_idx)), shape=(rows, cols)).tocsr()