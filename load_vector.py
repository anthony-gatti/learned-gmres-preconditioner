import numpy as np

def load_vector(filename):
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
            rows, cols = map(int, line.split())
        except ValueError:
            raise ValueError(f"Failed to parse vector dimensions: '{line}'")

        data = []
        for _ in range(rows):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file while reading vector data.")
            real, imag = map(float, line.strip().split())
            data.append(complex(real, imag))

        return np.array(data).reshape((rows,))