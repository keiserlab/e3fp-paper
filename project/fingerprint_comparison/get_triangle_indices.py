"""Get indices for equally sized slices of lower triangle matrix.

Author: Seth Axen
E-mail: seth.axen@gmail.com
"""
import sys


def get_triangle_indices(dim, n):
    """Get indices for equally sized slices of lower triangle matrix."""
    num_entries = dim * (dim - 1) / 2.
    min_batch_size = int(num_entries / n)
    start_inds = [1]
    while len(start_inds) < n:
        i = start_inds[-1]
        j = min_batch_size + (i - 1) * (i - 2) / 2.
        end_ind = int(round((1 + (1 + 8 * j)**.5) / 2., 0))
        start_inds.append(end_ind)
    start_end_inds = [(x - 1, y - 1) for x, y
                      in zip(start_inds, start_inds[1:] + [dim + 1])]
    return start_end_inds


if __name__ == "__main__":
    usage = "python get_triangle_indices.py <dimensions> <batch_num>"
    try:
        dim, n = sys.argv[1:]
        dim = int(dim)
        n = int(n)
    except:
        sys.exit(usage)
    indices = get_triangle_indices(dim, n)
    for start, end in indices:
        print("{}...{}".format(start, end))
