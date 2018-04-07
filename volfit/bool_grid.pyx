cimport cython
cimport numpy as np
from libc.math cimport ceil

import numpy as np

ctypedef np.float64_t DTYPE64
ctypedef np.float32_t DTYPE32

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def bool_grid(np.ndarray[DTYPE32, ndim=2] coordinates,
              np.ndarray[DTYPE64, ndim=4] mesh,
              np.ndarray[DTYPE64, ndim=1] radii,
              float probe):
    cdef int x
    cdef int y
    cdef int z
    cdef int[3] bin_xyz
    cdef int i
    cdef int dim
    cdef float sqdist
    cdef DTYPE32[3] atom
    cdef int max_n_cells
    cdef int[3] min_idx
    cdef int[3] max_idx
    cdef float spacing = mesh[2, 0, 0, 1] - mesh[2, 0, 0, 0]
    cdef  np.ndarray[np.uint8_t, ndim=3, cast=True] gbool = np.zeros((mesh.shape[1], mesh.shape[2], mesh.shape[3]), dtype=bool)
    
    for i in range(radii.shape[0]):
        atom = coordinates[i]
        max_n_cells = <int>(ceil((radii[i] + probe) // spacing) + 1)
        for dim in range(3):
            bin_xyz[dim] = <int>((atom[dim] - mesh[dim, 0, 0, 0]) // spacing)
            min_idx[dim] = 0
            if bin_xyz[dim] - max_n_cells > 0:
                min_idx[dim] = bin_xyz[dim] - max_n_cells
            max_idx[dim] = gbool.shape[dim]
            if bin_xyz[dim] + max_n_cells < gbool.shape[dim]:
                max_idx[dim] = bin_xyz[dim] + max_n_cells
        for x in range(min_idx[0], max_idx[0]):
            for y in range(min_idx[1], max_idx[1]):
                for z in range(min_idx[2], max_idx[2]):
                    sqdist = 0
                    for dim in range(mesh.shape[0]):
                        sqdist += (mesh[dim, x, y, z] - coordinates[i, dim]) ** 2
                    gbool[x, y, z] |= sqdist < (radii[i] + probe) ** 2
    return gbool


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)
def bool_grid_flat(np.ndarray[DTYPE32, ndim=2] coordinates,
                   np.ndarray[DTYPE64, ndim=3] mesh,
                   np.ndarray[DTYPE64, ndim=1] radii,
                   float probe, int axis=0):
    cdef int x
    cdef int y
    cdef int z
    cdef int[3] bin_xyz
    cdef int i
    cdef int dim
    cdef float sqdist
    cdef DTYPE32[3] atom
    cdef int max_n_cells
    cdef int[2] min_idx
    cdef int[2] max_idx
    cdef float spacing = mesh[0, 1, 0] - mesh[0, 0, 0]
    cdef  np.ndarray[np.uint8_t, ndim=2, cast=True] gbool
    cdef int[2] dimensions

    
    # Select the 2 dimensions of interest
    i = 0
    for dim in range(3):
        if dim != axis:
            dimensions[i] = dim
            i += 1
    # Build the result array
    gbool = np.zeros(
        (mesh.shape[1], mesh.shape[2]),
        dtype=bool
    )
    # Do the job
    for i in range(radii.shape[0]):
        atom = coordinates[i]
        max_n_cells = <int>(ceil((radii[i] + probe) // spacing) + 1)
        for dim in range(2):
            bin_xyz[dim] = <int>((atom[dimensions[dim]] - mesh[dim, 0, 0]) // spacing)
            min_idx[dim] = 0
            if bin_xyz[dim] - max_n_cells > 0:
                min_idx[dim] = bin_xyz[dim] - max_n_cells
            max_idx[dim] = gbool.shape[dim]
            if bin_xyz[dim] + max_n_cells < gbool.shape[dim]:
                max_idx[dim] = bin_xyz[dim] + max_n_cells
        for x in range(min_idx[0], max_idx[0]):
            for y in range(min_idx[1], max_idx[1]):
                sqdist = 0
                for dim in range(2):
                    #sqdist += (mesh[dim, x, y] - coordinates[i, dim]) ** 2
                    sqdist += (mesh[dim, x, y] - atom[dimensions[dim]]) ** 2
                gbool[x, y] |= sqdist < (radii[i] + probe) ** 2
    return gbool
