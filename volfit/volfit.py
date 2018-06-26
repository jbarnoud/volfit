import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.distances as mad
import MDAnalysis.analysis.align as maa
import nglview as nv
import gridData
import numba
import collections
import functools

from .bool_grid import bool_grid, bool_grid_flat

AA_VDW_RADII = {  # in Å
    'C': 1.7,
    'H': 1.2,
    #'H': 0,
    'N': 1.55,
    'O': 1.52,
    'P': 1.8,
    'S': 1.8,
    'opls_146': 1.2,
    #'opls_146': 0,
    'opls_145': 1.7,
    'HC': 1.2,
    #'HC': 0,
}
AA_PROBE = 1.85
#AA_PROBE = 1.14
#AA_PROBE = 0
SPACING = 0.05

#SIGMA = 4
SIGMA = 3.3
TRADIUS = 0.5 * (2 ** (1/6) * SIGMA)

MAPPINGS = {
    'BENZ': [
        ['C1', 'C2', 'H1', 'H2'],
        ['C3', 'C4', 'H3', 'H4'],
        ['C5', 'C6', 'H5', 'H6'],
    ],
    'NAPH': [
        ['C8', 'C7', 'H6', 'H7'],
        ['C6', 'C5', 'H5', 'H4'],
        ['C9', 'C4'],
        ['C1', 'C10', 'H1', 'H8'],
        ['C2', 'C3', 'H2', 'H3'],
    ],
    'TECE': [
        ['C14', 'C10', 'H9', 'H6'],
        ['C17', 'C18', 'H11', 'H12'],
        ['C7', 'C12'],
        ['C2', 'C1', 'H1', 'H2'],
        ['C5', 'C9', 'H4', 'H7'],
        ['C11', 'C6', 'H5'],
        ['C4', 'C3', 'H3'],
        ['C15', 'C16', 'H10'],
        ['C8', 'C13', 'H8'],
    ],
    'CORO': [
		['H3', 'C1', 'C6', 'H4'],
		['C13', 'H6', 'C12', 'H5'],
		['H9', 'C20', 'C21', 'H10'],
		['H11', 'C23', 'C24', 'H12'],
		['H2', 'C15', 'C16', 'H1'],
		['H7', 'C17', 'C18', 'H8'],
		['C14', 'C8'],
		['C22', 'C10'],
		['C2', 'C3'],
		['C11', 'C9'],
		['C5', 'C4'],
		['C19', 'C7'],
	],
}


def read_ndx_mapping(lines):
    mapping = []
    current = None
    for line in lines:
        if line.startswith('['):
            current = []
            mapping.append(current)
        elif current is None:
            raise IOError('NDX file cannot have indices before a header.')
        else:
            current.extend(int(x) - 1 for x in line.split())
    return mapping


def orient(u):
    u.atoms.positions -= u.atoms.center_of_mass()
    for dim in range(3):
        pa = u.atoms.principal_axes()[dim, :]
        axis = np.zeros((3, ))
        axis[dim] = 1
        angle = np.degrees(mda.lib.mdamath.angle(pa, axis))
        u.atoms.rotateby(angle, np.cross(pa, axis), point=np.array([0, 0, 0]))


def build_mesh(coordinates, radii, spacing, probe):
    maxes = coordinates.max(axis=0) + spacing + radii.max() + probe
    mines = coordinates.min(axis=0) - spacing - radii.max() - probe
    xx = np.arange(mines[0], maxes[0], spacing)
    yy = np.arange(mines[1], maxes[1], spacing)
    zz = np.arange(mines[2], maxes[2], spacing)
    mesh = np.stack(np.meshgrid(xx, yy, zz, indexing='ij'))
    return mesh


def build_mesh_flat(coordinates, radii, spacing, probe, axis=0):
    maxes = coordinates.max(axis=0) + spacing + radii.max() + probe
    mines = coordinates.min(axis=0) - spacing - radii.max() - probe
    dimensions = [0, 1, 2]
    del dimensions[axis]
    dim_arrays = [
        np.arange(mines[dim], maxes[dim], spacing) for dim in dimensions
    ]
    mesh = np.stack(np.meshgrid(*dim_arrays, indexing='ij'))
    return mesh


def get_score(reference, mobile):
    return np.fabs(reference.astype(int) - mobile.astype(int)).sum()


def compare_models(u, ucg_base, mappings, spacing, probe):
    # Prepare the AA structure
    mda.lib.mdamath.make_whole(u.atoms)
    orient(u)
    radii_aa = np.array([AA_VDW_RADII[elem] for elem in u.atoms.types])
    
    # Create a naive mapped CG structure so we can align the real CG structure
    centers = np.stack([u.atoms[mapping].center_of_geometry() for mapping in mappings])
    ucg_centers = mda.Universe(ucg_base._topology, centers[None, :])
    
    # Prepare the CG structure
    maa.alignto(ucg_base, ucg_centers)
    radii_cg = np.repeat(TRADIUS, len(ucg_base.atoms))
    
    # Build the mesh
    total_coords = np.vstack([u.atoms.positions, ucg_base.atoms.positions])
    total_radii = np.hstack([radii_aa, radii_cg])
    mesh = build_mesh(total_coords, total_radii, spacing, probe)
    
    # Compute the AA volume
    aa_grid = bool_grid(u.atoms.positions, mesh, radii_aa, probe)
    aa_volume = aa_grid.sum() * spacing ** 3
    
    # Compute the CG volume
    cg_grid = bool_grid(ucg_base.atoms.positions, mesh, radii_cg, probe)
    cg_volume = cg_grid.sum() * spacing ** 3
    
    # Compute the mismatch
    mismatch = get_score(aa_grid, cg_grid) * spacing ** 3
    
    diff_grid = aa_grid.astype(int) - cg_grid.astype(int)
    
    return aa_volume, cg_volume, mismatch, diff_grid


def compare_flat(u, ucg_base, mappings, spacing, probe, axis=0):
    # Prepare the AA structure
    mda.lib.mdamath.make_whole(u.atoms)
    orient(u)
    radii_aa = np.array([AA_VDW_RADII[elem] for elem in u.atoms.types])
    
    # Create a naive mapped CG structure so we can align the real CG structure
    centers = np.stack([u.atoms[mapping].center_of_geometry() for mapping in mappings])
    ucg_centers = mda.Universe(ucg_base._topology, centers[None, :])
    
    # Prepare the CG structure
    maa.alignto(ucg_base, ucg_centers)
    radii_cg = np.repeat(TRADIUS, len(ucg_base.atoms))
    
    # Build the mesh
    total_coords = np.vstack([u.atoms.positions, ucg_base.atoms.positions])
    total_radii = np.hstack([radii_aa, radii_cg])
    mesh = build_mesh_flat(total_coords, total_radii, spacing, probe, axis=axis)
    
    # Compute the AA volume
    aa_grid = bool_grid_flat(u.atoms.positions, mesh, radii_aa, probe, axis=axis)
    aa_volume = aa_grid.sum() * spacing ** 2
    
    # Compute the CG volume
    cg_grid = bool_grid_flat(ucg_base.atoms.positions, mesh, radii_cg, probe, axis=axis)
    cg_volume = cg_grid.sum() * spacing ** 2
    
    # Compute the mismatch
    mismatch = get_score(aa_grid, cg_grid) * spacing ** 2
    
    diff_grid = aa_grid.astype(int) - cg_grid.astype(int)
    
    return aa_volume, cg_volume, mismatch, diff_grid


def get_attraction(centers, coordinates):
    #return 50 * np.sum(np.exp(np.sum((coordinates - centers) ** 2, axis=1)))
    #return np.sum(np.sum((coordinates - centers) ** 2, axis=1)** 10)
    return 0


def minimize_score(reference, mobile_coords, centers, mesh, spacing, radii, probe, steps, exclude=(), axis=None):
    dimensions = [0, 1, 2]
    if axis is None:
        gridder = bool_grid
        mesh_axes = (1, 2, 3)
    else:
        gridder = functools.partial(bool_grid_flat, axis=axis)
        mesh_axes = (1, 2)
        del dimensions[axis]
           
    mobile = gridder(mobile_coords, mesh, radii, probe)
    error = get_score(reference, mobile)
    attraction = get_attraction(centers, mobile_coords)
    score = error + attraction
    scores_mc = np.zeros((steps + 1, ))
    scores = np.zeros((steps + 1, ))
    errors = np.zeros((steps + 1, ))
    attractions = np.zeros((steps + 1, ))
    scores[0] = score
    scores_mc[0] = score
    errors[0] = error
    attractions[0] = attraction
    traj = np.zeros((steps, mobile_coords.shape[0], mobile_coords.shape[1]), dtype=np.float32)
    coordinates = mobile_coords
    traj[0] = coordinates
    mines = mesh.min(axis=mesh_axes)
    maxes = mesh.max(axis=mesh_axes)
    accepted = 0
    volume_cg = 0
    
    for iteration in range(steps):
        scores_mc[iteration + 1] = scores_mc[iteration]
        scores[iteration + 1] = scores[iteration]
        errors[iteration + 1] = errors[iteration]
        
        move_idx = np.random.randint(0, coordinates.shape[0])
        if move_idx in exclude:
            continue

        dim_idx = np.random.choice(dimensions)
        translation = np.random.normal(0, 0.5 * spacing, 1)
        new_coordinates = coordinates.copy()
        new_coordinates[move_idx, dim_idx] += translation

        gbool_cg_mc = gridder(new_coordinates, mesh, radii, probe)

        error = get_score(reference, gbool_cg_mc)
        attraction = get_attraction(centers, new_coordinates)
        score_mc = error + attraction
        is_in_box = (np.all(new_coordinates[:, dimensions] - radii[:, None] > mines)
                     and np.all(new_coordinates[:, dimensions] + radii[:, None] < maxes))
        if score_mc < score and is_in_box:
            accepted += 1
            coordinates = new_coordinates
            score = score_mc
            traj[accepted] = coordinates
            volume_cg = gbool_cg_mc.sum()
        scores_mc[iteration + 1] = score_mc
        scores[iteration + 1] = score
        errors[iteration + 1] = error
        attractions[iteration + 1] = attraction

        if iteration % 100 == 0:
            print('{} -- {} accepted -- score {} -- volume CG {} cells'
                  .format(iteration, accepted, score, volume_cg))
    return traj[:accepted + 1], scores, scores_mc, attractions, errors
