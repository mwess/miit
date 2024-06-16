import numpy
import scipy.ndimage as nd


def com_affine_matrix(fixed: numpy.array, moving: numpy.array):
    mat = numpy.eye(3)
    fixed_com = nd.center_of_mass(fixed)
    moving_com = nd.center_of_mass(moving)
    mat[0, 2] = fixed_com[0] - moving_com[0]
    mat[1, 2] = fixed_com[1] - moving_com[1]
    return mat

def write_mat_to_file(mat, fname):
    out_str = f'{mat[0,0]} {mat[0,1]} {mat[0,2]}\n{mat[1,0]} {mat[1,1]} {mat[1,2]}\n{mat[2,0]} {mat[2,1]} {mat[2,2]}'
    with open(fname, 'w') as f:
        f.write(out_str)     