import numpy as np
import torch
import torch.nn.functional as F
#from pytransform3d.rotations import matrix_from_axis_angle, axis_angle_from_two_directions
#from pytransform3d.rotations import euler_from_matrix

import numpy as np
import trimesh

def get_plane(fpath):
    crimefile = open(fpath, 'r')
    line = crimefile.readlines()[0].replace('\n', '')
    data = np.array(line.split(',')).astype(float).reshape(2, 3)
    normals = data[0, :]
    origin = data[1, :]
    return origin, normals

def get_mesh_and_plane(mesh_path, plane_path):
    mesh_o = trimesh.load_mesh(mesh_path)
    origin, normals = get_plane(plane_path)
    scene = trimesh.Scene([mesh_o, trimesh.path.creation.grid(side=20,
                                                  plane_origin = origin,
                                                  plane_normal = normals)])
    return scene

def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)

def normal_to_rotation_matrix(normal):
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Base vector (up direction)
    base = np.array([0, 0, 1])

    # Cross product to find the rotation axis
    v = np.cross(base, normal)

    # Normalize v
    v = v / np.linalg.norm(v)

    # Dot product to find the cosine of the angle
    c = np.dot(base, normal)

    # Compute the sine of the angle
    s = np.sqrt(1 - c ** 2)

    # Identity matrix
    I = np.eye(3)

    # Outer product of v
    v_outer = np.outer(v, v)

    # Cross product matrix of v
    vx, vy, vz = v
    v_cross = np.array([[0, -vz, vy],
                        [vz, 0, -vx],
                        [-vy, vx, 0]])

    # Rotation matrix
    R = v_outer + c * (I - v_outer) + s * v_cross

    return R

def rotation_matrix_to_normal(R):
    # Extract the third column (normal to the plane)
    normal = R[:, 2]

    # Normalize the normal vector
    normal = normal / torch.linalg.norm(normal)

    return normal


# def normal_to_rotation_matrix(normal):
#     base = np.array([0, 0, 1])
#     normal = normal / np.linalg.norm(normal)
#
#     axis_angle = axis_angle_from_two_directions(base, normal)
#     rotation_matrix = matrix_from_axis_angle(axis_angle)
#
#     return rotation_matrix


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def homography_mae(y_true, y_pred, convention='XYZ'):
    # Extract rotation matrices
    R_true = y_true[:, :3, :3]
    R_pred = y_pred[:, :3, :3]

    # Convert rotation matrices to Euler angles
    euler_true = matrix_to_euler_angles(R_true, convention)
    euler_pred = matrix_to_euler_angles(R_pred, convention)

    # Calculate MAE difference of Euler angles
    rotation_mae = mae(euler_true, euler_pred)

    # Extract translation vectors
    t_true = y_true[:, :3, 3]
    t_pred = y_pred[:, :3, 3]

    # Calculate MAE difference of translation vectors
    translation_mae = mae(t_true, t_pred)

    return torch.rad2deg(rotation_mae), translation_mae

# a function that converts a rotation matrix to an axis-angle representation, then computes the norm.
# This function calculates the rotation angle from the trace (sum of diagonal elements) of the rotation matrix,
# using the relationship between the trace and the rotation angle.
def rotation_matrix_to_angle(R):
    """
    Convert a rotation matrix to axis-angle representation then compute the norm.

    Args:
        R: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        angle: rotation angle for each matrix in R as a tensor of shape (...).
    """

    # Ensure rotation matrix is square
    assert R.shape[-2] == R.shape[-1] == 3

    # Compute rotation angle
    cos_theta = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
    # clamp values to handle numerical issues
    cos_theta = torch.clamp(cos_theta, -1., 1.)
    theta = torch.acos(cos_theta)

    return theta

# calculates the MAE of the relative rotation angles (in the axis-angle representation) and the translation vectors.
# This should provide a more robust metric for rotation error, especially for large rotations.
# Note that the rotation MAE is now a scalar value, representing the average magnitude of the relative rotations.
def homography_mae_robust(y_true, y_pred, normals):
    # Extract rotation matrices
    R_true = y_true[:, :3, :3]
    R_pred = y_pred[:, :3, :3]

    # Calculate relative rotation matrices
    R_rel = torch.matmul(R_pred, R_true.transpose(-2, -1))

    # Convert relative rotation matrices to angles
    angle_rel = rotation_matrix_to_angle(R_rel)

    # Calculate MAE of relative angles
    rotation_mae = torch.mean(torch.abs(angle_rel))

    # Extract translation vectors
    t_true = y_true[:, :3, 3]
    t_pred = y_pred[:, :3, 3]

    # Calculate MAE difference of translation vectors
    #translation_mae = torch.norm(t_true - t_pred, dim=1).mean()
    dot_product = torch.sum(normals * (t_true - t_pred), dim=1)
    dot_sum = torch.abs(dot_product)
    translation_dist = torch.mean(dot_sum)


    return torch.rad2deg(rotation_mae), translation_dist

# # poses impl
# # batch*n
# def normalize_vector(v):
#     batch = v.shape[0]
#     v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
#     gpu = v_mag.get_device()
#     if gpu < 0:
#         eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cpu'))
#     else:
#         eps = torch.autograd.Variable(torch.FloatTensor([1e-8])).to(torch.device('cuda:%d' % gpu))
#     v_mag = torch.max(v_mag, eps)
#     v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
#     v = v / v_mag
#     return v
#
#
# # u, v batch*n
# def cross_product(u, v):
#     batch = u.shape[0]
#     # print (u.shape)
#     # print (v.shape)
#     i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
#     j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
#     k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
#
#     out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3
#
#     return out


# poses batch*6
# poses
# def compute_rotation_matrix_from_ortho6d(poses):
#     x_raw = poses[:, 0:3]  # batch*3
#     y_raw = poses[:, 3:6]  # batch*3
#
#     x = normalize_vector(x_raw)  # batch*3
#     z = cross_product(x, y_raw)  # batch*3
#     z = normalize_vector(z)  # batch*3
#     y = cross_product(z, x)  # batch*3
#
#     x = x.view(-1, 3, 1)
#     y = y.view(-1, 3, 1)
#     z = z.view(-1, 3, 1)
#     matrix = torch.cat((x, y, z), 2)  # batch*3*3
#     return matrix


# normal = np.array([1, 0, 1])
# R = normal_to_rotation_matrix(normal)
#
# R = torch.tensor(R).unsqueeze(0)
# torch6d = matrix_to_rotation_6d(R)
# rot_torch = rotation_6d_to_matrix(torch6d)
#
# rot_origin = compute_rotation_matrix_from_ortho6d(torch6d)
#
#
#
# # Convert the rotation matrix to Euler angles in radians
# euler_angles_radians = euler_from_matrix(R, 0, 1, 2, extrinsic=True)
#
#
# # Convert Euler angles to degrees
# euler_angles_degrees = np.degrees(euler_angles_radians)
#
# # Print the Euler angles in degrees
# print("Euler angles (in degrees):", euler_angles_degrees)
# newNorm = rotation_matrix_to_normal(R)
# g = 2
