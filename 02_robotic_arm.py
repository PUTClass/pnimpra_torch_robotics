import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def rotation(theta):
    """
    theta: [K] angles in radians
    returns:
        R: [K, 2, 2] Rotation matrices 
    """
    batch_size = theta.shape[0]
    R = torch.empty(batch_size, 2, 2)

    c = torch.cos(theta)
    s = torch.sin(theta)

    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c

    return R


def forward_kinematics(q, L):
    """
    q: [K] joint angles
    L: [K] link lengths
    returns:
        T: [K, 3, 3]
        joint_pos: [K, 2]
    """
    # Planar robots have a cumulative property of angles e.g. y_pos = l_1*sin(Φ_1) + l_2*sin(Φ_1+Φ_2)
    q_abs = torch.cumsum(q, dim=0)

    # Rotations for each link in world frame
    R = rotation(q_abs)
    
    # Local link vectors - x-axis points forwards
    p = torch.stack([L, torch.zeros_like(L)], dim=1)
    
    # Rotate link vectors to world frame
    p_world = (R @ p.unsqueeze(2)).squeeze(2)
    
    # The position of the end of each link in world frame
    # We still need cumsum here to add up the link vectors for absolute positions
    joint_pos = torch.cumsum(p_world, dim=0)

    # Construct the homogeneous transformation matrices
    K = q.shape[0]
    T = torch.zeros(K, 3, 3, dtype=q.dtype, device=q.device)
    T[:, :2, :2] = R
    T[:, :2, 2] = joint_pos
    T[:, 2, 2] = 1.0

    return T, joint_pos


# Joint angles (radians)

q = torch.tensor(
    [torch.pi/3, torch.pi/3, 0.0]
)

# Link lengths
L = torch.tensor(
    [0.7, 0.7, 0.2]
)

T, pos = forward_kinematics(q, L)

### Visualization ###

robot_base_pos = np.zeros(shape=(1, 2))
robot_links_pos = pos.numpy()
robot_links_pos = np.concat((robot_base_pos, robot_links_pos), axis=0)

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter([0], [0], color='black', s=100, zorder=5, label='Robot Base')
arm_line, = ax.plot(robot_links_pos[:, 0], robot_links_pos[:, 1], 
                    'o-', lw=4, color='orange', alpha=0.8, label='Robot Arm')

ax.set_xlim(-2.5, 3.0)
ax.set_ylim(-1.0, 3.0)
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Robotic arm visualization')
ax.legend()
ax.grid()

plt.show()