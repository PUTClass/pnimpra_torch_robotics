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


def batched_rotation(theta):
    """
    Implement the batched version of the `rotation` function above.
    theta: [T, K] angles in radians across T timesteps and K joints
    returns:
        R: [T, K, 2, 2] Rotation matrices 
    """
    T, K = theta.shape
    R = torch.empty(T, K, 2, 2)

    c = torch.cos(theta)
    s = torch.sin(theta)

    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c

    return R

def batched_forward_kinematics(q, L):
    """
    Implement the batched version of the `forward_kinematics` function.
    q: [T, K] joint angles
    L: [K] link lengths
    returns:
        joint_pos: [T, K, 2] Positions of all joints over all timesteps
    """
    # Cumulative sum along the joints axis (dim=1)
    q_abs = torch.cumsum(q, dim=1)

    # Batched rotation matrices
    R = batched_rotation(q_abs)
    
    # Local link vectors [K, 2]
    p = torch.stack([L, torch.zeros_like(L)], dim=1)
    
    # Reshape p to broadcast over T axis and perform batched matmul
    # R is [T, K, 2, 2], p.view is [1, K, 2, 1]
    p_world = (R @ p.view(1, -1, 2, 1)).squeeze(-1)
    
    # Cumsum along the joints axis to add up world link vectors
    joint_pos = torch.cumsum(p_world, dim=1)

    return joint_pos

# Link lengths for a 3-DOF robot
L = torch.tensor([0.7, 0.5, 0.3])
T_steps = 100

# ---------------------------------------------------------
# A set of joint angles
# ---------------------------------------------------------
t = torch.linspace(0, 1, T_steps)
q1 = torch.linspace(0, torch.pi/2, T_steps)
q2 = torch.linspace(0, torch.pi/2, T_steps)
q3 = (torch.pi/4) * torch.sin(2 * torch.pi * t) # Varies in [-pi/4, pi/4]

q_batched = torch.stack([q1, q2, q3], dim=1)

# ---------------------------------------------------------
# Compute batched kinematics
# --------------------------------------------------------- 
joint_pos_batched = batched_forward_kinematics(q_batched, L)

# ---------------------------------------------------------
# Calculate the total distance traveled by the end effector
# ---------------------------------------------------------
end_effector_pos = joint_pos_batched[:, -1, :] # Shape [T, 2]

# Calculate the sum of Euclidean distances between consecutive steps:
diffs = end_effector_pos[1:] - end_effector_pos[:-1] # Differences between frames [T-1, 2]
distances = torch.norm(diffs, dim=1)                 # Euclidean norm of differences [T-1]
total_distance = distances.sum().item()
print(f"Total distance traveled by the end effector: {total_distance:.4f}")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

fig, ax = plt.subplots(figsize=(8, 8))

# Plot the trace of the traveled path
trace_np = end_effector_pos.numpy()
ax.plot(trace_np[:, 0], trace_np[:, 1], 'g--', lw=2, alpha=0.6, label='End Effector Trace')

# Draw the robot in its final state
final_pos = joint_pos_batched[-1].numpy()
robot_base_pos = np.zeros(shape=(1, 2))
robot_links_pos = np.concat((robot_base_pos, final_pos), axis=0)

ax.scatter([0], [0], color='black', s=100, zorder=5, label='Robot Base')
arm_line, = ax.plot(robot_links_pos[:, 0], robot_links_pos[:, 1], 
                    'o-', lw=4, color='orange', alpha=0.8, label='Robot Arm (Final Pose)')

ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-1.0, 2.0)
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Batched Robotic Arm Trajectory Task')
ax.legend()
ax.grid()

plt.show()