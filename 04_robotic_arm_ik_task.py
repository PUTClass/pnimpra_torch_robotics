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


# ==============================================================================
# Initial conditions - joint angles and link lengths
# ==============================================================================

q = torch.tensor(
    [torch.pi/6, torch.pi/4, torch.pi/9], requires_grad=True)

L = torch.tensor(
    [1.3, 0.2, 0.3]
)

# ==============================================================================
# Optimization TODO
# ==============================================================================

TARGET_POS = torch.tensor([1.0, 0.0])
EPOCHS = 10000
optimizer = torch.optim.SGD([q], lr=0.1)

for e in range(EPOCHS):
    optimizer.zero_grad()
    T, joint_pos = forward_kinematics(q, L)
    
    # 1. Target reaching loss
    loss_target = ((joint_pos[-1] - TARGET_POS)**2).sum()
    
    # Total loss with a weighting factor (hyperparameter) for the smoothness terms
    loss = loss_target
    
    loss.backward()
    optimizer.step()

    if e % 100 == 0:
        print(f"Epoch {e} | Total: {loss.item():.4f}")

# ==============================================================================
# VISUALIZATION
# ==============================================================================
_, final_pos = forward_kinematics(q, L)

robot_base_pos = np.zeros(shape=(1, 2))
robot_links_pos = final_pos.detach().numpy()
robot_links_pos = np.concat((robot_base_pos, robot_links_pos), axis=0)
robot_target_pos = TARGET_POS.numpy()

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter([0], [0], color='black', s=100, zorder=5, label='Robot Base')

ax.scatter([robot_target_pos[0]], [robot_target_pos[1]], 
           alpha=0.5, color='blue', s=100, zorder=5, label='Robot Target Position')

arm_line, = ax.plot(robot_links_pos[:, 0], robot_links_pos[:, 1], 
                    'o-', lw=4, color='orange', alpha=1.0, label='Robot Arm')

ax.set_xlim(-2.5, 3.0)
ax.set_ylim(-1.0, 3.0)
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Robotic arm visualization')
ax.legend()
ax.grid()

plt.show()