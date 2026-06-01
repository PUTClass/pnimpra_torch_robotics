import torch

print("=== 1. Creating and Initializing Tensors ===")

scalar = torch.tensor(3.14)
joint_angles = torch.tensor([0.5, 1.2])
zeros = torch.zeros((2, 3))
ones = torch.ones((3, 2))
identity = torch.eye(3)        # 3x3 identity matrix
random_mat = torch.rand((2, 2)) # Uniform [0, 1]

print(f"Scalar: {scalar}, Shape: {scalar.shape}")
print(f"Vector (1D): {joint_angles}, Shape: {joint_angles.shape}")
print("Identity Matrix:\n", identity)
print("Random Matrix:\n", random_mat)

print("\n=== 2. Matrix Multiplication ===")
# Define a rotation matrix (90 degrees)
theta = torch.tensor([torch.pi / 2])
R = torch.tensor([
    [torch.cos(theta), -torch.sin(theta)],
    [torch.sin(theta),  torch.cos(theta)]
])
# 1D vector
v = torch.tensor([1.0, 0.0])

# Using @ operator (equivalent to torch.matmul)
v_rotated = torch.matmul(R, v) # or R @ v
print("Original v:", v)
print("Rotated v (R @ v):", v_rotated)

print("\n=== 3. Broadcasting ===")
# Adding a smaller shape to a larger shape
batch_of_points = torch.tensor([
    [1.0, 1.0],
    [2.0, 2.0],
    [3.0, 3.0]
]) # Shape [3, 2]

translation = torch.tensor([0.5, -0.5]) # Shape [2]
# PyTorch automatically expands translation to [3, 2] to add them!
translated_points = batch_of_points + translation
print("Batch of points:\n", batch_of_points)
print("Translated by [0.5, -0.5]:\n", translated_points)

print("\n=== 4. Stacking and Concatenating ===")
t1 = torch.tensor([1, 2])
t2 = torch.tensor([3, 4])
t3 = torch.tensor([5, 6])

# Stack creates a *new* dimension
stacked_dim0 = torch.stack([t1, t2, t3], dim=0) # Shape: [3, 2]
stacked_dim1 = torch.stack([t1, t2, t3], dim=1) # Shape: [2, 3]
print("Stack along dim 0:\n", stacked_dim0)
print("Stack along dim 1:\n", stacked_dim1)

# Concat appends along an *existing* dimension
cat_vector = torch.cat([t1, t2, t3], dim=0) # Shape: [6]
print("Concatenated vector:\n", cat_vector)

print("\n=== 5. Reshaping (View vs. Reshape) ===")
flat_data = torch.arange(1, 13) # [1, 2, ..., 12]
print("Flat data shape:", flat_data.shape)

# View reorganizes the data without copying memory
matrix_3x4 = flat_data.view(3, 4)
print("Viewed as 3x4 matrix:\n", matrix_3x4)

# -1 tells PyTorch to "figure out the missing dimension"
matrix_4xN = flat_data.view(4, -1) 
print("Viewed as 4x-1 (becomes 4x3):\n", matrix_4xN)

print("\n=== 6. Unsqueeze and Squeeze ===")
# Adding a dummy dimension (very common for batching!)
vector_1d = torch.tensor([1, 2, 3])      # Shape: [3]
vector_col = vector_1d.unsqueeze(1)      # Shape: [3, 1]
vector_row = vector_1d.unsqueeze(0)      # Shape: [1, 3]

print("1D Vector shape:", vector_1d.shape)
print("Unsqueeze(1) (Column) shape:", vector_col.shape, "->\n", vector_col)
print("Unsqueeze(0) (Row) shape:", vector_row.shape, "->\n", vector_row)

# Squeeze removes dimensions of size 1
squeezed_col = vector_col.squeeze(1)     # Back to Shape: [3]
print("Squeeze(1) shape:", squeezed_col.shape)

print("\n=== 7. Batched Matrix-Vector Multiplication ===")
# Batch of 5 rotation matrices, shape: [5, 2, 2]
batch_R = torch.stack([torch.eye(2) * (i+1) for i in range(5)], dim=0)

# Batch of 5 vectors, shape: [5, 2]
batch_v = torch.ones((5, 2))

# We can't do [5, 2, 2] @ [5, 2] directly. 
# We need to turn the vector into a column vector [5, 2, 1]
batch_v_col = batch_v.unsqueeze(2)

# Now [5, 2, 2] @ [5, 2, 1] = [5, 2, 1]
batch_result = batch_R @ batch_v_col

# We squeeze back to [5, 2]
final_result = batch_result.squeeze(2)

print("Batch R shape:", batch_R.shape)
print("Final Result shape:", final_result.shape)
print("Final Result Data:\n", final_result)