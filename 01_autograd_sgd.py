import torch
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# We want to find the minimum (lowest point) of this function.
# f(x) = (x - 3)^2 + 4 * sin(2 * x)
# ---------------------------------------------------------
def my_function(x):
    return (x - 3)**2 + 4 * torch.sin(2 * x)

# ---------------------------------------------------------
# Choose a starting guess for x. 
# requires_grad=True is used to track gradients!
x = torch.tensor([-4.0], requires_grad=True)

# Use an optimizer (Gradient Descent) to automatically update x
# lr is the "learning rate" (how big of a step it takes each time)
optimizer = torch.optim.SGD([x], lr=0.10)

# Keep track of the values for plotting
x_history = []
y_history = []

# ---------------------------------------------------------
# Optimization Loop
# ---------------------------------------------------------
epochs = 50
for step in range(epochs):
    # Clear old gradients from the previous step
    optimizer.zero_grad()
    
    # Calculate the function value (this acts as our "Loss")
    y = my_function(x)
    
    # Store the current position in our history lists
    # .item() extracts the standard Python number from the PyTorch Tensor
    x_history.append(x.item())
    y_history.append(y.item())
    
    # PyTorch calculates the slope (derivative dy/dx) at our current point x
    y.backward()
    
    # Take a step down the slope to find the minimum
    optimizer.step()

print(f"Final x: {x.item():.4f}")
print(f"Final minimum value y: {y.item():.4f}")

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
x_vals_np = np.linspace(-4, 8, 200)
x_vals_torch = torch.tensor(x_vals_np)
y_vals_np = my_function(x_vals_torch).numpy()

plt.figure(figsize=(10, 6))

plt.plot(x_vals_np, y_vals_np, 'k-', lw=2, label="Function: y = (x-3)² + 4*sin(2x)")

plt.plot(x_history, y_history, 'ro-', lw=2, alpha=0.6, markersize=6, label="Optimizer Path")

plt.plot(x_history[0], y_history[0], 'go', markersize=10, label="Start")
plt.plot(x_history[-1], y_history[-1], 'go', marker='*', markersize=14, label="Found Minimum")

plt.title("Finding a Function Minimum using PyTorch Autograd")
plt.xlabel("x")
plt.ylabel("f(x) value")
plt.legend()
plt.grid(True)
plt.show()
