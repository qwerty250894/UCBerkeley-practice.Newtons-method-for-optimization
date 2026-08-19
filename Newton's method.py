import sympy as sp

# 1. Define symbol and objective function
x = sp.symbols("x")
f = x**3 - 3 * x  # Function with a local minimum at x = 1

# 2. Compute first and second derivatives ONCE outside the loop
f1 = sp.diff(f, x)  # f'(x)
f2 = sp.diff(f1, x)  # f''(x)

print(f"Function: {f}")
print(f"First derivative: {f1}")
print(f"Second derivative: {f2}")

# 3. Initialize parameters
x_curr = 3.0  # Starting guess
tolerance = 1e-6
max_iters = 100

# 4. Optimization loop
""" Optimization loop that solves for Newton's optimization"""

for step in range(1, max_iters + 1):
    # Evaluate derivatives at the current point
    df1 = float(f1.subs(x, x_curr))
    df2 = float(f2.subs(x, x_curr))

    if df2 == 0:
        print("Second derivative is zero. Stopping to avoid division by zero.")
        break

    # Newton's optimization update: x_next = x_curr - f'(x) / f''(x)
    x_next = x_curr - (df1 / df2)

    print(f"Step {step}: x = {x_next:.6f}")

    # Check for convergence
    if abs(x_next - x_curr) <= tolerance:
        print(f"Converged to optimal x = {x_next:.6f} in {step} steps.")
        break

    x_curr = x_next
