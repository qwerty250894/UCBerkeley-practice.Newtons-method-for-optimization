import sympy as sp
import numpy as np
#hihihihihihihiih Nice work
def newton_optimize_multi(
    f: sp.Expr,
    vars_list: list[sp.Symbol],
    x_start: list[float],
    tolerance: float = 1e-6,
    max_iters: int = 100,
) -> np.ndarray | None:
    """Finds a stationary point of a multivariate function using Newton's method."""
    
    # 1. Compute symbolic Gradient (vector) and Hessian (matrix) ONCE
    grad = sp.Matrix([sp.diff(f, v) for v in vars_list])
    hess = sp.hessian(f, vars_list)

    print(f"Function: {f}")
    print(f"Gradient: {grad}")
    print(f"Hessian:\n{hess}")

    x_curr = np.array(x_start, dtype=float)

    # 2. Optimization loop
    for step in range(1, max_iters + 1):
        # Map current values to their respective symbolic variables
        subs_dict = dict(zip(vars_list, x_curr))

        # Evaluate Gradient and Hessian at the current point
        grad_val = np.array(grad.subs(subs_dict)).astype(float).flatten()
        hess_val = np.array(hess.subs(subs_dict)).astype(float)

        # Solve H * delta = grad to avoid explicitly inverting the Hessian
        try:
            step_direction = np.linalg.solve(hess_val, grad_val)
        except np.linalg.LinAlgError:
            print("Hessian is singular (non-invertible). Stopping.")
            return None

        # Newton's update: x_next = x_curr - H^-1 * grad
        x_next = x_curr - step_direction

        print(f"Step {step}: x = {np.round(x_next, 6)}")

        # Check for convergence using the Euclidean norm of the step size
        if np.linalg.norm(x_next - x_curr) <= tolerance:
            print(f"Converged to optimal x = {np.round(x_next, 6)} in {step} steps.")
            return x_next

        x_curr = x_next

    print("Reached max iterations without meeting tolerance criteria.")
    return x_curr


# --- Usage Example ---
x, y = sp.symbols("x y")
# A 2D function: f(x,y) = x^2 + 2y^2 + xy - 4x - 5y
f_multi = x**2 + 2*y**2 + x*y - 4*x - 5*y

optimal_point = newton_optimize_multi(
    f=f_multi, 
    vars_list=[x, y], 
    x_start=[0.0, 0.0]
)