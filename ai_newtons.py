import sympy as sp





def newton_optimize(

    f: sp.Expr,

    x: sp.Symbol,

    x_start: float = 3.0,

    tolerance: float = 1e-6,

    max_iters: int = 100,

) -> float | None:

    """Finds a stationary point of a scalar function using Newton's optimization method.



    Computes the first and second symbolic derivatives of `f` with respect to `x`,

    and iteratively applies the update rule:

        x_{k+1} = x_k - f'(x_k) / f''(x_k)



    Args:

        f (sp.Expr): The symbolic objective function to optimize.

        x (sp.Symbol): The symbolic variable to optimize with respect to.

        x_start (float, optional): Initial starting value for x. Defaults to 3.0.

        tolerance (float, optional): Absolute threshold for convergence check

            |x_next - x_curr|. Defaults to 1e-6.

        max_iters (int, optional): Maximum allowed iterations. Defaults to 100.



    Returns:

        float | None: The value of x at convergence, or None if numerical issues

            (division by zero) occur or max_iters is reached without converging.

    """

    # 1. Compute first and second derivatives ONCE outside the loop

    f1 = sp.diff(f, x)  # f'(x)

    f2 = sp.diff(f1, x)  # f''(x)



    print(f"Function: {f}")

    print(f"First derivative: {f1}")

    print(f"Second derivative: {f2}")



    x_curr = float(x_start)



    # 2. Optimization loop

    for step in range(1, max_iters + 1):

        # Evaluate derivatives at the current point

        df1 = float(f1.subs(x, x_curr))

        df2 = float(f2.subs(x, x_curr))



        if df2 == 0:

            print(

                "Second derivative is zero. Stopping to avoid division by zero."

            )

            return None



        # Newton's optimization update: x_next = x_curr - f'(x) / f''(x)

        x_next = x_curr - (df1 / df2)



        print(f"Step {step}: x = {x_next:.6f}")



        # Check for convergence

        if abs(x_next - x_curr) <= tolerance:

            print(f"Converged to optimal x = {x_next:.6f} in {step} steps.")

            return x_next



        x_curr = x_next



    print("Reached max iterations without meeting tolerance criteria.")

    return x_curr





# Usage

x = sp.symbols("x")

f = x**3 - 3 * x



optimal_x = newton_optimize(f=f, x=x, x_start=3.0)