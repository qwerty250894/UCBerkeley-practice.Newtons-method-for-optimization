import pytest
import sympy as sp

# This imports your function from the other file you created
from ai_newtons import newton_optimize

def test_finds_minimum_of_simple_parabola():
    """Testing a simple bowl shape: f(x) = x^2."""
    x = sp.symbols("x")
    f = x**2
    
    # We start at x=2.0. The lowest point of x^2 is at x=0.
    result = newton_optimize(f, x, x_start=2.0)
    
    # "assert" is just us telling the computer: "I expect this to be true!"
    # We check if the result is super close to 0.
    assert abs(result - 0.0) < 1e-5

def test_cubic_function_example():
    """Testing the example you provided in your code."""
    x = sp.symbols("x")
    f = x**3 - 3 * x
    
    # Starting at 3.0, it should slide down the curve and settle exactly at 1.0.
    result = newton_optimize(f, x, x_start=3.0)
    
    assert abs(result - 1.0) < 1e-5

def test_division_by_zero_safety():
    """Testing what happens if the function is a straight line."""
    x = sp.symbols("x")
    f = x # A straight line. The second derivative of this is 0.
    
    result = newton_optimize(f, x, x_start=1.0)
    
    # Because the second derivative is 0, your code should safely stop 
    # and return None to prevent a math error.
    assert result is None