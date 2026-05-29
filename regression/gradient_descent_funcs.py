"""Reusable gradient-descent helpers for simple linear regression (y = m*x + b).

Imported by ``gradient_descent_heights.py``. Implements the gradient of the
mean-squared error with respect to the intercept (b) and slope (m) and runs the
descent loop.
"""


def get_gradient_at_b(x, y, b, m):
    """Partial derivative of the MSE loss with respect to the intercept b."""
    N = len(x)
    diff = 0
    for i in range(N):
        diff += (y[i] - ((m * x[i]) + b))
    return -(2 / N) * diff


def get_gradient_at_m(x, y, b, m):
    """Partial derivative of the MSE loss with respect to the slope m."""
    N = len(x)
    diff = 0
    for i in range(N):
        diff += x[i] * (y[i] - ((m * x[i]) + b))
    return -(2 / N) * diff


def step_gradient(b_current, m_current, x, y, learning_rate):
    """Take a single gradient-descent step and return the updated [b, m]."""
    b_gradient = get_gradient_at_b(x, y, b_current, m_current)
    m_gradient = get_gradient_at_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]


def gradient_descent(x, y, learning_rate=0.0001, num_iterations=1000):
    """Run gradient descent from b=m=0 and return the fitted [b, m]."""
    b = 0
    m = 0
    for _ in range(num_iterations):
        b, m = step_gradient(b, m, x, y, learning_rate)
    return [b, m]
