"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, List, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two floating point numbers.

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        float: The product of x and y

    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: The same number x

    """
    return x


def add(x: float, y: float) -> float:
    """Add two floating point numbers.

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        float: The sum of x and y

    """
    return x + y


def neg(x: float) -> float:
    """Negate a floating point number.

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: The negation of x

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Check if x is less than y.

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        bool: True if x < y, False otherwise

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if x is equal to y.

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        bool: True if x == y, False otherwise

    """
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of two floating point numbers.

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        float: The larger of x and y

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two floating point numbers are close to each other.

    Args:
    ----
        x (float): First number
        y (float): Second number

    Returns:
    -------
        bool: True if |x - y| < 1e-2, False otherwise

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid function.

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: The sigmoid of x

    """
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU (Rectified Linear Unit) function.

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: x if x > 0, else 0

    """
    return x if x > 0 else 0


def log(x: float) -> float:
    """Compute the natural logarithm.

    Args:
    ----
        x (float): Input number (must be positive)

    Returns:
    -------
        float: The natural logarithm of x

    """
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function.

    Args:
    ----
        x (float): Input number

    Returns:
    -------
        float: e^x

    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the gradient of the natural logarithm.

    Args:
    ----
        x (float): Input number
        d (float): Gradient from the next layer

    Returns:
    -------
        float: The gradient of log(x) * d

    """
    return d / x


def inv(x: float) -> float:
    """Compute the inverse of a number.

    Args:
    ----
        x (float): Input number (must be non-zero)

    Returns:
    -------
        float: 1 / x

    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the gradient of the inverse function.

    Args:
    ----
        x (float): Input number
        d (float): Gradient from the next layer

    Returns:
    -------
        float: The gradient of (1/x) * d

    """
    return -d / x**2


def relu_back(x: float, d: float) -> float:
    """Compute the gradient of the ReLU function.

    Args:
    ----
        x (float): Input number
        d (float): Gradient from the next layer

    Returns:
    -------
        float: d if x > 0, else 0

    """
    return d * (x > 0)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable[[float], float], ls: List[float]) -> List[float]:
    """Apply a function to each element of a list.

    Args:
    ----
        f (Callable[[float], float]): Function to apply
        ls (List[float]): Input list

    Returns:
    -------
        List[float]: A new list with the function applied to each element

    """
    return [f(x) for x in ls]


def zipWith(
    f: Callable[[float, float], float], ls1: List[float], ls2: List[float]
) -> List[float]:
    """Higher-order function that combines elements from two iterables using a given func

    Args:
    ----
        f (Callable[[float, float], float]): Function to apply
        ls1 (List[float]): First input list
        ls2 (List[float]): Second input list

    Returns:
    -------
        List[float]: A new list with the function applied to each pair of elements

    """
    return [f(x, y) for x, y in zip(ls1, ls2)]


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce.

    "Args":
        fn: combine two values
        start: start value $x_0$

    "Returns":
         Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    def apply(ls: Iterable[float]) -> float:
        my_list = list(ls).copy()

        if len(my_list) == 0:
            return start
        else:
            current_value = my_list.pop()
            return fn(current_value, apply(my_list))

    return apply


def negList(ls: List[float]) -> List[float]:
    """Negate each element of a list with map.

    Args:
    ----
        ls (List[float]): Input list

    Returns:
    -------
        List[float]: A new list with each element negated

    """
    return map(neg, ls)


def addLists(ls1: List[float], ls2: List[float]) -> List[float]:
    """Add two lists together with zipWith.

    Args:
    ----
        ls1 (List[float]): First input list
        ls2 (List[float]): Second input list

    Returns:
    -------
        List[float]: A new list with each element of ls1 and ls2 added together

    """
    return zipWith(add, ls1, ls2)


def sum(ls: List[float]) -> float:
    """Sum a list with reduce.

    Args:
    ----
        ls (List[float]): Input list

    Returns:
    -------
        float: The sum of the elements in ls

    """
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul`."""
    return reduce(mul, 1.0)(ls)
