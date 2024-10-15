from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        None: Updates the derivative values of each leaf through accumulate_derivative`.

    """
    args = list(vals)
    args[arg] += epsilon
    f_plus = f(*args)
    args[arg] -= 2 * epsilon
    f_minus = f(*args)
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative for this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Return the unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf node."""
        ...

    def is_constant(self) -> bool:
        """Check if the variable is a constant node."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get the parent nodes of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to get the derivatives for the input variables."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    sorted_nodes = []
    visited = set()

    def visit(node: Variable) -> None:
        if id(node) not in visited and not node.is_constant():
            visited.add(id(node))
            for parent in node.parents:
                visit(parent)
            sorted_nodes.append(node)

    visit(variable)
    return reversed(sorted_nodes)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv: The derivative we want to propagate backward to the leaves.

    Returns:
    -------
        None: Updates the derivative values of each leaf through `accumulate_derivative`.

    """
    # Topological sort to get the order of operations
    sorted_vars = topological_sort(variable)

    # Dictionary to hold accumulated derivatives
    derivatives = {variable.unique_id: deriv}

    # Backpropagate through each variable
    for v in sorted_vars:
        d_output = derivatives.get(v.unique_id, 0)  # Get current derivative

        # If it's a leaf node, accumulate the derivative
        if v.is_leaf():
            v.accumulate_derivative(d_output)
        else:
            # Chain rule to get derivatives for inputs
            for input_var, local_derivative in v.chain_rule(d_output):
                if input_var.unique_id not in derivatives:
                    derivatives[input_var.unique_id] = 0
                derivatives[input_var.unique_id] += local_derivative


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Get the saved tensors from the context."""
        return self.saved_values
