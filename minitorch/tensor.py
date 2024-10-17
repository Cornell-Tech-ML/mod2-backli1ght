"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Set the tensor to require gradients if `x` is True."""
        self.history = History()

    def requires_grad(self) -> bool:
        """Check if the tensor requires gradients."""
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    # Properties
    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Returns
        int : size of the tensor

        """
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns
        int : dimensionality of the tensor

        """
        return self._tensor.dims

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    # Functions
    def __add__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    def __lt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:  # type: ignore[override]
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        return self + b

    def __rmul__(self, b: TensorLike) -> Tensor:
        return self * b

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Compute the all operation over the specified dimension.

        Args:
        ----
            dim (Optional[int]): The dimension to perform the operation on.
                                 If None, the operation is applied to the entire tensor.

        "Returns":
            Tensor: The result of the all operation.

        """
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, y: Tensor) -> Tensor:
        """Check if the current tensor is close to another tensor.

        "Args":
            y (Tensor): The tensor to compare against.

        "Returns":
            Tensor: A tensor indicating element-wise closeness.
        """
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        """Apply the sigmoid activation function to the tensor.

        "Returns":
            Tensor: A tensor with the sigmoid function applied element-wise.
        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Apply the ReLU (Rectified Linear Unit) activation function to the tensor.

        "Returns":
            Tensor: A tensor with the ReLU function applied element-wise.
        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Apply the natural logarithm to the tensor.

        "Returns":
            Tensor: A tensor with the natural logarithm applied element-wise.
        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Apply the exponential function to the tensor.

        "Returns":
            Tensor: A tensor with the exponential function applied element-wise.
        """
        return Exp.apply(self)

    def item(self) -> float:
        """Get the value of a single-element tensor as a Python float.

        "Returns":
            float: The value of the tensor.

        "Raises":
            AssertionError: If the tensor does not contain exactly one element.
        """
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Compute the sum over dimension `dim`"""
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Compute the mean over dimension `dim`"""
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum() / self.size

    def permute(self, *order: int) -> Tensor:
        """Permute tensor dimensions to *order"""
        return Permute.apply(self, tensor(list(order)))

    def view(self, *shape: int) -> Tensor:
        """Change the shape of the tensor to a new shape with the same size"""
        return View.apply(self, tensor(list(shape)))

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`

        "Parameters"
        ----------
            other : Tensor
                The tensor to be expanded for broadcasting.

        "Returns"
        -------
            Tensor: Expanded version of `other` with the right derivatives.
        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Create a tensor filled with zeros.

        "Args":
            shape (Optional[UserShape]): The shape of the tensor to create.
                                          If None, the shape of the current tensor is used.

        "Returns":
            Tensor: A tensor filled with zeros of the specified shape.
        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return the underlying storage, shape, and strides of the tensor.

        "Returns":
            Tuple[Storage, Shape, Strides]: A tuple containing the storage, shape, and strides of the tensor.
        """
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Create a new tensor that shares the same data but does not require gradients.

        "Returns":
            Tensor: A detached tensor that does not track gradients.
        """
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the tensor is constant.

        "Returns":
            bool: True if the tensor does not require gradients (i.e., has no history).
        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Get the parent variables of this tensor.

        "Returns":
            Iterable[Variable]: An iterable of parent variables that contributed to this tensor.
        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute gradients for parent variables.

        "Args":
            d_output (Any): The gradient of the output with respect to this tensor.

        "Returns":
            Iterable[Tuple[Variable, Any]]: A list of tuples containing parent variables and their corresponding gradients.
        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Compute the backward pass for this tensor.

        "Args":
            grad_output (Optional[Tensor]): The gradient of the output with respect to this tensor.
                                             If None, a default gradient of 1.0 is used for scalar tensors.

        "Raises":
            AssertionError: If the tensor is non-scalar and grad_output is not provided.
        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def zero_grad_(self) -> None:  # pragma: no cover
        """Reset the derivative on this variable."""
        self.grad = None
