from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union


from numba import cuda

import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
    ----
        index : index tuple of ints
        strides : tensor strides

    Returns:
    -------
        Position in storage

    """
    pos = 0
    l = len(index)
    for i in range(l):
        pos = pos + index[i] * strides[i]
    return pos


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
    ----
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    ordinal = ordinal + 0
    for d in range(len(shape) - 1, -1, -1):
        out_index[d] = ordinal % shape[d]
        ordinal = ordinal // shape[d]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
    ----
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
    -------
        None

    """
    lb = len(big_shape)
    l = len(shape)
    for i in range(l):
        ib = i + lb - l
        index = big_index[ib] if shape[i] != 1 else 0
        out_index[i] = index


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
    ----
        shape1 : first shape
        shape2 : second shape

    Returns:
    -------
        broadcasted shape

    Raises:
    ------
        IndexingError : if cannot broadcast

    """
    resShape = []
    l1 = len(shape1)
    l2 = len(shape2)
    l = max(l1, l2)
    for i in range(l):
        di1 = shape1[l1 - 1 - i] if i < l1 else 0
        di2 = shape2[l2 - 1 - i] if i < l2 else 0
        if di1 > 1 and di2 > 1 and di1 != di2:
            raise IndexingError(f"shape1: {shape1} must match shape2: {shape2}.")
        resShape.insert(0, max(di1, di2))
    return tuple(resShape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return a contiguous stride for a given shape."""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert the tensor storage to a CUDA array if it is not already."""
        if not cuda.is_cuda_array(self._storage):
            self._storage = cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns
        -------
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Broadcast two shapes to create a new union shape.

        "Args":
            shape_a: First shape to broadcast.
            shape_b: Second shape to broadcast.

        "Returns":
            UserShape: The broadcasted shape resulting from combining shape_a and shape_b.

        "Raises":
            IndexingError: If the shapes cannot be broadcast together.
        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Retrieve the position in storage for a given index.

        "Args":
            index: An integer or a sequence of integers representing the index.

        "Returns":
            int: The position in storage corresponding to the provided index.

        "Raises":
            IndexingError: If the index is out of range or has an incorrect size.
        """
        # if isinstance(index, int):
        #     aindex: Index = array([index])
        # if isinstance(index, tuple):
        #     aindex = array(index)
        if isinstance(index, int):
            aindex = np.array([index], dtype=np.int32)
        else:
            aindex = np.array(index, dtype=np.int32)
        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Yields all valid indices for the tensor.

        This method generates all possible valid indices for the tensor based on its shape.
        It iterates over each position in the tensor,
        converts the linear index to a multi-dimensional index,
        and yields the index as a tuple.

        "Yields":
            UserIndex: A tuple representing a valid index for the tensor.
        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Get a random valid index.

        "Returns":
            UserIndex: A randomly generated valid index for the tensor.
        """
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Retrieve the value at the specified index in the tensor.

        "Args":
            key: A sequence of integers representing the index.

        "Returns":
            float: The value at the specified index in the tensor.
        """
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Set the value at the specified index in the tensor.

        "Args":
            key: A sequence of integers representing the index.
            val: The value to set at the specified index.

        "Returns":
            None
        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return core tensor data as a tuple.

        "Returns":
            Tuple[Storage, Shape, Strides]: A tuple containing the storage, shape, and strides of the tensor.
        """
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
        ----
            *order: a permutation of the dimensions

        Returns:
        -------
            New `TensorData` with the same storage and a new dimension order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        newShape = tuple([self.shape[i] for i in order])
        newStride = tuple([self.strides[i] for i in order])
        return TensorData(self._storage, newShape, newStride)

    def to_string(self) -> str:
        """Convert the tensor data to a string representation.

        "Returns":
            str: A string representation of the tensor data.
        """
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
