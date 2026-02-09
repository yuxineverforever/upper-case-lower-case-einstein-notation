"""
EinSQL: Einstein Summation for SQL

A library for optimizing tensor contractions using Einstein notation
and generating PostgreSQL queries for database-driven tensor computations.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import prod
from typing import Any

import numpy as np
import opt_einsum as oe
import torch


# =============================================================================
# Cost Model Constants
# =============================================================================

# Communication cost multiplier for data transfer operations
COMM_COST_MULTIPLIER = 1

# Cost multiplier for relational operations (joins, aggregations)
RELOP_COST_MULTIPLIER = 15


# =============================================================================
# Utility Functions
# =============================================================================

def index_to_subscript(i: int) -> str:
    """Convert integer index to subscript character (i, j, k, ...)."""
    if ord("i") + i > ord("z"):
        raise ValueError(f"Index {i} exceeds maximum subscript range (i-z)")
    return chr(ord("i") + i)


def find_all_factors(n: int) -> list[int]:
    """Find all factors of a positive integer."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    factors = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return sorted(factors)


def get_label_dimensions(einsum_string: str, shapes: list[tuple]) -> dict[str, int]:
    """
    Extract dimension sizes for each label in an einsum expression.

    Args:
        einsum_string: Einstein summation notation (e.g., "ij,jk->ik")
        shapes: List of tensor shapes

    Returns:
        Dictionary mapping labels to their dimensions
    """
    label_dims = {}
    input_labels = einsum_string.split("->")[0].split(",")

    for i, subscript_labels in enumerate(input_labels):
        for label, dimension in zip(list(subscript_labels), shapes[i]):
            if label not in label_dims:
                label_dims[label] = dimension
            elif label_dims[label] != dimension:
                raise ValueError(
                    f"Dimension mismatch for label '{label}': "
                    f"expected {label_dims[label]}, got {dimension}"
                )
    return label_dims


# =============================================================================
# Expression AST Classes
# =============================================================================

class Expr(ABC):
    """Abstract base class for all expressions."""

    @abstractmethod
    def flops(self) -> float:
        """Return the number of floating point operations."""
        pass


class UnaryOp(Expr):
    """Unary operation expression."""

    def __init__(self, opcode: str, expr: Expr):
        self.opcode = opcode
        self.expr = expr

    def flops(self) -> float:
        return 1.0 + self.expr.flops()


class BinaryOp(Expr):
    """Binary operation expression."""

    def __init__(self, opcode: str, lhs: Expr, rhs: Expr):
        self.opcode = opcode
        self.lhs = lhs
        self.rhs = rhs

    def flops(self) -> float:
        return 1.0 + self.lhs.flops() + self.rhs.flops()


class Constant(Expr):
    """Constant value expression."""

    def __init__(self, value: float):
        self.value = value

    def flops(self) -> float:
        return 0.0


# =============================================================================
# Tiling Scheme
# =============================================================================

@dataclass
class TilingScheme:
    """
    Represents a tiling configuration for a tensor.

    Tiling divides a tensor into smaller blocks for efficient processing.
    Each scheme tracks the cost metrics for its configuration.
    """
    node: Any
    shape: tuple[int, ...]
    tile_shape: tuple[int, ...]

    # Computed fields
    tile_size: int = field(init=False)
    value_count: tuple[int, ...] = field(init=False)
    num_tuples: int = field(init=False)

    # Cost metrics
    cost: float = 0.0
    comm: float = 0.0
    flops: float = 0.0
    accumulated_cost: float = 0.0
    accumulated_comm: float = 0.0
    accumulated_flops: float = 0.0

    # Dependency tracking
    source: tuple = field(default_factory=tuple)
    dependencies: set = field(default_factory=set)

    def __post_init__(self):
        self.tile_size = prod(self.tile_shape)
        self.value_count = tuple(
            length // tile_length
            for length, tile_length in zip(self.shape, self.tile_shape)
        )
        self.num_tuples = prod(self.value_count)

    def __hash__(self):
        return hash((id(self.node), self.tile_shape))


# =============================================================================
# Code Generation: PostgreSQL Kernels
# =============================================================================

class KernelAdd:
    """Generates C code for element-wise addition kernel."""

    def __init__(self, name: str, ndim: int):
        self.name = name
        self.ndim = ndim

    def codegen(self) -> str:
        subscripts = [index_to_subscript(i) for i in range(self.ndim)]
        dim_vars = [f"n{s}" for s in subscripts]

        lines = [
            f"PG_FUNCTION_INFO_V1({self.name});",
            f"Datum {self.name}(PG_FUNCTION_ARGS) {{",
            "  ArrayType *mat_a = PG_GETARG_ARRAYTYPE_P(0);",
            "  ArrayType *mat_b = PG_GETARG_ARRAYTYPE_P(1);",
        ]

        # Dimension extraction
        for i, d in enumerate(dim_vars):
            lines.append(f"  int {d} = (int)ARR_DIMS(mat_a)[{i}];")

        lines.extend([
            "  double *a = (double *)ARR_DATA_PTR(mat_a);",
            "  double *b = (double *)ARR_DATA_PTR(mat_b);",
        ])

        # Output allocation
        out_size = " * ".join(dim_vars + ["sizeof(double)"])
        lines.extend([
            f"  double *c = (double *)palloc({out_size});",
            f"  memset(c, 0, {out_size});",
        ])

        # Nested loops
        for i, (s, d) in enumerate(zip(subscripts, dim_vars)):
            indent = "  " * (i + 1)
            lines.append(f"{indent}for (int {s} = 0; {s} < {d}; ++{s}) {{")

        # Computation
        index_expr = " + ".join(
            " * ".join([s] + dim_vars[j + 1:])
            for j, s in enumerate(subscripts)
        )
        inner_indent = "  " * (len(subscripts) + 1)
        lines.append(f"{inner_indent}c[{index_expr}] = a[{index_expr}] + b[{index_expr}];")

        # Close loops
        for i in range(len(subscripts) - 1, -1, -1):
            lines.append("  " * (i + 1) + "}")

        # Result construction
        lines.extend([
            "  ArrayType *result = NULL;",
            f"  int dims[{len(dim_vars)}] = {{{', '.join(dim_vars)}}};",
            f"  int lbs[{len(dim_vars)}] = {{{', '.join(['1'] * len(dim_vars))}}};",
            f"  result = construct_md_array((Datum *)c, NULL, {len(dim_vars)}, dims, lbs, FLOAT8OID,",
            "                            sizeof(float8), true, 'd');",
            "  pfree(c);",
            "  PG_RETURN_ARRAYTYPE_P(result);",
            "}",
            "",
        ])

        return "\n".join(lines)


class KernelMul:
    """Generates C code for tensor multiplication kernel."""

    def __init__(
        self,
        name: str,
        subscripts: list[str],
        lhs: list[str],
        rhs: list[str],
        out: list[str],
        is_complex: bool = False,
    ):
        """
        Initialize multiplication kernel.

        Example: For ij,jk->ik:
            subscripts = ['i', 'j', 'k']
            lhs = ['i', 'j']
            rhs = ['j', 'k']
            out = ['i', 'k']
        """
        self.name = name
        self.subscripts = subscripts
        self.lhs = lhs
        self.rhs = rhs
        self.out = out
        self.is_complex = is_complex

    def codegen(self) -> str:
        dim_vars = [f"n{s}" for s in self.subscripts]
        lhs_dim_vars = [f"n{s}" for s in self.lhs]
        rhs_dim_vars = [f"n{s}" for s in self.rhs]
        out_dim_vars = [f"n{s}" for s in self.out]

        lines = [
            f"PG_FUNCTION_INFO_V1({self.name});",
            f"Datum {self.name}(PG_FUNCTION_ARGS) {{",
            "  ArrayType *mat_a = PG_GETARG_ARRAYTYPE_P(0);",
            "  ArrayType *mat_b = PG_GETARG_ARRAYTYPE_P(1);",
        ]

        # Dimension extraction
        for s, d in zip(self.subscripts, dim_vars):
            if s in self.lhs:
                lines.append(f"  int {d} = (int)ARR_DIMS(mat_a)[{self.lhs.index(s)}];")
            else:
                lines.append(f"  int {d} = (int)ARR_DIMS(mat_b)[{self.rhs.index(s)}];")

        lines.extend([
            "  double *a = (double *)ARR_DATA_PTR(mat_a);",
            "  double *b = (double *)ARR_DATA_PTR(mat_b);",
        ])

        # Output allocation
        out_size = " * ".join(out_dim_vars)
        if self.is_complex:
            out_size += " * 2"
        out_size += " * sizeof(double)"

        lines.extend([
            f"  double *c = (double *)palloc({out_size});",
            f"  memset(c, 0, {out_size});",
        ])

        # Nested loops
        for i, (s, d) in enumerate(zip(self.subscripts, dim_vars)):
            indent = "  " * (i + 1)
            lines.append(f"{indent}for (int {s} = 0; {s} < {d}; ++{s}) {{")

        inner_indent = "  " * (len(self.subscripts) + 1)

        if not self.is_complex:
            lhs_index = " + ".join(
                " * ".join([s] + lhs_dim_vars[j + 1:])
                for j, s in enumerate(self.lhs)
            )
            rhs_index = " + ".join(
                " * ".join([s] + rhs_dim_vars[j + 1:])
                for j, s in enumerate(self.rhs)
            )
            out_index = " + ".join(
                " * ".join([s] + out_dim_vars[j + 1:])
                for j, s in enumerate(self.out)
            )
            lines.append(f"{inner_indent}c[{out_index}] += a[{lhs_index}] * b[{rhs_index}];")
        else:
            lines.extend(self._generate_complex_computation(
                inner_indent, lhs_dim_vars, rhs_dim_vars, out_dim_vars
            ))

        # Close loops
        for i in range(len(self.subscripts) - 1, -1, -1):
            lines.append("  " * (i + 1) + "}")

        # Result construction
        lines.append("  ArrayType *result = NULL;")

        if not self.is_complex:
            lines.extend([
                f"  int dims[{len(out_dim_vars)}] = {{{', '.join(out_dim_vars)}}};",
                f"  int lbs[{len(out_dim_vars)}] = {{{', '.join(['1'] * len(out_dim_vars))}}};",
                f"  result = construct_md_array((Datum *)c, NULL, {len(out_dim_vars)}, dims, lbs, FLOAT8OID,",
                "                            sizeof(float8), true, 'd');",
            ])
        else:
            ndims = len(out_dim_vars) + 1
            lines.extend([
                f"  int dims[{ndims}] = {{{', '.join(out_dim_vars + ['2'])}}};",
                f"  int lbs[{ndims}] = {{{', '.join(['1'] * ndims)}}};",
                f"  result = construct_md_array((Datum *)c, NULL, {ndims}, dims, lbs, FLOAT8OID,",
                "                            sizeof(float8), true, 'd');",
            ])

        lines.extend([
            "  pfree(c);",
            "  PG_RETURN_ARRAYTYPE_P(result);",
            "}",
            "",
        ])

        return "\n".join(lines)

    def _generate_complex_computation(
        self,
        indent: str,
        lhs_dim_vars: list[str],
        rhs_dim_vars: list[str],
        out_dim_vars: list[str],
    ) -> list[str]:
        """Generate complex number multiplication code."""
        def make_complex_index(subscripts: list[str], dim_vars: list[str], offset: int = 0) -> str:
            expr = " + ".join(
                " * ".join([s] + dim_vars[j + 1:] + ["2"])
                for j, s in enumerate(subscripts)
            )
            if offset:
                expr += f" + {offset}"
            return expr

        lhs_real = make_complex_index(self.lhs, lhs_dim_vars)
        lhs_imag = make_complex_index(self.lhs, lhs_dim_vars, 1)
        rhs_real = make_complex_index(self.rhs, rhs_dim_vars)
        rhs_imag = make_complex_index(self.rhs, rhs_dim_vars, 1)
        out_real = make_complex_index(self.out, out_dim_vars)
        out_imag = make_complex_index(self.out, out_dim_vars, 1)

        return [
            f"{indent}double ar = a[{lhs_real}];",
            f"{indent}double ai = a[{lhs_imag}];",
            f"{indent}double br = b[{rhs_real}];",
            f"{indent}double bi = b[{rhs_imag}];",
            f"{indent}c[{out_real}] += ar * br - ai * bi;",
            f"{indent}c[{out_imag}] += ar * bi + ai * br;",
        ]


# =============================================================================
# Indexed Term (for Einstein notation)
# =============================================================================

class IndexedTerm:
    """A tensor with index labels for Einstein notation."""

    def __init__(self, tensor: Tensor, indices: str):
        self.tensor = tensor
        self.indices = indices


# =============================================================================
# Tensor Classes
# =============================================================================

class Tensor(Expr):
    """
    Base class for tensors with tiling scheme support.

    A tensor tracks all possible tiling configurations and their
    associated cost metrics for optimization purposes.
    """

    def __init__(self, name: str, shape: tuple, is_complex: bool = False):
        self.name = name
        self.shape = shape
        self.is_complex = is_complex

        # Generate all possible tiling schemes
        factors = [find_all_factors(d) for d in shape]
        self.tile_shapes = list(itertools.product(*factors))
        self.schemes: dict[tuple, TilingScheme] = {
            tile_shape: TilingScheme(self, self.shape, tile_shape)
            for tile_shape in self.tile_shapes
        }

    def __add__(self, other: Tensor) -> BinaryOp:
        return BinaryOp("+", self, other)

    def __sub__(self, other: Tensor) -> BinaryOp:
        return BinaryOp("-", self, other)

    def __truediv__(self, other: Tensor) -> BinaryOp:
        return BinaryOp("/", self, other)

    def __getitem__(self, indices: str) -> IndexedTerm:
        return IndexedTerm(self, indices)

    def flops(self) -> float:
        return 0.0

    def gen_sql(self, sqlgen: SQLGenerator, tile_shape: tuple | None = None):
        """Generate SQL for this tensor (override in subclasses)."""
        pass


class DenseTensor(Tensor):
    """Dense tensor with PyTorch backend."""

    def __init__(self, name: str, data: torch.Tensor):
        super().__init__(name, data.shape)
        self.data = data
        self._compute_sparsity_metrics()

    def _compute_sparsity_metrics(self):
        """Compute sparsity information for each tiling scheme."""
        tuple_counter = {s: [set() for _ in range(len(self.shape) + 1)] for s in self.tile_shapes}
        self._traverse_elements(tuple_counter, self.data, [])

        for tile_shape, scheme in self.schemes.items():
            sets = tuple_counter[tile_shape]
            scheme.num_tuples = len(sets[-1])
            scheme.value_count = tuple(len(s) for s in sets[:-1])

    def _traverse_elements(
        self,
        tuple_counter: dict,
        submatrix: torch.Tensor,
        index: list[int],
    ):
        """Recursively traverse tensor elements to count non-zeros."""
        dim = len(index)

        if dim + 1 == len(self.shape):
            for i in range(self.shape[dim]):
                if submatrix[i] == 0:
                    continue

                index.append(i)
                for tile_shape, sets in tuple_counter.items():
                    outer_index = tuple(idx // ts for idx, ts in zip(index, tile_shape))
                    for d, idx in enumerate(outer_index):
                        sets[d].add(idx)
                    sets[-1].add(outer_index)
                index.pop()
            return

        for i in range(self.shape[dim]):
            index.append(i)
            self._traverse_elements(tuple_counter, submatrix[i], index)
            index.pop()

    def gen_sql(self, sqlgen: SQLGenerator, tile_shape: tuple | None = None):
        if tile_shape is None:
            tile_shape = self.shape

        # Drop existing table
        sqlgen.add_table(f"DROP TABLE IF EXISTS {self.name};")

        # Create table
        columns = [f"{index_to_subscript(i)} INTEGER" for i in range(len(self.shape))]
        columns.append(f"val DOUBLE PRECISION{'[]' * len(self.shape)}")
        sqlgen.add_table(f"CREATE TABLE {self.name} ({', '.join(columns)});")

        # Generate and insert values
        values = []
        self._gen_tile_values(values, tile_shape, self.data, [])

        if values:
            formatted_values = [
                f"({', '.join(str(i) for i in outer_index)}, ARRAY{val})"
                for outer_index, val in values
            ]
            sqlgen.add_table(f"INSERT INTO {self.name} VALUES {', '.join(formatted_values)};")

    def _gen_tile_values(
        self,
        results: list,
        tile_shape: tuple,
        tile: torch.Tensor,
        outer_index: list[int],
    ):
        """Generate tile values for SQL insertion."""
        dim = len(outer_index)

        for i in range(self.shape[dim] // tile_shape[dim]):
            outer_index.append(i)
            subtile = torch.narrow(tile, dim, i * tile_shape[dim], tile_shape[dim])

            if len(outer_index) == len(self.shape):
                if torch.count_nonzero(subtile) > 0:
                    array_str = self._tile_to_array_string(tile_shape, subtile, [])
                    results.append((tuple(outer_index), array_str))
            else:
                self._gen_tile_values(results, tile_shape, subtile, outer_index)

            outer_index.pop()

    def _tile_to_array_string(
        self,
        tile_shape: tuple,
        tile: torch.Tensor,
        inner_index: list[int],
    ) -> str:
        """Convert a tile to PostgreSQL array string format."""
        dim = len(inner_index)

        if dim + 1 == len(tile_shape):
            return f"[{', '.join(str(float(f)) for f in tile)}]"

        elements = []
        for i in range(tile_shape[dim]):
            inner_index.append(i)
            elements.append(self._tile_to_array_string(tile_shape, tile[i], inner_index))
            inner_index.pop()

        return f"[{', '.join(elements)}]"


class ComplexTensor(Tensor):
    """Complex-valued tensor with NumPy backend."""

    def __init__(self, name: str, data: np.ndarray):
        super().__init__(name, data.shape, is_complex=True)
        self.data = data
        self._compute_sparsity_metrics()

    def _compute_sparsity_metrics(self):
        """Compute sparsity information for each tiling scheme."""
        tuple_counter = {s: [set() for _ in range(len(self.shape) + 1)] for s in self.tile_shapes}
        self._traverse_elements(tuple_counter, self.data, [])

        for tile_shape, scheme in self.schemes.items():
            sets = tuple_counter[tile_shape]
            scheme.num_tuples = len(sets[-1])
            scheme.value_count = tuple(len(s) for s in sets[:-1])

    def _traverse_elements(
        self,
        tuple_counter: dict,
        submatrix: np.ndarray,
        index: list[int],
    ):
        """Recursively traverse tensor elements to count non-zeros."""
        dim = len(index)

        if dim + 1 == len(self.shape):
            for i in range(self.shape[dim]):
                if np.isclose(submatrix[i], [0.0, 0.0]).all():
                    continue

                index.append(i)
                for tile_shape, sets in tuple_counter.items():
                    outer_index = tuple(idx // ts for idx, ts in zip(index, tile_shape))
                    for d, idx in enumerate(outer_index):
                        sets[d].add(idx)
                    sets[-1].add(outer_index)
                index.pop()
            return

        for i in range(self.shape[dim]):
            index.append(i)
            self._traverse_elements(tuple_counter, submatrix[i], index)
            index.pop()

    def gen_sql(self, sqlgen: SQLGenerator, tile_shape: tuple | None = None):
        if tile_shape is None:
            tile_shape = self.shape

        sqlgen.add_table(f"DROP TABLE IF EXISTS {self.name};")

        # Extra dimension for complex (real, imag)
        columns = [f"{index_to_subscript(i)} INTEGER" for i in range(len(self.shape))]
        columns.append(f"val DOUBLE PRECISION{'[]' * (len(self.shape) + 1)}")
        sqlgen.add_table(f"CREATE TABLE {self.name} ({', '.join(columns)});")

        values = []
        self._gen_tile_values(values, tile_shape, self.data, [])

        if values:
            formatted_values = [
                f"({', '.join(str(i) for i in outer_index)}, ARRAY{val})"
                for outer_index, val in values
            ]
            sqlgen.add_table(f"INSERT INTO {self.name} VALUES {', '.join(formatted_values)};")

    def _gen_tile_values(
        self,
        results: list,
        tile_shape: tuple,
        tile: np.ndarray,
        outer_index: list[int],
    ):
        """Generate tile values for SQL insertion."""
        dim = len(outer_index)

        for i in range(self.shape[dim] // tile_shape[dim]):
            outer_index.append(i)
            subtile = np.take(tile, range(i * tile_shape[dim], (i + 1) * tile_shape[dim]), dim)

            if len(outer_index) == len(self.shape):
                if np.count_nonzero(subtile) > 0:
                    array_str = self._tile_to_array_string(tile_shape, subtile, [])
                    results.append((tuple(outer_index), array_str))
            else:
                self._gen_tile_values(results, tile_shape, subtile, outer_index)

            outer_index.pop()

    def _tile_to_array_string(
        self,
        tile_shape: tuple,
        tile: np.ndarray,
        inner_index: list[int],
    ) -> str:
        """Convert a tile to PostgreSQL array string format with complex numbers."""
        dim = len(inner_index)

        if dim + 1 == len(tile_shape):
            return f"[{', '.join(f'[{c.real}, {c.imag}]' for c in tile)}]"

        elements = []
        for i in range(tile_shape[dim]):
            inner_index.append(i)
            elements.append(self._tile_to_array_string(tile_shape, tile[i], inner_index))
            inner_index.pop()

        return f"[{', '.join(elements)}]"


class SparseTensor(Tensor):
    """Sparse tensor with COO format support."""

    def __init__(self, name: str, data: torch.Tensor):
        """
        Initialize sparse tensor.

        Args:
            name: Tensor name
            data: PyTorch sparse COO tensor
        """
        super().__init__(name, data.shape)
        self._compute_sparsity_metrics(data)

    def _compute_sparsity_metrics(self, data: torch.Tensor):
        """Compute sparsity information from COO format."""
        tuple_counter = {s: (set(), set(), set()) for s in self.tile_shapes}

        for i, j, val in zip(*data.indices(), data.values()):
            if val == 0:
                continue

            for tile_shape, sets in tuple_counter.items():
                key = (int(i) // tile_shape[0], int(j) // tile_shape[1])
                sets[0].add(key[0])
                sets[1].add(key[1])
                sets[2].add(key)

        for tile_shape, scheme in self.schemes.items():
            sets = tuple_counter[tile_shape]
            scheme.num_tuples = len(sets[2])
            scheme.value_count = (len(sets[0]), len(sets[1]))

    def gen_sql(self, sqlgen: SQLGenerator, tile_shape: tuple | None = None):
        # Sparse tensor SQL generation not yet implemented
        pass


# =============================================================================
# Reduction (Einstein Contraction)
# =============================================================================

class Reduction(Tensor):
    """
    Tensor contraction using Einstein summation notation.

    Computes operations like matrix multiplication by contracting
    over shared indices.
    """

    @staticmethod
    def infer_shape_and_keys(
        output: str,
        inputs: tuple[IndexedTerm, IndexedTerm],
    ) -> tuple[tuple, list, list, str]:
        """
        Infer output shape and join/aggregation keys from Einstein notation.

        Returns:
            - shape: Output tensor shape
            - join_keys: Indices to join on
            - aggregation_keys: Indices to aggregate over
            - pattern: String pattern for kernel naming
        """
        subscripts = {}
        join_keys = []
        pattern = ""

        for tensor_term in inputs:
            for i, subscript in enumerate(tensor_term.indices):
                if subscript not in subscripts:
                    pattern += index_to_subscript(len(subscripts))
                    subscripts[subscript] = (
                        tensor_term.tensor.shape[i],
                        (tensor_term.tensor, i),
                        index_to_subscript(len(subscripts)),
                    )
                else:
                    value = subscripts[subscript]
                    if value[0] != tensor_term.tensor.shape[i]:
                        raise ValueError(
                            f"Dimension mismatch for subscript '{subscript}': "
                            f"expected {value[0]}, got {tensor_term.tensor.shape[i]}"
                        )
                    join_keys.append((value[1], (tensor_term.tensor, i)))
                    pattern += index_to_subscript(value[1][1])
            pattern += f"{len(tensor_term.tensor.shape)}_"

        shape = tuple(subscripts[s][0] for s in output)
        aggregation_keys = [subscripts[s][1] for s in output]

        for s in output:
            pattern += subscripts[s][2]

        return shape, join_keys, aggregation_keys, pattern

    def __init__(
        self,
        name: str,
        aggregation_op: str,
        output: str,
        inputs: tuple[IndexedTerm, IndexedTerm],
        join_op: str = "mul",
    ):
        shape, join_keys, aggregation_keys, pattern = self.infer_shape_and_keys(output, inputs)
        super().__init__(name, shape, inputs[0].tensor.is_complex)

        self.join_op = join_op or "mul"
        self.join_keys = join_keys
        self.aggregation_op = aggregation_op
        self.aggregation_keys = aggregation_keys
        self.pattern = pattern
        self.lhs = inputs[0]
        self.rhs = inputs[1]

        # Initialize costs to infinity
        for scheme in self.schemes.values():
            scheme.cost = float("inf")
            scheme.accumulated_cost = float("inf")

        # Compute cost for each scheme
        self._compute_all_costs(inputs)

    def _compute_all_costs(self, inputs: tuple[IndexedTerm, IndexedTerm]):
        """Compute costs for all scheme combinations."""
        lhs_key_indices = tuple(key[0][1] for key in self.join_keys)
        rhs_key_indices = tuple(key[1][1] for key in self.join_keys)

        # Group LHS schemes by join key dimensions
        lhs_groups: dict[tuple, list[TilingScheme]] = {}
        for tile_shape, lhs in inputs[0].tensor.schemes.items():
            key = tuple(tile_shape[idx] for idx in lhs_key_indices)
            lhs_groups.setdefault(key, []).append(lhs)

        # Match RHS schemes with compatible LHS schemes
        for tile_shape, rhs in inputs[1].tensor.schemes.items():
            key = tuple(tile_shape[idx] for idx in rhs_key_indices)
            if key in lhs_groups:
                for lhs in lhs_groups[key]:
                    self._update_cost(lhs, rhs)

    def _update_cost(self, lhs: TilingScheme, rhs: TilingScheme):
        """Update cost for a specific scheme combination."""
        # Determine output tile shape
        tile_shape = []
        agg_num_tuples = 1

        for key in self.aggregation_keys:
            if key[0] is lhs.node:
                tile_shape.append(lhs.tile_shape[key[1]])
                agg_num_tuples *= lhs.value_count[key[1]]
            elif key[0] is rhs.node:
                tile_shape.append(rhs.tile_shape[key[1]])
                agg_num_tuples *= rhs.value_count[key[1]]
            else:
                raise RuntimeError("Aggregation key not found in either operand")

        tile_shape = tuple(tile_shape)
        scheme = self.schemes.get(tile_shape)
        if scheme is None:
            return

        # Calculate communication cost
        lhs_size = (len(lhs.tile_shape) + lhs.tile_size) * lhs.num_tuples
        rhs_size = (len(rhs.tile_shape) + rhs.tile_size) * rhs.num_tuples
        join_comm = COMM_COST_MULTIPLIER * (lhs_size + rhs_size)

        # Calculate join tuple count
        join_divisor = prod(
            max(lhs.value_count[key[0][1]], rhs.value_count[key[1][1]])
            for key in self.join_keys
        )
        join_num_tuples = lhs.num_tuples * rhs.num_tuples / join_divisor

        # Calculate tile join cost
        tile_join_cost = lhs.tile_size * rhs.tile_size
        for key in self.join_keys:
            if key[0][0] is lhs.node:
                tile_join_cost /= lhs.tile_shape[key[0][1]]
            elif key[0][0] is rhs.node:
                tile_join_cost /= rhs.tile_shape[key[0][1]]

        # Calculate number of groups per tile
        tile_num_group = 1.0
        for key in self.aggregation_keys:
            if key[0] is lhs.node:
                tile_num_group *= lhs.tile_shape[key[1]]
            elif key[0] is rhs.node:
                tile_num_group *= rhs.tile_shape[key[1]]

        join_kernel_cost = 2 * tile_join_cost - tile_num_group
        join_flops = join_num_tuples * join_kernel_cost

        agg_num_tuples = min(join_num_tuples / 2, agg_num_tuples)

        # Aggregation communication cost
        agg_comm = (
            COMM_COST_MULTIPLIER
            * (len(self.aggregation_keys) + len(self.join_keys) + scheme.tile_size)
            * join_num_tuples
        )

        agg_kernel_cost = scheme.tile_size
        agg_flops = (join_num_tuples / agg_num_tuples - 1) * agg_num_tuples * agg_kernel_cost

        # Total cost with relational operation overhead
        cost = join_comm + join_flops + agg_comm + agg_flops
        cost += join_num_tuples * RELOP_COST_MULTIPLIER
        cost += (join_num_tuples / agg_num_tuples - 1) * agg_num_tuples * RELOP_COST_MULTIPLIER

        # Update scheme if this is the best cost so far
        dependencies = lhs.dependencies | rhs.dependencies | {lhs, rhs}
        accumulated_cost = cost + sum(s.cost for s in dependencies)

        if accumulated_cost < scheme.accumulated_cost:
            scheme.cost = cost
            scheme.comm = join_comm + agg_comm
            scheme.flops = join_flops + agg_flops
            scheme.accumulated_cost = accumulated_cost
            scheme.accumulated_comm = scheme.comm + sum(s.comm for s in dependencies)
            scheme.accumulated_flops = scheme.flops + sum(s.flops for s in dependencies)
            scheme.source = (lhs, rhs)
            scheme.dependencies = dependencies

    def gen_sql(self, sqlgen: SQLGenerator, tile_shape: tuple | None = None):
        """Generate SQL query for this reduction."""
        select_cols = [
            f"{key[0].name}.{index_to_subscript(key[1])} AS {index_to_subscript(i)}"
            for i, key in enumerate(self.aggregation_keys)
        ]

        join_func = f"{self.join_op}_{self.pattern}"
        if self.is_complex:
            join_func += "_complex"

        select_cols.append(
            f"{self.aggregation_op}({join_func}("
            f"{self.lhs.tensor.name}.val, {self.rhs.tensor.name}.val)) AS val"
        )

        from_clause = [self.lhs.tensor.name, self.rhs.tensor.name]

        where_clause = [
            f"{lhs[0].name}.{index_to_subscript(lhs[1])} = {rhs[0].name}.{index_to_subscript(rhs[1])}"
            for lhs, rhs in self.join_keys
        ]

        group_by_clause = [
            f"{tensor.name}.{index_to_subscript(i)}"
            for tensor, i in self.aggregation_keys
        ]

        sql = (
            f"{self.name} AS ("
            f"SELECT {', '.join(select_cols)}\n"
            f"FROM {', '.join(from_clause)}\n"
            f"WHERE {' AND '.join(where_clause)}\n"
            f"GROUP BY {', '.join(group_by_clause)})"
        )

        sqlgen.add_intermediate_query(sql)


# =============================================================================
# Einsum Function
# =============================================================================

def einsum(name: str, output: str, *args, **kwargs) -> Tensor:
    """
    Compute Einstein summation with automatic contraction path optimization.

    Supports two APIs:
    1. Legacy: einsum("result", "ij", A["ik"], B["kj"])
    2. Modern: einsum("result", "ij,jk->ik", A, B)

    Args:
        name: Name for the output tensor
        output: Einstein notation string or output indices
        *args: Input tensors (with or without index subscripts)

    Returns:
        Resulting tensor from the contraction
    """
    tensors = list(args)

    # Legacy API with IndexedTerms
    if isinstance(tensors[0], IndexedTerm):
        return Reduction(name, "SUM", output, tuple(tensors))

    # Modern API with full einsum string
    einsum_string = output
    label_dims = get_label_dimensions(einsum_string, [t.shape for t in tensors])

    views = oe.helpers.build_views(einsum_string, label_dims)
    _, path_info = oe.contract_path(einsum_string, *views)

    for i, contraction in enumerate(path_info.contraction_list):
        indices = contraction[0]
        formula = contraction[2]

        input_0 = tensors[indices[0]]
        input_1 = tensors[indices[1]]

        # Remove contracted tensors
        for index in sorted(indices, reverse=True):
            del tensors[index]

        # Parse formula
        subscript_labels = formula.split("->")[0].split(",")
        input_0 = input_0[subscript_labels[0]]
        input_1 = input_1[subscript_labels[1]]
        out_labels = formula.split("->")[1]

        output_name = f"_t{i}" if tensors else name
        tensors.append(einsum(output_name, out_labels, input_0, input_1))

    if len(tensors) != 1:
        raise RuntimeError("Einsum contraction did not produce exactly one tensor")

    return tensors[0]


# =============================================================================
# SQL Generator
# =============================================================================

class SQLGenerator:
    """
    Generates SQL queries and C kernel code for tensor operations.

    Performs topological sorting of operations and generates:
    - INSERT statements for input tensors
    - SELECT queries with CTEs for intermediate results
    """

    # PostgreSQL C extension header
    PG_HEADER = '''#include "postgres.h"

#include "catalog/pg_type.h"
#include "utils/array.h"

#include "fmgr.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

'''

    # Pre-defined function declarations
    FUNCTION_DECLARATIONS = '''CREATE OR REPLACE FUNCTION mul_ij2_jk2_ik(double precision[], double precision[])
RETURNS double precision[] AS '/usr/lib/postgresql/12/lib/libeinsql.so', 'mul_ij2_jk2_ik'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION matrix_add(double precision[], double precision[])
RETURNS double precision[] AS '/usr/lib/postgresql/12/lib/libeinsql.so', 'matrix_add'
LANGUAGE C STRICT;

CREATE OR REPLACE AGGREGATE SUM(double precision[]) (
    sfunc = matrix_add,
    stype = double precision[]
);

CREATE OR REPLACE FUNCTION mul_ij2_j1_i_complex(double precision[][][], double precision[][])
RETURNS double precision[][] AS '/usr/lib/postgresql/12/lib/libeinsql.so', 'mul_ij2_j1_i_complex'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION mul_ij2_jk2_ik_complex(double precision[][][], double precision[][][])
RETURNS double precision[][][] AS '/usr/lib/postgresql/12/lib/libeinsql.so', 'mul_ij2_jk2_ik_complex'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION mul_ijk3_jkl3_il_complex(double precision[][][][], double precision[][][][])
RETURNS double precision[][][] AS '/usr/lib/postgresql/12/lib/libeinsql.so', 'mul_ijk3_jkl3_il_complex'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION mul_ijk3_klm3_ijlm_complex(double precision[][][][], double precision[][][][])
RETURNS double precision[][][][][] AS '/usr/lib/postgresql/12/lib/libeinsql.so', 'mul_ijk3_klm3_ijlm_complex'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION add_2(double precision[][], double precision[][])
RETURNS double precision[][] AS '/usr/lib/postgresql/12/lib/libeinsql.so', 'add_2'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION add_3(double precision[][][], double precision[][][])
RETURNS double precision[][][] AS '/usr/lib/postgresql/12/lib/libeinsql.so', 'add_3'
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION add_5(double precision[][][][][], double precision[][][][][])
RETURNS double precision[][][][][] AS '/usr/lib/postgresql/12/lib/libeinsql.so', 'add_5'
LANGUAGE C STRICT;

CREATE OR REPLACE AGGREGATE SUM(double precision[][]) (
    sfunc = add_2,
    stype = double precision[][]
);

CREATE OR REPLACE AGGREGATE SUM(double precision[][][]) (
    sfunc = add_3,
    stype = double precision[][][]
);

CREATE OR REPLACE AGGREGATE SUM(double precision[][][][][]) (
    sfunc = add_5,
    stype = double precision[][][][][]
);
'''

    def __init__(
        self,
        root: Tensor,
        insert_filename: str,
        query_filename: str | None = None,
        tile_shape: tuple | None = None,
    ):
        """
        Generate SQL files for a tensor computation.

        Args:
            root: Root tensor of the computation graph
            insert_filename: Output file for INSERT statements
            query_filename: Output file for SELECT queries (optional)
            tile_shape: Specific tiling scheme to use (optional)
        """
        self.tables: list[str] = []
        self.intermediate_queries: list[str] = []

        # Select best scheme if not specified
        if tile_shape is None:
            best_scheme = min(
                root.schemes.values(),
                key=lambda s: s.accumulated_cost
            )
            working_list = [best_scheme]
        else:
            working_list = [root.schemes[tile_shape]]

        # Build dependency graph
        graph = self._build_dependency_graph(working_list)

        # Topological sort
        ordered_schemes = self._topological_sort(graph)

        # Generate SQL for each scheme
        for scheme in ordered_schemes:
            scheme.node.gen_sql(self, scheme.tile_shape)

        # Write output files
        self._write_insert_file(insert_filename)

        if query_filename:
            final_table = ordered_schemes[-1].node.name
            self._write_query_file(query_filename, final_table)

    def _build_dependency_graph(
        self,
        working_list: list[TilingScheme],
    ) -> dict[TilingScheme, tuple[set, set]]:
        """Build adjacency list representation of dependency graph."""
        graph: dict[TilingScheme, tuple[set, set]] = {}

        while working_list:
            scheme = working_list.pop(0)

            if scheme not in graph:
                graph[scheme] = (set(), set())

            for src in scheme.source:
                working_list.append(src)
                if src not in graph:
                    graph[src] = (set(), set())
                graph[src][1].add(scheme)
                graph[scheme][0].add(src)

        return graph

    def _topological_sort(
        self,
        graph: dict[TilingScheme, tuple[set, set]],
    ) -> list[TilingScheme]:
        """Perform topological sort on dependency graph."""
        ordered = []
        graph = {k: (set(v[0]), set(v[1])) for k, v in graph.items()}  # Copy

        while graph:
            # Find node with no incoming edges
            ready = None
            for scheme, (incoming, outgoing) in graph.items():
                if not incoming:
                    ready = scheme
                    break

            if ready is None:
                raise RuntimeError("Cycle detected in dependency graph")

            # Remove from graph
            _, outgoing = graph.pop(ready)
            for out in outgoing:
                graph[out][0].discard(ready)

            ordered.append(ready)

        return ordered

    def _write_insert_file(self, filename: str):
        """Write INSERT statements to file."""
        with open(filename, "w") as f:
            for sql in self.tables:
                f.write(sql)
                f.write("\n\n")

    def _write_query_file(self, filename: str, final_table: str):
        """Write SELECT query to file."""
        with open(filename, "w") as f:
            f.write(self.FUNCTION_DECLARATIONS)
            f.write("\nWITH\n")
            f.write(",\n\n".join(self.intermediate_queries))
            f.write("\n\n")
            f.write(f"SELECT * FROM {final_table};")

    def add_table(self, sql: str):
        """Add a table creation/insertion statement."""
        self.tables.append(sql)

    def add_intermediate_query(self, sql: str):
        """Add an intermediate CTE query."""
        self.intermediate_queries.append(sql)


# =============================================================================
# Aliases for backward compatibility
# =============================================================================

# Legacy class names
DenseMatrix = DenseTensor
DenseTensorComplex = ComplexTensor
SparseMatrix = SparseTensor
SQLGen = SQLGenerator
itos = index_to_subscript
