"""
EinSQL: Einstein Summation for SQL

A library for optimizing tensor contractions using Einstein notation
and generating PostgreSQL queries for database-driven tensor computations.
"""

from .einsql import (
    # Constants
    COMM_COST_MULTIPLIER,
    RELOP_COST_MULTIPLIER,
    # Utility functions
    index_to_subscript,
    find_all_factors,
    get_label_dimensions,
    itos,  # Alias for backward compatibility
    # Expression classes
    Expr,
    UnaryOp,
    BinaryOp,
    Constant,
    # Tiling
    TilingScheme,
    # Kernel generators
    KernelAdd,
    KernelMul,
    # Tensor classes
    Tensor,
    IndexedTerm,
    DenseTensor,
    ComplexTensor,
    SparseTensor,
    Reduction,
    # Aliases for backward compatibility
    DenseMatrix,
    DenseTensorComplex,
    SparseMatrix,
    # Main function
    einsum,
    # SQL generation
    SQLGenerator,
    SQLGen,  # Alias for backward compatibility
)

__version__ = "0.1.0"
__all__ = [
    # Constants
    "COMM_COST_MULTIPLIER",
    "RELOP_COST_MULTIPLIER",
    # Utility functions
    "index_to_subscript",
    "find_all_factors",
    "get_label_dimensions",
    "itos",
    # Expression classes
    "Expr",
    "UnaryOp",
    "BinaryOp",
    "Constant",
    # Tiling
    "TilingScheme",
    # Kernel generators
    "KernelAdd",
    "KernelMul",
    # Tensor classes
    "Tensor",
    "IndexedTerm",
    "DenseTensor",
    "ComplexTensor",
    "SparseTensor",
    "Reduction",
    # Aliases
    "DenseMatrix",
    "DenseTensorComplex",
    "SparseMatrix",
    # Main function
    "einsum",
    # SQL generation
    "SQLGenerator",
    "SQLGen",
]
