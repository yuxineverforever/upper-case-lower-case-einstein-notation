# Upper-Case-Lower-Case-Einstein-Notation

**Upper-Case-Lower-Case-Einstein-Notation** - A library for optimizing tensor contractions using Einstein notation and generating PostgreSQL queries for database-driven tensor computations.

## Overview

This library bridges the gap between tensor computing and relational databases. It enables PostgreSQL and PlinyCompute to execute complex Einstein summation (einsum) operations through:

- **Automatic tiling scheme optimization**: Enumerates all possible tensor tilings and selects the optimal configuration based on a cost model
- **Query generation**: Produces PostgreSQL queries with Common Table Expressions (CTEs) for efficient execution
- **Kernel generation**: Creates PostgreSQL user-defined functions (UDFs) for tensor operations
- **Sparsity-aware optimization**: Cost model accounts for tensor sparsity patterns

## Features

- Einstein notation support via `einsum()` function
- Automatic contraction path optimization using [opt_einsum](https://github.com/dgasmith/opt_einsum)
- Multiple tensor types: Dense, Sparse, and Complex
- Cost-based tiling scheme selection
- PostgreSQL C extension code generation
- Topological sorting for optimal query execution order

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL 12+
- PyTorch
- NumPy
- opt_einsum

### Install Dependencies

```bash
pip install torch numpy opt_einsum
```

### Docker Setup

A Dockerfile is provided for a complete development environment:

```bash
cd docker
docker build -t einsql .
docker run -it einsql
```

After entering the docker container, you can run a simple test:
```bash
python gan_sparse_test.py # generate the schema
sh gan_sparse_test_0_build.sh # generate the TACO kernel
python run_postgresql_2steps.py gan_sparse_test_0_insert.sql gan_sparse_test_0_query.sql # Run the EinSum with PostgreSQL
```

The Docker image includes:
- PostgreSQL 12
- Python 3 with all dependencies
- Tensor Compiler (Taco) for sparse tensor support

## Quick Start

### Basic Matrix Multiplication

```python
import torch
import einsql

# Create dense tensors
A = einsql.DenseTensor("A", torch.randn(4, 8))
B = einsql.DenseTensor("B", torch.randn(8, 16))

# Compute matrix multiplication using Einstein notation
# C[i,k] = sum_j A[i,j] * B[j,k]
C = einsql.einsum("C", "ik", A["ij"], B["jk"])

# Generate SQL files
einsql.SQLGenerator(C, "matmul_insert.sql", "matmul_query.sql")
```

### Multi-Tensor Contraction

```python
import numpy as np
import einsql

# Create tensors
X = einsql.DenseTensor("X", torch.randn(4, 3))
W_Q = einsql.DenseTensor("W_Q", torch.randn(3, 8))
W_K = einsql.DenseTensor("W_K", torch.randn(3, 8))
A = einsql.DenseTensor("A", torch.randn(4, 4))

# Graph attention: attention[n,m] = sum over d,h,f X[n,d] * W_Q[d,h] * A[n,m] * X[m,f] * W_K[f,h]
result = einsql.einsum("attention", "nd,dh,nm,mf,fh->nm", X, W_Q, A, X, W_K)

# Get optimal tiling scheme
schemes = sorted(result.schemes.items(), key=lambda kv: kv[1].accumulated_cost)
best_scheme = schemes[0][1]
print(f"Optimal tiling: {best_scheme.tile_shape}")
print(f"Total cost: {best_scheme.accumulated_cost}")
```

### Working with Sparse Tensors

```python
import torch
import einsql

# Create sparse tensors (identity matrices)
A = einsql.SparseTensor("A", torch.eye(100).to_sparse_coo())
B = einsql.SparseTensor("B", torch.eye(100).to_sparse_coo())

# The cost model will favor smaller tiles for sparse data
result = einsql.einsum("C", "ik", A["ij"], B["jk"])

# View scheme costs
for tile_shape, scheme in sorted(result.schemes.items(), key=lambda x: x[1].accumulated_cost)[:3]:
    print(f"Tile shape: {tile_shape}, Cost: {scheme.accumulated_cost:.2f}")
```

## Architecture

### Tiling Scheme

Each tensor maintains a set of possible tiling configurations. A `TilingScheme` tracks:

- `tile_shape`: The dimensions of each tile
- `cost`: Estimated execution cost
- `comm`: Communication cost (data transfer)
- `flops`: Computational cost
- `accumulated_cost`: Total cost including dependencies

### Cost Model

The cost model considers:

1. **Communication cost**: Data transfer between operators
2. **Computation cost**: FLOPs for tile operations
3. **Relational operation cost**: Join and aggregation overhead
4. **Sparsity**: Fewer non-zero tiles reduce cost

Key constants (configurable):
- `COMM_COST_MULTIPLIER = 1`: Weight for communication costs
- `RELOP_COST_MULTIPLIER = 15`: Weight for relational operations


## Running Tests

```bash
cd einsql
python -m pytest test.py -v
```

Or using unittest:

```bash
python -m unittest test -v
```

## PostgreSQL Setup

1. Build the C extensions:

```bash
cd postgres-mkl
mkdir build && cd build
cmake ..
make
sudo make install
```

2. Load the extension in PostgreSQL:

```sql
CREATE EXTENSION einsql;
```

3. Run generated SQL files:

```bash
psql -d mydb -f matmul_insert.sql
psql -d mydb -f matmul_query.sql
```

## Examples

See the `src/` directory for complete examples:

- `gan_sparse_test.py`: Simple graph attention network test
- `gan_sparse_amazon.py`: Amazon product graph dataset
- `gan_sparse_citation_full.py`: Citation network analysis

## Performance Considerations

1. **Tile Size Selection**: Smaller tiles benefit sparse data; larger tiles reduce overhead for dense data
2. **Sparsity Awareness**: The cost model automatically accounts for non-zero patterns
3. **Contraction Order**: Uses opt_einsum for optimal contraction path

## License

MIT License

## Acknowledgments

- [opt_einsum](https://github.com/dgasmith/opt_einsum) for contraction path optimization
- [Tensor Algebra Compiler (TACO)](https://github.com/tensor-compiler/taco) for sparse tensor compilation
- Intel MKL for optimized linear algebra operations
