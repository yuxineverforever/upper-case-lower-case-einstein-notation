import itertools
import uuid
from math import prod
import subprocess

import torch
import numpy as np
import opt_einsum as oe
from typing import Tuple, List


class Expr:
    def __init__(self):
        return

    def flops(self):
        pass


class UnaryOp(Expr):
    def __init__(self, opcode: str, expr: Expr):
        super().__init__()
        self.opcode = opcode
        self.expr = expr


class BinaryOp(Expr):
    def __init__(self, opcode: str, lhs: Expr, rhs: Expr):
        super().__init__()
        self.opcode = opcode
        self.lhs = lhs
        self.rhs = rhs

    def flops(self):
        return 1.0 + self.lhs.flops() + self.rhs.flops()


class Constant(Expr):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def flops(self):
        return 0


class KernelAdd:
    def __init__(self, name, ndim):
        self.name = name
        self.ndim = ndim

    def codegen(self) -> str:
        c_source = ""
        c_source += f"PG_FUNCTION_INFO_V1({self.name});\n"
        c_source += f"Datum {self.name}(PG_FUNCTION_ARGS) {{\n"
        c_source += "  ArrayType *mat_a = PG_GETARG_ARRAYTYPE_P(0);\n"
        c_source += "  ArrayType *mat_b = PG_GETARG_ARRAYTYPE_P(1);\n"

        subscripts = [itos(i) for i in range(self.ndim)]
        dim_vars = [f"n{s}" for s in subscripts]
        for i, d in enumerate(dim_vars):
            c_source += f"  int {d} = (int)ARR_DIMS(mat_a)[{i}];\n"

        c_source += "  double *a = (double *)ARR_DATA_PTR(mat_a);\n"
        c_source += "  double *b = (double *)ARR_DATA_PTR(mat_b);\n"

        out_size = " * ".join(dim_vars + ["sizeof(double)"])
        c_source += f"  double *c = (double *)palloc({out_size});\n"
        c_source += f"  memset(c, 0, {out_size});\n"

        rbraces = []
        for i, (s, d) in enumerate(zip(subscripts, dim_vars)):
            c_source += "  " * (i + 1)
            c_source += f"for (int {s} = 0; {s} < {d}; ++{s}) {{\n"
            rbraces.append("  " * (i + 1) + "}\n")

        index = " + ".join(
            " * ".join([s] + dim_vars[j + 1 :]) for j, s in enumerate(subscripts)
        )
        c_source += "  " * (len(rbraces) + 1)
        c_source += f"c[{index}] = a[{index}] + b[{index}];\n"

        for rbrace in reversed(rbraces):
            c_source += rbrace

        c_source += "  ArrayType *result = NULL;\n"

        c_source += f"  int dims[{len(dim_vars)}] = {{{', '.join(dim_vars)}}};\n"
        c_source += (
            f"  int lbs[{len(dim_vars)}] = {{{', '.join(['1'] * len(dim_vars))}}};\n"
        )
        c_source += f"""    result = construct_md_array((Datum *)c, NULL, {len(dim_vars)}, dims, lbs, FLOAT8OID,
                              sizeof(float8), true, 'd');\n"""
        c_source += "  pfree(c);\n"
        c_source += "  PG_RETURN_ARRAYTYPE_P(result);\n"
        c_source += "}\n\n"
        return c_source


class KernelMul:
    def __init__(self, name, subscripts, lhs, rhs, out, is_complex):
        """A kernel represent a einsum expression.
        For example, ij,jk->ik will have
        MulKernel(['i', 'j', 'k'], ['i', 'j'], ['j', 'k'], ['i', 'k'])"""
        self.name = name
        self.subscripts = subscripts
        self.lhs = lhs
        self.rhs = rhs
        self.out = out
        self.is_complex = is_complex

    def codegen(self) -> str:
        c_source = ""
        c_source += f"PG_FUNCTION_INFO_V1({self.name});\n"
        c_source += f"Datum {self.name}(PG_FUNCTION_ARGS) {{\n"
        c_source += "  ArrayType *mat_a = PG_GETARG_ARRAYTYPE_P(0);\n"
        c_source += "  ArrayType *mat_b = PG_GETARG_ARRAYTYPE_P(1);\n"

        # int ni, nj, nk
        dim_vars = [f"n{s}" for s in self.subscripts]
        lhs_dim_vars = [f"n{s}" for s in self.lhs]
        rhs_dim_vars = [f"n{s}" for s in self.rhs]
        out_dim_vars = [f"n{s}" for s in self.out]

        for s, d in zip(self.subscripts, dim_vars):
            if s in self.lhs:
                c_source += f"  int {d} = (int)ARR_DIMS(mat_a)[{self.lhs.index(s)}];\n"
            else:
                c_source += f"  int {d} = (int)ARR_DIMS(mat_b)[{self.rhs.index(s)}];\n"

        c_source += "  double *a = (double *)ARR_DATA_PTR(mat_a);\n"
        c_source += "  double *b = (double *)ARR_DATA_PTR(mat_b);\n"

        out_size = " * ".join(out_dim_vars)
        if self.is_complex:
            out_size += " * 2"
        out_size += " * sizeof(double)"

        c_source += f"  double *c = (double *)palloc({out_size});\n"
        c_source += f"  memset(c, 0, {out_size});\n"

        rbraces = []
        for i, (s, d) in enumerate(zip(self.subscripts, dim_vars)):
            c_source += "  " * (i + 1)
            c_source += f"for (int {s} = 0; {s} < {d}; ++{s}) {{\n"
            rbraces.append("  " * (i + 1) + "}\n")

        if not self.is_complex:
            # ["i * nj * nk * nl", "j * nk * nl", "k * nl", "l"]
            lhs_index = " + ".join(
                " * ".join([s] + lhs_dim_vars[j + 1 :]) for j, s in enumerate(self.lhs)
            )
            rhs_index = " + ".join(
                " * ".join([s] + rhs_dim_vars[j + 1 :]) for j, s in enumerate(self.rhs)
            )
            out_index = " + ".join(
                " * ".join([s] + out_dim_vars[j + 1 :]) for j, s in enumerate(self.out)
            )
            c_source += "  " * (len(rbraces) + 1)
            c_source += f"c[{out_index}] += a[{lhs_index}] * b[{rhs_index}];\n"
        else:
            lhs_real_index = " + ".join(
                " * ".join([s] + lhs_dim_vars[j + 1 :] + ["2"])
                for j, s in enumerate(self.lhs)
            )
            lhs_image_index = (
                " + ".join(
                    " * ".join([s] + lhs_dim_vars[j + 1 :] + ["2"])
                    for j, s in enumerate(self.lhs)
                )
                + " + 1"
            )

            rhs_real_index = " + ".join(
                " * ".join([s] + rhs_dim_vars[j + 1 :] + ["2"])
                for j, s in enumerate(self.lhs)
            )
            rhs_image_index = (
                " + ".join(
                    " * ".join([s] + rhs_dim_vars[j + 1 :] + ["2"])
                    for j, s in enumerate(self.lhs)
                )
                + " + 1"
            )

            out_real_index = " + ".join(
                " * ".join([s] + out_dim_vars[j + 1 :] + ["2"])
                for j, s in enumerate(self.out)
            )
            out_image_index = (
                " + ".join(
                    " * ".join([s] + out_dim_vars[j + 1 :] + ["2"])
                    for j, s in enumerate(self.out)
                )
                + " + 1"
            )

            c_source += "  " * (len(rbraces) + 1)
            c_source += f"double ar = a[{lhs_real_index}];\n"
            c_source += "  " * (len(rbraces) + 1)
            c_source += f"double ai = a[{lhs_image_index}];\n"
            c_source += "  " * (len(rbraces) + 1)
            c_source += f"double br = b[{rhs_real_index}];\n"
            c_source += "  " * (len(rbraces) + 1)
            c_source += f"double bi = b[{rhs_image_index}];\n"

            c_source += "  " * (len(rbraces) + 1)
            c_source += f"c[{out_real_index}] += ar * br - ai * bi;\n"
            c_source += "  " * (len(rbraces) + 1)
            c_source += f"c[{out_image_index}] += ar * bi + ai * br;\n"

        for rbrace in reversed(rbraces):
            c_source += rbrace

        c_source += "  ArrayType *result = NULL;\n"

        if not self.is_complex:
            c_source += (
                f"  int dims[{len(out_dim_vars)}] = {{{', '.join(out_dim_vars)}}};\n"
            )
            c_source += f"  int lbs[{len(out_dim_vars)}] = {{{', '.join(['1'] * len(out_dim_vars))}}};\n"
            c_source += f"""    result = construct_md_array((Datum *)c, NULL, {len(out_dim_vars)}, dims, lbs, FLOAT8OID,
                              sizeof(float8), true, 'd');\n"""
        else:
            c_source += f"  int dims[{len(out_dim_vars) + 1}] = {{{', '.join(out_dim_vars + ['2'])}}};\n"
            c_source += f"  int lbs[{len(out_dim_vars) + 1}] = {{{', '.join(['1'] * (len(out_dim_vars) + 1))}}};\n"
            c_source += f"""    result = construct_md_array((Datum *)c, NULL, {len(out_dim_vars) + 1}, dims, lbs, FLOAT8OID,
                              sizeof(float8), true, 'd');\n"""
        c_source += "  pfree(c);\n"
        c_source += "  PG_RETURN_ARRAYTYPE_P(result);\n"
        c_source += "}\n\n"

        return c_source

class TacoKernelMul:
    def __init__(self, name, subscripts, lhs, rhs, out, prefix=""):
        self.name = name
        self.subscripts = subscripts
        self.lhs = lhs
        self.rhs = rhs
        self.out = out

        self.expr = f"C({','.join(self.out)})=A({','.join(self.lhs)})*B({','.join(self.rhs)})"
        self.out_format = f"C:{'d'*len(self.out)}:{','.join(str(i) for i in range(len(self.out)))}"
        self.lhs_format = f"A:{'d'*len(self.lhs)}:{','.join(str(i) for i in range(len(self.lhs)))}"
        self.rhs_format = f"B:{'d'*len(self.rhs)}:{','.join(str(i) for i in range(len(self.rhs)))}"
        self.prefix = prefix
        self.taco_prefix = f"{prefix}_{self.name}_"
        self.kernel_src = f"{prefix}_{self.name}.c"
        self.udf_src = f"{prefix}_{self.name}_pg_udf.c"
    
    def codegen(self):
        # taco "y(i)=A(i,j,k,l)*x(j,l,m,n)" -f=y:d:0 -f=A:dddd:0,1,2,3 -f=x:dddd:0,1,2,3 -prefix=ijkl_jlmn -write-source=taco_kernel_ijkl_jlmn.c
        # expr = f"C({','.join(self.out)})=A({','.join(self.lhs)})*B({','.join(self.rhs)})"
        # out_format = f"C:{'d'*len(self.out)}:{','.join(str(i) for i in range(len(self.out)))}"
        # lhs_format = f"A:{'d'*len(self.lhs)}:{','.join(str(i) for i in range(len(self.lhs)))}"
        # rhs_format = f"B:{'d'*len(self.rhs)}:{','.join(str(i) for i in range(len(self.rhs)))}"
        # prefix = self.name + "_"
        # source = f"tack_kernel_{self.name}.c"

        taco_cmd = [
            "taco",
            self.expr,
            "-f=" + self.out_format,
            "-f=" + self.lhs_format,
            "-f=" + self.rhs_format,
            "-prefix=" + self.taco_prefix,
            "-write-source=" + self.kernel_src,
        ]
        subprocess.run(taco_cmd, check=True)

        # open the generated C file and read its content
        with open(self.kernel_src, "r") as f:
            c_source = f.read()

        # prepend the postgresql header
        c_source = r"""#include "postgres.h"

#include "catalog/pg_type.h"
#include "utils/array.h"

#include "fmgr.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif


""" + c_source
        
        # append the postgresql udf
        c_source += f"PG_FUNCTION_INFO_V1({self.name});\n"
        c_source += f"Datum {self.name}(PG_FUNCTION_ARGS) {{\n"
        c_source += "  ArrayType *mat_a = PG_GETARG_ARRAYTYPE_P(0);\n"
        c_source += "  ArrayType *mat_b = PG_GETARG_ARRAYTYPE_P(1);\n"
        c_source += f"  int32_t dims_a[{len(self.lhs)}];\n"
        c_source += f"  int32_t dims_b[{len(self.rhs)}];\n"
        c_source += f"  int32_t dims_c[{len(self.out)}];\n"
        c_source += f"  for (int i = 0; i < {len(self.lhs)}; i++) {{\n"
        c_source += f"    dims_a[i] = (int32_t)ARR_DIMS(mat_a)[i];\n"
        c_source += f"  }}\n"
        c_source += f"  for (int i = 0; i < {len(self.rhs)}; i++) {{\n"
        c_source += f"    dims_b[i] = (int32_t)ARR_DIMS(mat_b)[i];\n"
        c_source += f"  }}\n"
        c_source += f"  for (int i = 0; i < {len(self.out)}; i++) {{\n"
        c_source += f"    dims_c[i] = (int32_t)ARR_DIMS(mat_a)[i];\n"
        c_source += f"  }}\n"
        out_dim_indices = []
        for s in self.out:
            found = False
            for i, d in enumerate(self.lhs):
                if s != d:
                    continue
                out_dim_indices.append(("mat_a", i))
                found = True
                break
            if found:
                continue
            for i, d in enumerate(self.rhs):
                if s != d:
                    continue
                out_dim_indices.append(("mat_b", i))
                found = True
                break
            if not found:
                raise ValueError(f"Dimension {s} not found in lhs or rhs")
        for i, (mat, index) in enumerate(out_dim_indices):
            c_source += f"  dims_c[{i}] = (int32_t)ARR_DIMS({mat})[{index}];\n"
        c_source += f"  int32_t mode_ordering_a[] = {{{','.join(str(i) for i in range(len(self.lhs)))}}};\n"
        c_source += f"  int32_t mode_ordering_b[] = {{{','.join(str(i) for i in range(len(self.rhs)))}}};\n"
        c_source += f"  int32_t mode_ordering_c[] = {{{','.join(str(i) for i in range(len(self.out)))}}};\n"
        c_source += f"  taco_mode_t mode_types_a[] = {{{','.join('0' for i in range(len(self.lhs)))}}};\n"
        c_source += f"  taco_mode_t mode_types_b[] = {{{','.join('0' for i in range(len(self.rhs)))}}};\n"
        c_source += f"  taco_mode_t mode_types_c[] = {{{','.join('0' for i in range(len(self.out)))}}};\n"
        c_source += f"  taco_tensor_t *a = init_taco_tensor_t({len(self.lhs)}, sizeof(double), dims_a, mode_ordering_a, mode_types_a);\n"
        c_source += f"  taco_tensor_t *b = init_taco_tensor_t({len(self.rhs)}, sizeof(double), dims_b, mode_ordering_b, mode_types_b);\n"
        c_source += f"  taco_tensor_t *c = init_taco_tensor_t({len(self.out)}, sizeof(double), dims_c, mode_ordering_c, mode_types_c);\n"
        c_source += f"  a->vals = (uint8_t *)ARR_DATA_PTR(mat_a);\n"
        c_source += f"  b->vals = (uint8_t *)ARR_DATA_PTR(mat_b);\n"
        c_source += f"  c->vals = (uint8_t *)palloc({'*'.join(f'dims_c[{i}]' for i in range(len(self.out)))} * sizeof(double));\n"
        c_source += f"  {self.taco_prefix}compute(c, a, b);\n"
        c_source += f"  int lbs[{len(self.out)}] = {{{', '.join(['1'] * len(self.out))}}};\n"
        c_source += f"""  ArrayType *result = construct_md_array((Datum *)c->vals, NULL, {len(self.out)}, dims_c, lbs, FLOAT8OID,
                            sizeof(float8), true, 'd');\n"""
        c_source += "  pfree(c->vals);\n"
        c_source += "  deinit_taco_tensor_t(a);\n"
        c_source += "  deinit_taco_tensor_t(b);\n"
        c_source += "  deinit_taco_tensor_t(c);\n"
        c_source += "  PG_RETURN_ARRAYTYPE_P(result);\n"
        c_source += "}\n\n"

        with open(self.udf_src, "w+") as f:
            f.write(c_source)
        



class Scheme:
    def __init__(
        self,
        node,
        shape: Tuple[int, int],
        tile_shape: Tuple[int, int],
    ):
        self.node = node
        self.shape = shape
        self.tile_shape = tile_shape
        self.tile_size = prod(tile_shape)
        self.cost = 0.0
        self.comm = 0.0
        self.flops = 0.0
        self.accumulated_cost = 0.0
        self.accumulated_comm = 0.0
        self.accumulated_flops = 0.0
        self.source = tuple()
        self.dependencies = set()

        self.value_count = tuple(
            [
                length // tile_length
                for length, tile_length in zip(self.shape, self.tile_shape)
            ]
        )
        self.num_tuples = prod(self.value_count)


def find_all_factors(d: int) -> List[int]:
    factors = [x for i in range(1, int(d**0.5) + 1) if d % i == 0 for x in (i, d // i)]
    return sorted(list(set(factors)))


class Tensor(Expr):
    def __init__(self, name: str, shape: tuple, is_complex=False):
        self.name = name
        self.shape = shape
        self.is_complex = is_complex

        # Find all factors of each dimension
        self.factors = (find_all_factors(d) for d in shape)
        # Find all possible tiling schemes
        self.tile_shapes = [s for s in itertools.product(*self.factors)]
        self.schemes = {
            tile_shape: Scheme(
                self,
                self.shape,
                tile_shape,
            )
            for tile_shape in self.tile_shapes
        }

    def __add__(self, other):
        return BinaryOp("+", self, other)

    def __sub__(self, other):
        return BinaryOp("-", self, other)

    def __div__(self, other):
        return BinaryOp("/", self, other)

    def __getitem__(self, indices: str):
        return IndexedTerm(self, indices)

    def flops(self):
        return 0.0

    def gen_sql(self, sqlgen, tile_shape=None):
        pass


class IndexedTerm:
    def __init__(self, tensor: Tensor, indices: str):
        self.tensor = tensor
        self.indices = indices


def itos(i: int) -> str:
    assert ord("i") + i <= ord("z")
    return chr(ord("i") + i)


class TorchDenseMatrix(Tensor):
    def __init__(self, name, mat):
        super().__init__(name, mat.shape)
        self.mat = mat

        tuple_counter = dict()
        for s in self.tile_shapes:
            tuple_counter[s] = [set() for i in range(len(self.shape) + 1)]
        self.compute_sparsity(tuple_counter, self.mat, [])

        for _, scheme in self.schemes.items():
            sets = tuple_counter[scheme.tile_shape]
            scheme.num_tuples = len(sets[-1])
            scheme.value_count = tuple([len(s) for s in sets[: len(sets) - 1]])

    def compute_sparsity(self, tuple_counter, submatrix, index: List[int]):
        dim = len(index)
        if dim + 1 == len(self.shape):
            for i in range(0, self.shape[dim]):
                if submatrix[i] == 0:
                    continue

                index.append(i)
                for tile_shape, sets in tuple_counter.items():
                    inner_index = []
                    for d in range(len(self.shape)):
                        sets[d].add(index[d] // tile_shape[d])
                        inner_index.append(index[d] // tile_shape[d])

                    sets[-1].add(tuple(inner_index))
                index.pop()
            return

        for i in range(0, self.shape[dim]):
            index.append(i)
            self.compute_sparsity(tuple_counter, submatrix[i], index)
            index.pop()

    def gen_sql(self, sqlgen, tile_shape=None):
        if tile_shape is None:
            tile_shape = self.shape

        sqlgen.add_table(f"DROP TABLE IF EXISTS {self.name}_{'_'.join(str(i) for i in tile_shape)};")

        # We use i to z as the index name. Make sure the dimension is not too high.
        colums = [f"{itos(i)} INTEGER" for i, _ in enumerate(self.shape)]
        colums.append(f"val DOUBLE PRECISION{'[]' * len(self.shape)}")
        sqlgen.add_table(f"CREATE TABLE {self.name}_{'_'.join(str(i) for i in tile_shape)} ({', '.join(colums)});")

        # Enumerate all outer indices
        values = []
        self.gen_values(values, tile_shape, self.mat, [])
        values = [
            f"({', '.join(str(i) for i in outer_index)}, ARRAY{val})"
            for outer_index, val in values
        ]
        values = ", ".join(values)
        sqlgen.add_table(f"INSERT INTO {self.name}_{'_'.join(str(i) for i in tile_shape)} VALUES {values};")

    def gen_values(self, results: List, tile_shape, tile, outer_index: List[int]):
        dim = len(outer_index)
        for i in range(0, self.shape[dim] // tile_shape[dim]):
            outer_index.append(i)
            subtile = torch.narrow(tile, dim, i * tile_shape[dim], tile_shape[dim])
            if len(outer_index) == len(self.shape):
                if torch.count_nonzero(subtile) > 0:
                    s = self.enumerate_tiles(tile_shape, subtile, outer_index, [])
                    results.append((tuple(outer_index), s))
            else:
                self.gen_values(results, tile_shape, subtile, outer_index)
            outer_index.pop()

    def enumerate_tiles(
        self,
        tile_shape,
        tile,
        outer_index: List[int],
        inner_index: List[int],
    ):
        dim = len(inner_index)
        if dim + 1 == len(tile_shape):
            return f"[{', '.join(str(float(f)) for f in tile)}]"

        local_results = []
        for i in range(0, tile_shape[dim]):
            inner_index.append(i)
            subtile = tile[i]
            local_results.append(
                self.enumerate_tiles(tile_shape, subtile, outer_index, inner_index)
            )
            inner_index.pop()

        return f"[{', '.join(t for t in local_results)}]"


class NumpyNDArray(Tensor):
    def __init__(self, name, mat):
        super().__init__(name, mat.shape)
        self.mat = mat

        tuple_counter = dict()
        for s in self.tile_shapes:
            # We append an extra set to record all outer indices, so that we can
            # compute num_tuples later.
            tuple_counter[s] = [set() for i in range(len(self.shape) + 1)]
        self.compute_sparsity(tuple_counter, self.mat, [])

        for _, scheme in self.schemes.items():
            sets = tuple_counter[scheme.tile_shape]
            scheme.num_tuples = len(sets[-1])
            scheme.value_count = tuple([len(s) for s in sets[: len(sets) - 1]])

    def compute_sparsity(self, tuple_counter, submatrix, index: List[int]):
        dim = len(index)
        if dim + 1 == len(self.shape):
            for i in range(0, self.shape[dim]):
                if np.isclose(submatrix[i], 0.0).all():
                    continue

                index.append(i)
                for tile_shape, sets in tuple_counter.items():
                    outer_index = []
                    for d in range(len(self.shape)):
                        sets[d].add(index[d] // tile_shape[d])
                        outer_index.append(index[d] // tile_shape[d])

                    sets[-1].add(tuple(outer_index))
                index.pop()
            return

        for i in range(0, self.shape[dim]):
            index.append(i)
            self.compute_sparsity(tuple_counter, submatrix[i], index)
            index.pop()

    def gen_sql(self, sqlgen, tile_shape=None):
        if tile_shape is None:
            tile_shape = self.shape

        sqlgen.add_table(f"DROP TABLE IF EXISTS {self.name}_{'_'.join(str(i) for i in tile_shape)};")

        # We use i to z as the index name. Make sure the dimension is not too high.
        colums = [f"{itos(i)} INTEGER" for i, _ in enumerate(self.shape)]

        colums.append(f"val DOUBLE PRECISION{'[]' * (len(self.shape))}")
        sqlgen.add_table(f"CREATE TABLE {self.name}_{'_'.join(str(i) for i in tile_shape)} ({', '.join(colums)});")

        # Enumerate all outer indices
        values = []
        self.gen_values(values, tile_shape, self.mat, [])
        values = [
            f"({', '.join(str(i) for i in outer_index)}, ARRAY{val})"
            for outer_index, val in values
        ]
        values = ", ".join(values)
        sqlgen.add_table(f"INSERT INTO {self.name}_{'_'.join(str(i) for i in tile_shape)} VALUES {values};")

    def gen_values(self, results: List, tile_shape, tile, outer_index: List[int]):
        dim = len(outer_index)
        for i in range(0, self.shape[dim] // tile_shape[dim]):
            outer_index.append(i)
            # subtile = np.narrow(tile, dim, i * tile_shape[dim], tile_shape[dim])
            subtile = np.take(
                tile, range(i * tile_shape[dim], (i + 1) * tile_shape[dim]), dim
            )
            if len(outer_index) == len(self.shape):
                if np.count_nonzero(subtile) > 0:
                    s = self.enumerate_tiles(tile_shape, subtile, outer_index, [])
                    results.append((tuple(outer_index), s))
            else:
                self.gen_values(results, tile_shape, subtile, outer_index)
            outer_index.pop()

    def enumerate_tiles(
        self,
        tile_shape,
        tile,
        outer_index: List[int],
        inner_index: List[int],
    ):
        dim = len(inner_index)
        if dim + 1 == len(tile_shape):
            return f"[{', '.join(f'{c}' for c in tile)}]"

        local_results = []
        for i in range(0, tile_shape[dim]):
            inner_index.append(i)
            subtile = tile[i]
            local_results.append(
                self.enumerate_tiles(tile_shape, subtile, outer_index, inner_index)
            )
            inner_index.pop()

        return f"[{', '.join(t for t in local_results)}]"


class NumpyNDArrayComplex(Tensor):
    def __init__(self, name, mat):
        super().__init__(name, mat.shape, True)
        self.mat = mat

        tuple_counter = dict()
        for s in self.tile_shapes:
            tuple_counter[s] = [set() for i in range(len(self.shape) + 1)]
        self.compute_sparsity(tuple_counter, self.mat, [])

        for _, scheme in self.schemes.items():
            sets = tuple_counter[scheme.tile_shape]
            scheme.num_tuples = len(sets[-1])
            scheme.value_count = tuple([len(s) for s in sets[: len(sets) - 1]])

    def compute_sparsity(self, tuple_counter, submatrix, index: List[int]):
        dim = len(index)
        if dim + 1 == len(self.shape):
            for i in range(0, self.shape[dim]):
                if np.isclose(submatrix[i], [0.0, 0.0]).all():
                    continue

                index.append(i)
                for tile_shape, sets in tuple_counter.items():
                    inner_index = []
                    for d in range(len(self.shape)):
                        sets[d].add(index[d] // tile_shape[d])
                        inner_index.append(index[d] // tile_shape[d])

                    sets[-1].add(tuple(inner_index))
                index.pop()
            return

        for i in range(0, self.shape[dim]):
            index.append(i)
            self.compute_sparsity(tuple_counter, submatrix[i], index)
            index.pop()

    def gen_sql(self, sqlgen, tile_shape=None):
        if tile_shape is None:
            tile_shape = self.shape

        sqlgen.add_table(f"DROP TABLE IF EXISTS {self.name}_{'_'.join(str(i) for i in tile_shape)};")

        # We use i to z as the index name. Make sure the dimension is not too high.
        colums = [f"{itos(i)} INTEGER" for i, _ in enumerate(self.shape)]
        # Append one more '[]' to store complex number
        colums.append(f"val DOUBLE PRECISION{'[]' * (len(self.shape) + 1)}")
        sqlgen.add_table(f"CREATE TABLE {self.name}_{'_'.join(str(i) for i in tile_shape)} ({', '.join(colums)});")

        # Enumerate all outer indices
        values = []
        self.gen_values(values, tile_shape, self.mat, [])
        values = [
            f"({', '.join(str(i) for i in outer_index)}, ARRAY{val})"
            for outer_index, val in values
        ]
        values = ", ".join(values)
        sqlgen.add_table(f"INSERT INTO {self.name}_{'_'.join(str(i) for i in tile_shape)} VALUES {values};")

    def gen_values(self, results: List, tile_shape, tile, outer_index: List[int]):
        dim = len(outer_index)
        for i in range(0, self.shape[dim] // tile_shape[dim]):
            outer_index.append(i)
            # subtile = np.narrow(tile, dim, i * tile_shape[dim], tile_shape[dim])
            subtile = np.take(
                tile, range(i * tile_shape[dim], (i + 1) * tile_shape[dim]), dim
            )
            if len(outer_index) == len(self.shape):
                if np.count_nonzero(subtile) > 0:
                    s = self.enumerate_tiles(tile_shape, subtile, outer_index, [])
                    results.append((tuple(outer_index), s))
            else:
                self.gen_values(results, tile_shape, subtile, outer_index)
            outer_index.pop()

    def enumerate_tiles(
        self,
        tile_shape,
        tile,
        outer_index: List[int],
        inner_index: List[int],
    ):
        dim = len(inner_index)
        if dim + 1 == len(tile_shape):
            return f"[{', '.join(f'[{c.real}, {c.imag}]' for c in tile)}]"

        local_results = []
        for i in range(0, tile_shape[dim]):
            inner_index.append(i)
            subtile = tile[i]
            local_results.append(
                self.enumerate_tiles(tile_shape, subtile, outer_index, inner_index)
            )
            inner_index.pop()

        return f"[{', '.join(t for t in local_results)}]"


class SparseMatrix(Tensor):
    def __init__(self, name: str, mat):
        super().__init__(name, mat.shape)

        tuple_counter = {s: (set(), set(), set()) for s in self.tile_shapes}
        for i, j, val in zip(*mat.indices(), mat.values()):
            if val == 0:
                continue
            for tile_shape, sets in tuple_counter.items():
                key = (int(i) // tile_shape[0], int(j) // tile_shape[1])
                # len(sets[0]) == value_count[0]
                sets[0].add(key[0])
                # len(sets[1]) == value_count[1]
                sets[1].add(key[1])
                # len(sets[2]) == num_tuples
                sets[2].add(key)

        for _, scheme in self.schemes.items():
            sets = tuple_counter[scheme.tile_shape]
            scheme.num_tuples = len(sets[2])
            scheme.value_count = (len(sets[0]), len(sets[1]))

    def gen_sql(self, results: List[str], tile_shape=None):
        pass


class Reduction(Tensor):
    @staticmethod
    def infer_shape_and_keys(output: str, input: Tuple[IndexedTerm, IndexedTerm]):
        # { subscript -> (dim, (Tensor, index)) }
        subscripts = dict()

        # [ ((Tensor, index), (Tensor, index)) ]
        join_keys = []

        pattern = ""
        for tensor in input:
            for i, subscript in enumerate(tensor.indices):
                value = subscripts.get(subscript)
                if value is None:
                    pattern += itos(len(subscripts))
                    subscripts[subscript] = (
                        tensor.tensor.shape[i],
                        (tensor.tensor, i),
                        itos(len(subscripts)),
                    )
                else:
                    assert value[0] == tensor.tensor.shape[i]
                    join_keys.append((value[1], (tensor.tensor, i)))
                    pattern += itos(value[1][1])
            pattern += f"{len(tensor.tensor.shape)}_"
        shape = tuple(subscripts[s][0] for s in output)

        # [ (Tensor, index) ]
        aggregation_keys = [subscripts.get(s)[1] for s in output]

        for s in output:
            pattern += subscripts.get(s)[2]

        return shape, join_keys, aggregation_keys, pattern

    def __init__(
        self,
        name: str,
        aggregation_op: str,
        output: str,
        input: Tuple[IndexedTerm, IndexedTerm],
        join_op="mul",
    ):
        shape, join_keys, aggregation_keys, pattern = Reduction.infer_shape_and_keys(
            output, input
        )
        super().__init__(name, shape, input[0].tensor.is_complex)
        self.join_op = join_op
        self.join_keys = join_keys
        self.aggregation_op = aggregation_op
        self.aggregation_keys = aggregation_keys
        self.pattern = pattern
        self.lhs = input[0]
        self.rhs = input[1]

        if self.join_op is None:
            self.join_op = "mul"

        for _, s in self.schemes.items():
            s.cost = float("inf")
            s.accumulated_cost = float("inf")

        # NOTE: We assume there are at most 2 input operands...
        assert len(input) == 2

        # Compute cost for each echeme
        self.lhs_key_indices = tuple([key[0][1] for key in self.join_keys])
        self.rhs_key_indices = tuple([key[1][1] for key in self.join_keys])

        lhs_groups = dict()
        for tile_shape, lhs in input[0].tensor.schemes.items():
            key = tuple([tile_shape[index] for index in self.lhs_key_indices])
            value = lhs_groups.get(key)
            if value is None:
                lhs_groups[key] = [lhs]
            else:
                value.append(lhs)

        for tile_shape, rhs in input[1].tensor.schemes.items():
            key = tuple([tile_shape[index] for index in self.rhs_key_indices])
            for lhs in lhs_groups[key]:
                self.update_cost(lhs, rhs)

    def update_cost(self, lhs: Scheme, rhs: Scheme):
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
                assert False
        tile_shape = tuple(tile_shape)
        scheme = self.schemes.get(tile_shape)

        lhs_size = (len(lhs.tile_shape) + lhs.tile_size) * lhs.num_tuples
        rhs_size = (len(rhs.tile_shape) + rhs.tile_size) * rhs.num_tuples
        # 1 is a magic number
        join_comm = 1 * (lhs_size + rhs_size)

        join_num_tuples = (
            lhs.num_tuples
            * rhs.num_tuples
            / prod(
                max(lhs.value_count[key[0][1]], rhs.value_count[key[1][1]])
                for key in self.join_keys
            )
        )

        tile_join_cost = lhs.tile_size * rhs.tile_size
        for key in self.join_keys:
            if key[0][0] is lhs.node:
                tile_join_cost /= lhs.tile_shape[key[0][1]]
            elif key[0][0] is rhs.node:
                tile_join_cost /= rhs.tile_shape[key[0][1]]
            else:
                assert False

        tile_num_group = 1.0
        for key in self.aggregation_keys:
            if key[0] is lhs.node:
                tile_num_group *= lhs.tile_shape[key[1]]
            elif key[0] is rhs.node:
                tile_num_group *= rhs.tile_shape[key[1]]
            else:
                assert False

        join_kernel_cost = 2 * tile_join_cost - tile_num_group

        join_flops = join_num_tuples * join_kernel_cost

        agg_num_tuples = min(join_num_tuples / 2, agg_num_tuples)

        # 1 is a magic number
        agg_comm = (
            1
            * (len(self.aggregation_keys) + len(self.join_keys) + scheme.tile_size)
            * join_num_tuples
        )

        agg_kernel_cost = scheme.tile_size
        agg_flops = (
            (join_num_tuples / agg_num_tuples - 1) * agg_num_tuples * agg_kernel_cost
        )

        cost = join_comm + join_flops + agg_comm + agg_flops
        # relop is expensive
        cost += join_num_tuples * 15
        cost += (join_num_tuples / agg_num_tuples - 1) * agg_num_tuples * 15

        dependencies = lhs.dependencies.union(rhs.dependencies)
        dependencies.add(lhs)
        dependencies.add(rhs)
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

    def gen_sql(self, sqlgen, tile_shape=None):
        lhs_shape = self.schemes[tile_shape].source[0].tile_shape
        rhs_shape = self.schemes[tile_shape].source[1].tile_shape
        table_name = {
            self.lhs.tensor.name: f"{self.lhs.tensor.name}_{'_'.join(str(i) for i in lhs_shape)}",
            self.rhs.tensor.name: f"{self.rhs.tensor.name}_{'_'.join(str(i) for i in rhs_shape)}",
        }

        select = [
            f"{table_name[key[0].name]}.{itos(key[1])} AS {itos(i)}"
            for i, key in enumerate(self.aggregation_keys)
        ]
        agg = self.aggregation_op
        join = f"{self.join_op}_{self.pattern}"
        if self.is_complex:
            join += "_complex"
        select.append(
            f"{agg}({join}({table_name[self.lhs.tensor.name]}.val, {table_name[self.rhs.tensor.name]}.val)) AS val"
        )

        if self.is_complex:
            sqlgen.add_join_kernel(join)
            sqlgen.add_agg_kernel(f"add_{len(self.shape) + 1}")
        else:
            sqlgen.add_taco_join_kernel(join)
            sqlgen.add_agg_kernel(f"add_{len(self.shape)}")

        from_clause = [table_name[self.lhs.tensor.name], table_name[self.rhs.tensor.name]]
        # where_clause = [
        #     f"{lhs[0].name}.{itos(lhs[1])} = {rhs[0].name}.{itos(rhs[1])}"
        #     for lhs, rhs in self.join_keys
        # ]
        where_clause = [
            f"{table_name[lhs[0].name]}.{itos(lhs[1])} = {table_name[rhs[0].name]}.{itos(rhs[1])}"
            for lhs, rhs in self.join_keys
        ]
        group_by_clause = [
            # f"{tensor.name}.{itos(i)}" for tensor, i in self.aggregation_keys
            f"{table_name[tensor.name]}.{itos(i)}" for tensor, i in self.aggregation_keys
        ]

        sql = (
            f"{self.name}_{'_'.join(str(i) for i in tile_shape)} AS ("
            f"SELECT {', '.join(col for col in select)}\n"
            f"FROM {', '.join(tensor for tensor in from_clause)}\n"
            f"WHERE {' AND '.join(cond for cond in where_clause)}\n"
            f"GROUP BY {', '.join(key for key in group_by_clause)})"
        )

        sqlgen.add_intermediate_query(sql)


def get_label_dims(einsum_string, shapes):
    label_dims = {}
    for i, subscript_labels in enumerate(einsum_string.split("->")[0].split(",")):
        for label, dimension in zip(list(subscript_labels), shapes[i]):
            if label not in label_dims:
                label_dims[label] = dimension
            elif label_dims[label] != dimension:
                raise Exception(
                    f"Dimension error for label '{label}'. "
                    f"curr={label_dims[label]}, incoming={dimension}"
                )
    return label_dims

def einsum_legacy(name: str, output: str, *argv):
    tensors = [i for i in argv]
    return Reduction(name, "SUM", output, tensors)


def einsum(output: str, *argv):
    tensors = [i for i in argv]
    # # Legacy API, require 2 IndexedTerm as input operands
    # if isinstance(tensors[0], IndexedTerm):
    #     return Reduction(name, "SUM", output, tensors)

    einsum_string = output
    label_dims = get_label_dims(einsum_string, [t.shape for t in tensors])

    views = oe.helpers.build_views(einsum_string, label_dims)
    path, path_info = oe.contract_path(einsum_string, *views)

    for i, contraction in enumerate(path_info.contraction_list):
        indices = contraction[0]
        formula = contraction[2]

        input_0 = tensors[indices[0]]
        input_1 = tensors[indices[1]]
        for index in sorted(indices, reverse=True):
            del tensors[index]

        subscript_labels = formula.split("->")[0].split(",")
        input_0 = input_0[subscript_labels[0]]
        input_1 = input_1[subscript_labels[1]]
        output = formula.split("->")[1]

        output_name = f"_t{i}"
        tensors.append(einsum_legacy(output_name, output, input_0, input_1))

    assert len(tensors) == 1
    return tensors[0]


class SQLGen:
    def __init__(self, root, prefix, tile_shape=None):
        self.prefix = prefix
        self.tables = []
        self.intermediate_queries = []
        self.join_kernels = {}
        self.taco_join_kernels = {}
        self.agg_kernels = {}

        # If tile_shape is not specified, use the scheme of the least
        # accumulated cost
        if tile_shape is None:
            working_list = [
                sorted(root.schemes.items(), key=lambda kv: kv[1].accumulated_cost)[0][
                    1
                ]
            ]
        else:
            working_list = [root.schemes[tile_shape]]

        # print(f"tile_shape: {working_list[0].tile_shape}")

        # Build adjacency list
        graph = dict()
        while len(working_list) > 0:
            scheme = working_list.pop(0)
            if scheme not in graph:
                graph[scheme] = (set(), set())
            for src in scheme.source:
                working_list.append(src)
                if src not in graph:
                    graph[src] = (set(), set())
                graph[src][1].add(scheme)
                graph[scheme][0].add(src)

        # Topological sort
        ordered_schemes = []
        while len(graph) > 0:
            scheme = None
            outs = None
            for s, (i, o) in graph.items():
                if len(i) == 0:
                    scheme = s
                    outs = o
                    break

            for out in outs:
                graph[out][0].remove(scheme)

            ordered_schemes.append(scheme)
            graph.pop(scheme)

        for scheme in ordered_schemes:
            scheme.node.gen_sql(self, scheme.tile_shape)

        with open(f"{prefix}_insert.sql", "w") as f:
            for sql in self.tables:
                f.write(sql)
                f.write("\n\n")

        with open(f"{prefix}_query.sql", "w") as f:
            for name, kernel in self.join_kernels.items():
                ndim_complex = 1 if kernel.is_complex else 0
                lhs_brk = "[]" * (len(kernel.lhs) + ndim_complex)
                rhs_brk = "[]" * (len(kernel.rhs) + ndim_complex)
                out_brk = "[]" * (len(kernel.out) + ndim_complex)
                f.write(
                    f"CREATE OR REPLACE FUNCTION {name}(double precision{lhs_brk}, double precision{rhs_brk})\n"
                    f"RETURNS double precision{out_brk} AS '/usr/lib/postgresql/12/lib/lib{prefix}_kernels.so', '{name}'\n"
                    "LANGUAGE C STRICT;\n"
                )

            for name, kernel in self.taco_join_kernels.items():
                lhs_brk = "[]" * len(kernel.lhs)
                rhs_brk = "[]" * len(kernel.rhs)
                out_brk = "[]" * len(kernel.out)
                f.write(
                    f"CREATE OR REPLACE FUNCTION {name}(double precision{lhs_brk}, double precision{rhs_brk})\n"
                    f"RETURNS double precision{out_brk} AS '/usr/lib/postgresql/12/lib/lib{prefix}_{kernel.name}.so', '{name}'\n"
                    "LANGUAGE C STRICT;\n"
                )

            for name, kernel in self.agg_kernels.items():
                brk = "[]" * kernel.ndim
                f.write(
                    f"CREATE OR REPLACE FUNCTION add_{kernel.ndim}(double precision{brk}, double precision{brk})\n"
                    f"RETURNS double precision{brk} AS '/usr/lib/postgresql/12/lib/lib{prefix}_kernels.so', '{name}'\n"
                    "LANGUAGE C STRICT;\n"
                )
                f.write(
                    f"CREATE OR REPLACE AGGREGATE SUM(double precision{brk}) (\n"
                    f"    sfunc = {name},\n"
                    f"    stype = double precision{brk}\n"
                    ");\n"
                )

            f.write("WITH\n")
            f.write(",\n\n".join(self.intermediate_queries))
            f.write("\n\n")
            f.write(f"SELECT * FROM {ordered_schemes[-1].node.name}_{'_'.join(str(i) for i in ordered_schemes[-1].tile_shape)};")

        with open(f"{prefix}_kernels.c", "w") as f:
            f.write(r"""#include "postgres.h"

#include "catalog/pg_type.h"
#include "utils/array.h"

#include "fmgr.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif


""")
            for name, kernel in self.join_kernels.items():
                f.write(kernel.codegen())

            for name, kernel in self.agg_kernels.items():
                f.write(kernel.codegen())
        
        for name, kernel in self.taco_join_kernels.items():
            kernel.codegen()
        
        with open(f"{prefix}_build.sh", "w") as f:
            f.write("set -e\n")
            f.write(f"gcc -I/usr/include/postgresql/12/server -O2 -g -DNDEBUG -fPIC -shared {prefix}_kernels.c -Wl,-soname,lib{prefix}_kernels.so -o lib{prefix}_kernels.so\n")
            f.write(f"cp lib{prefix}_kernels.so /usr/lib/postgresql/12/lib/lib{prefix}_kernels.so\n")

            for name, kernel in self.taco_join_kernels.items():
                src = f"{kernel.prefix}{kernel.name}.c"
                so = f"lib{kernel.prefix}{kernel.name}.so"
                f.write(f"gcc -I/usr/include/postgresql/12/server -O2 -g -DNDEBUG -fPIC -shared {kernel.prefix}_{kernel.name}_pg_udf.c -Wl,-soname,lib{prefix}_{kernel.name}.so -o lib{prefix}_{kernel.name}.so\n")
                f.write(f"cp lib{prefix}_{kernel.name}.so /usr/lib/postgresql/12/lib/lib{prefix}_{kernel.name}.so\n")


    def add_table(self, sql: str):
        self.tables.append(sql)

    def add_intermediate_query(self, sql: str):
        self.intermediate_queries.append(sql)

    def add_join_kernel(self, name):
        if name in self.join_kernels:
            return

        splits = name.split("_")
        assert splits[0] == "mul"

        # splits[1] and [2] should look like "ijk3", "klm3"
        subscripts = sorted(list(set(splits[1][:-1] + splits[2][:-1])))
        lhs = list(splits[1][:-1])
        rhs = list(splits[2][:-1])
        out = list(splits[3])
        is_complex = len(splits) == 5

        self.join_kernels[name] = KernelMul(name, subscripts, lhs, rhs, out, is_complex)
    
    def add_taco_join_kernel(self, name):
        if name in self.taco_join_kernels:
            return 
        
        splits = name.split("_")
        assert splits[0] == "mul"

        # splits[1] and [2] should look like "ijk3", "klm3"
        subscripts = sorted(list(set(splits[1][:-1] + splits[2][:-1])))
        lhs = list(splits[1][:-1])
        rhs = list(splits[2][:-1])
        out = list(splits[3])
        is_complex = len(splits) == 5
        assert is_complex == False

        self.taco_join_kernels[name] = TacoKernelMul(name, subscripts, lhs, rhs, out, self.prefix)


    def add_agg_kernel(self, name):
        if name in self.agg_kernels:
            return

        splits = name.split("_")
        assert splits[0] == "add"

        ndim = int(splits[1])

        self.agg_kernels[name] = KernelAdd(name, ndim)
