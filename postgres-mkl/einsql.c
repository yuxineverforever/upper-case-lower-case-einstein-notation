#include "mkl.h"

#include "postgres.h"

#include "catalog/pg_type.h"
#include "utils/array.h"

#include "fmgr.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

// ij,jk -> ik is equivalent to matrix multiply
PG_FUNCTION_INFO_V1(mul_ij2_jk2_ik);
Datum mul_ij2_jk2_ik(PG_FUNCTION_ARGS) {
  // Input: Two PostgreSQL arrays (matrices A and B)
  ArrayType *mat_a = PG_GETARG_ARRAYTYPE_P(0);
  ArrayType *mat_b = PG_GETARG_ARRAYTYPE_P(1);

  // Matrix dimensions
  int m, k, n;
  m = (int)ARR_DIMS(mat_a)[0];
  k = (int)ARR_DIMS(mat_a)[1];
  n = (int)ARR_DIMS(mat_b)[1];

  // Convert the PostgreSQL arrays into C arrays (double*)
  double *a = (double *)ARR_DATA_PTR(mat_a);
  double *b = (double *)ARR_DATA_PTR(mat_b);

  // Prepare result matrix
  double *c = (double *)palloc(m * n * sizeof(double));

  // Call Intel MKL's matrix multiplication routine (C = A * B)
  // cblas_dgemm: C = alpha*A*B + beta*C
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b,
              n, 0.0, c, n);

  // Prepare result as PostgreSQL array
  ArrayType *result = NULL;
  int dims[2] = {m, n};
  int lbs[2] = {1, 1};
  result = construct_md_array((Datum *)c, NULL, 2, dims, lbs, FLOAT8OID,
                              sizeof(float8), true, 'd');

  // Free allocated memory
  pfree(c);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(mul_ij2_j1_i_complex);
Datum mul_ij2_j1_i_complex(PG_FUNCTION_ARGS) {
  // Input: Two PostgreSQL arrays (matrices A and B)
  ArrayType *tensor_l = PG_GETARG_ARRAYTYPE_P(0);
  ArrayType *tensor_r = PG_GETARG_ARRAYTYPE_P(1);

  // Matrix dimensions
  int ni = (int)ARR_DIMS(tensor_l)[0];
  int nj = (int)ARR_DIMS(tensor_l)[1];

  // Convert the PostgreSQL arrays into C arrays (double*)
  double *l = (double *)ARR_DATA_PTR(tensor_l);
  double *r = (double *)ARR_DATA_PTR(tensor_r);

  // Prepare result matrix
  double *res = (double *)palloc(ni * 2 * sizeof(double));
  memset(res, 0, ni * 2 * sizeof(double));

  for (int i = 0; i < ni; ++i) {
    for (int j = 0; j < nj; ++j) {
      double a = l[i * nj * 2 + j * 2];
      double b = l[i * nj * 2 + j * 2 + 1];
      double c = r[j * 2];
      double d = r[j * 2 + 1];

      res[i * 2] += a * c - b * d;
      res[i * 2 + 1] += a * d + b * c;
    }
  }

  // Prepare result as PostgreSQL array
  ArrayType *result = NULL;
  int dims[2] = {ni, 2};
  int lbs[2] = {1, 1};
  result = construct_md_array((Datum *)res, NULL, 2, dims, lbs, FLOAT8OID,
                              sizeof(float8), true, 'd');

  // Free allocated memory
  pfree(res);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}

#if 0
PG_FUNCTION_INFO_V1(mul_ij2_jk2_ik_complex);
Datum mul_ij2_jk2_ik_complex(PG_FUNCTION_ARGS) {
  // Input: Two PostgreSQL arrays (matrices A and B)
  ArrayType *tensor_l = PG_GETARG_ARRAYTYPE_P(0);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(tensor_l);
}
#endif

#if 0
PG_FUNCTION_INFO_V1(mul_ij2_jk2_ik_complex);
Datum mul_ij2_jk2_ik_complex(PG_FUNCTION_ARGS) {
  // Input: Two PostgreSQL arrays (matrices A and B)
  ArrayType *tensor_l = PG_GETARG_ARRAYTYPE_P(0);
  ArrayType *tensor_r = PG_GETARG_ARRAYTYPE_P(1);

  // Matrix dimensions
  int ni = (int)ARR_DIMS(tensor_l)[0];
  int nj = (int)ARR_DIMS(tensor_l)[1];
  int nk = (int)ARR_DIMS(tensor_r)[1];

  double *res = (double *)palloc(ni * nk * 2 * sizeof(double));
  memset(res, 0, ni * nk * 2 * sizeof(double));
  // Prepare result as PostgreSQL array
  ArrayType *result = NULL;
  int dims[3] = {ni, nk, 2};
  int lbs[3] = {1, 1, 1};
  result = construct_md_array((Datum *)res, NULL, 3, dims, lbs, FLOAT8OID,
                              sizeof(float8), true, 'd');

  // Free allocated memory
  pfree(res);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}
#endif

#if 1
PG_FUNCTION_INFO_V1(mul_ij2_jk2_ik_complex);
Datum mul_ij2_jk2_ik_complex(PG_FUNCTION_ARGS) {
  // Input: Two PostgreSQL arrays (matrices A and B)
  ArrayType *tensor_l = PG_GETARG_ARRAYTYPE_P(0);
  ArrayType *tensor_r = PG_GETARG_ARRAYTYPE_P(1);

  // Matrix dimensions
  int ni = (int)ARR_DIMS(tensor_l)[0];
  int nj = (int)ARR_DIMS(tensor_l)[1];
  int nk = (int)ARR_DIMS(tensor_r)[1];

  // Convert the PostgreSQL arrays into C arrays (double*)
  double *l = (double *)ARR_DATA_PTR(tensor_l);
  double *r = (double *)ARR_DATA_PTR(tensor_r);

  // Prepare result matrix
  double *res = (double *)palloc(ni * nk * 2 * sizeof(double));
  memset(res, 0, ni * nk * 2 * sizeof(double));

  for (int i = 0; i < ni; ++i) {
    for (int j = 0; j < nj; ++j) {
      for (int k = 0; k < nk; ++k) {
        double a = l[i * (nj * 2) + j * 2];
        double b = l[i * (nj * 2) + j * 2 + 1];
        double c = r[j * (nk * 2) + k * 2];
        double d = r[j * (nk * 2) + k * 2 + 1];

        res[i * nk * 2 + k * 2] += a * c - b * d;
        res[i * nk * 2 + k * 2 + 1] += a * d + b * c;
      }
    }
  }

  // Prepare result as PostgreSQL array
  ArrayType *result = NULL;
  int dims[3] = {ni, nk, 2};
  int lbs[3] = {1, 1, 1};
  result = construct_md_array((Datum *)res, NULL, 3, dims, lbs, FLOAT8OID,
                              sizeof(float8), true, 'd');

  // Free allocated memory
  pfree(res);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}
#endif

PG_FUNCTION_INFO_V1(mul_ijk3_klm3_ijlm_complex);
Datum mul_ijk3_klm3_ijlm_complex(PG_FUNCTION_ARGS) {
  // Input: Two PostgreSQL arrays (matrices A and B)
  ArrayType *tensor_l = PG_GETARG_ARRAYTYPE_P(0);
  ArrayType *tensor_r = PG_GETARG_ARRAYTYPE_P(1);

  // Matrix dimensions
  int ni = (int)ARR_DIMS(tensor_l)[0];
  int nj = (int)ARR_DIMS(tensor_l)[1];
  int nk = (int)ARR_DIMS(tensor_l)[2];
  int nl = (int)ARR_DIMS(tensor_r)[1];
  int nm = (int)ARR_DIMS(tensor_r)[2];

  // Convert the PostgreSQL arrays into C arrays (double*)
  double *lhs = (double *)ARR_DATA_PTR(tensor_l);
  double *rhs = (double *)ARR_DATA_PTR(tensor_r);

  // Prepare result matrix
  double *res = (double *)palloc(ni * nj * nl * nm * 2 * sizeof(double));
  memset(res, 0, ni * nj * nl * nm * 2 * sizeof(double));

  for (int i = 0; i < ni; ++i) {
    for (int j = 0; j < nj; ++j) {
      for (int k = 0; k < nk; ++k) {
        for (int l = 0; l < nl; ++l) {
          for (int m = 0; m < nm; ++m) {
            double a = lhs[i * nj * nk * 2 + j * nk * 2 + k * 2];
            double b = lhs[i * nj * nk * 2 + j * nk * 2 + k * 2 + 1];
            double c = rhs[k * nl * nm * 2 + l * nm * 2 + m * 2];
            double d = rhs[k * nl * nm * 2 + l * nm * 2 + m * 2 + 1];

            res[i * nj * nl * nm * 2 + j * nl * nm * 2 + l * nm * 2 + m * 2] +=
                a * c - b * d;
            res[i * nj * nl * nm * 2 + j * nl * nm * 2 + l * nm * 2 + m * 2 +
                1] += a * d + b * c;
          }
        }
      }
    }
  }

  // Prepare result as PostgreSQL array
  ArrayType *result = NULL;
  int dims[5] = {ni, nj, nl, nm, 2};
  int lbs[5] = {1, 1, 1, 1};
  result = construct_md_array((Datum *)res, NULL, 5, dims, lbs, FLOAT8OID,
                              sizeof(float8), true, 'd');

  // Free allocated memory
  pfree(res);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(mul_ijk3_jkl3_il_complex);
Datum mul_ijk3_jkl3_il_complex(PG_FUNCTION_ARGS) {
  // Input: Two PostgreSQL arrays (matrices A and B)
  ArrayType *tensor_l = PG_GETARG_ARRAYTYPE_P(0);
  ArrayType *tensor_r = PG_GETARG_ARRAYTYPE_P(1);

  // Matrix dimensions
  int ni = (int)ARR_DIMS(tensor_l)[0];
  int nj = (int)ARR_DIMS(tensor_l)[1];
  int nk = (int)ARR_DIMS(tensor_l)[2];
  int nl = (int)ARR_DIMS(tensor_r)[2];

  // Convert the PostgreSQL arrays into C arrays (double*)
  double *lhs = (double *)ARR_DATA_PTR(tensor_l);
  double *rhs = (double *)ARR_DATA_PTR(tensor_r);

  // Prepare result matrix
  double *res = (double *)palloc(ni * nl * 2 * sizeof(double));
  memset(res, 0, ni * nl * 2 * sizeof(double));

  for (int i = 0; i < ni; ++i) {
    for (int j = 0; j < nj; ++j) {
      for (int k = 0; k < nk; ++k) {
        for (int l = 0; l < nl; ++l) {
          double a = lhs[i * nj * nk * 2 + j * nk * 2 + k * 2];
          double b = lhs[i * nj * nk * 2 + j * nk * 2 + k * 2 + 1];
          double c = rhs[j * nk * nl * 2 + k * nl * 2 + l * 2];
          double d = rhs[j * nk * nl * 2 + k * nl * 2 + l * 2 + 1];

          res[i * nl * 2 + l * 2] += a * c - b * d;
          res[i * nl * 2 + l * 2 + 1] += a * d + b * c;
        }
      }
    }
  }

  // Prepare result as PostgreSQL array
  ArrayType *result = NULL;
  int dims[3] = {ni, nl, 2};
  int lbs[3] = {1, 1, 1};
  result = construct_md_array((Datum *)res, NULL, 3, dims, lbs, FLOAT8OID,
                              sizeof(float8), true, 'd');

  // Free allocated memory
  pfree(res);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(add_2);
Datum add_2(PG_FUNCTION_ARGS) {
  ArrayType *tensor_l = PG_GETARG_ARRAYTYPE_P(0);
  ArrayType *tensor_r = PG_GETARG_ARRAYTYPE_P(1);

  int ni = (int)ARR_DIMS(tensor_l)[0];
  int nj = (int)ARR_DIMS(tensor_l)[1];

  // Convert the PostgreSQL arrays into C arrays (double*)
  double *l = (double *)ARR_DATA_PTR(tensor_l);
  double *r = (double *)ARR_DATA_PTR(tensor_r);

  // Prepare result matrix
  double *res = (double *)palloc(ni * nj * sizeof(double));
  memset(res, 0, ni * nj * sizeof(double));
  // mkl_domatadd('R', 'N', 'N', m, n, 1.0, l, n, 1.0, r, n, res, n);
  for (int i = 0; i < ni; ++i) {
    for (int j = 0; j < nj; ++j) {
      res[i * nj + j] = l[i * nj + j] + r[i * nj + j];
    }
  }

  ArrayType *result = NULL;
  int dims[2] = {ni, nj};
  int lbs[2] = {1, 1};
  result = construct_md_array((Datum *)res, NULL, 2, dims, lbs, FLOAT8OID,
                              sizeof(float8), true, 'd');

  // Free allocated memory
  pfree(res);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}

#if 0
PG_FUNCTION_INFO_V1(add_3);
Datum add_3(PG_FUNCTION_ARGS) {
  ArrayType *tensor_l = PG_GETARG_ARRAYTYPE_P(0);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(tensor_l);
}
#endif

#if 0
PG_FUNCTION_INFO_V1(add_3);
Datum add_3(PG_FUNCTION_ARGS) {
  ArrayType *tensor_l = PG_GETARG_ARRAYTYPE_P(0);
  ArrayType *tensor_r = PG_GETARG_ARRAYTYPE_P(1);

  int ni = (int)ARR_DIMS(tensor_l)[0];
  int nj = (int)ARR_DIMS(tensor_l)[1];
  int nk = (int)ARR_DIMS(tensor_l)[2];

  // Convert the PostgreSQL arrays into C arrays (double*)
  double *l = (double *)ARR_DATA_PTR(tensor_l);
  double *r = (double *)ARR_DATA_PTR(tensor_r);

  // Prepare result matrix
  double *res = (double *)palloc(ni * nj * nk * sizeof(double));
  memset(res, 0, ni * nj * nk * sizeof(double));

  ArrayType *result = NULL;
  int dims[3] = {ni, nj, nk};
  int lbs[3] = {1, 1};
  result = construct_md_array((Datum *)res, NULL, 3, dims, lbs, FLOAT8OID,
                              sizeof(float8), true, 'd');

  // Free allocated memory
  pfree(res);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}
#endif

#if 1
PG_FUNCTION_INFO_V1(add_3);
Datum add_3(PG_FUNCTION_ARGS) {
  ArrayType *tensor_l = PG_GETARG_ARRAYTYPE_P(0);
  ArrayType *tensor_r = PG_GETARG_ARRAYTYPE_P(1);

  int ni = (int)ARR_DIMS(tensor_l)[0];
  int nj = (int)ARR_DIMS(tensor_l)[1];
  int nk = (int)ARR_DIMS(tensor_l)[2];

  // Convert the PostgreSQL arrays into C arrays (double*)
  double *l = (double *)ARR_DATA_PTR(tensor_l);
  double *r = (double *)ARR_DATA_PTR(tensor_r);

  // Prepare result matrix
  double *res = (double *)palloc(ni * nj * nk * sizeof(double));
  memset(res, 0, ni * nj * nk * sizeof(double));
  // mkl_domatadd('R', 'N', 'N', m, n, 1.0, l, n, 1.0, r, n, res, n);
  for (int i = 0; i < ni; ++i) {
    for (int j = 0; j < nj; ++j) {
      for (int k = 0; k < nk; ++k) {
        res[i * (nj * nk) + j * nk + nk] =
            l[i * nj * nk + j * nk + nk] + r[i * nj * nk + j * nk + nk];
      }
    }
  }

  ArrayType *result = NULL;
  int dims[3] = {ni, nj, nk};
  int lbs[3] = {1, 1};
  result = construct_md_array((Datum *)res, NULL, 3, dims, lbs, FLOAT8OID,
                              sizeof(float8), true, 'd');

  // Free allocated memory
  pfree(res);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}
#endif

PG_FUNCTION_INFO_V1(add_5);
Datum add_5(PG_FUNCTION_ARGS) {
  ArrayType *tensor_l = PG_GETARG_ARRAYTYPE_P(0);
  ArrayType *tensor_r = PG_GETARG_ARRAYTYPE_P(1);

  int ni = (int)ARR_DIMS(tensor_l)[0];
  int nj = (int)ARR_DIMS(tensor_l)[1];
  int nk = (int)ARR_DIMS(tensor_l)[2];
  int nl = (int)ARR_DIMS(tensor_l)[3];
  int nm = (int)ARR_DIMS(tensor_l)[4];

  // Convert the PostgreSQL arrays into C arrays (double*)
  double *lhs = (double *)ARR_DATA_PTR(tensor_l);
  double *rhs = (double *)ARR_DATA_PTR(tensor_r);

  // Prepare result matrix
  double *res = (double *)palloc(ni * nj * nk * nl * nm * sizeof(double));
  memset(res, 0, ni * nj * nk * nl * nm * sizeof(double));
  // mkl_domatadd('R', 'N', 'N', m, n, 1.0, l, n, 1.0, r, n, res, n);
  for (int i = 0; i < ni; ++i) {
    for (int j = 0; j < nj; ++j) {
      for (int k = 0; k < nk; ++k) {
        for (int l = 0; l < nl; ++l) {
          for (int m = 0; m < nm; ++m) {
            res[i * nj * nk * nl * nm + j * nk * nl * nm + k * nl * nm +
                l * nm + m] = lhs[i * nj * nk * nl * nm + j * nk * nl * nm +
                                  k * nl * nm + l * nm + m] +
                              rhs[i * nj * nk * nl * nm + j * nk * nl * nm +
                                  k * nl * nm + l * nm + m];
          }
        }
      }
    }
  }

  ArrayType *result = NULL;
  int dims[5] = {ni, nj, nk, nl, nm};
  int lbs[5] = {1, 1, 1, 1, 1};
  result = construct_md_array((Datum *)res, NULL, 5, dims, lbs, FLOAT8OID,
                              sizeof(float8), true, 'd');

  // Free allocated memory
  pfree(res);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(matrix_add);
Datum matrix_add(PG_FUNCTION_ARGS) {
  ArrayType *mat_a = PG_GETARG_ARRAYTYPE_P(0);
  ArrayType *mat_b = PG_GETARG_ARRAYTYPE_P(1);

  int ndim = (int)ARR_NDIM(mat_a);
  int m, n;
  if (ndim == 1) {
    m = 1;
    n = (int)ARR_DIMS(mat_a)[0];
  } else {
    m = (int)ARR_DIMS(mat_a)[0];
    n = (int)ARR_DIMS(mat_a)[1];
  }

  // Convert the PostgreSQL arrays into C arrays (double*)
  double *a = (double *)ARR_DATA_PTR(mat_a);
  double *b = (double *)ARR_DATA_PTR(mat_b);

  // Prepare result matrix
  double *c = (double *)palloc(m * n * sizeof(double));

  mkl_domatadd('R', 'N', 'N', m, n, 1.0, a, n, 1.0, b, n, c, n);

  ArrayType *result = NULL;
  if (ndim == 1) {
    result =
        construct_array((Datum *)c, n, FLOAT8OID, sizeof(float8), true, 'd');
  } else if (ndim == 2) {
    // Prepare result as PostgreSQL array
    int dims[2] = {m, n};
    int lbs[2] = {1, 1};
    result = construct_md_array((Datum *)c, NULL, 2, dims, lbs, FLOAT8OID,
                                sizeof(float8), true, 'd');
  }

  // Free allocated memory
  pfree(c);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}
