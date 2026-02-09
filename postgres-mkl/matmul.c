#include "mkl.h"

#include "postgres.h"

#include "catalog/pg_type.h"
#include "utils/array.h"

#include "fmgr.h"

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

PG_FUNCTION_INFO_V1(scalar_multiply);
Datum scalar_multiply(PG_FUNCTION_ARGS) {
  // Input: Two PostgreSQL arrays (matrices A and B)
  double a = PG_GETARG_FLOAT8(0);
  ArrayType *mat_b = PG_GETARG_ARRAYTYPE_P(1);

  // Matrix dimensions
  int n = (int)ARR_DIMS(mat_b)[0];

  // Convert the PostgreSQL arrays into C arrays (double*)
  double *b = (double *)ARR_DATA_PTR(mat_b);

  // Prepare result matrix
  double *c = (double *)palloc(n * sizeof(double));

  // Call Intel MKL's matrix multiplication routine (C = A * B)

  cblas_daxpby(n, a, b, 1, 0.0, c, 1);

  // Prepare result as PostgreSQL array
  ArrayType *result =
      construct_array((Datum *)c, n, FLOAT8OID, sizeof(float8), true, 'd');

  // Free allocated memory
  pfree(c);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(matrix_multiply);
Datum matrix_multiply(PG_FUNCTION_ARGS) {
  // Input: Two PostgreSQL arrays (matrices A and B)
  ArrayType *mat_a = PG_GETARG_ARRAYTYPE_P(0);
  ArrayType *mat_b = PG_GETARG_ARRAYTYPE_P(1);

  int ndim_a = (int)ARR_NDIM(mat_a);
  // Matrix dimensions
  int m, k, n;
  if (ndim_a == 1) {
    m = 1;
    k = (int)ARR_DIMS(mat_a)[0];
  } else {
    m = (int)ARR_DIMS(mat_a)[0];
    k = (int)ARR_DIMS(mat_a)[1];
  }
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
  if (ndim_a == 1) {
    result =
        construct_array((Datum *)c, n, FLOAT8OID, sizeof(float8), true, 'd');
  } else {
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

PG_FUNCTION_INFO_V1(matrix_tanh);
Datum matrix_tanh(PG_FUNCTION_ARGS) {
  ArrayType *mat_a = PG_GETARG_ARRAYTYPE_P(0);

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

  // Prepare result matrix
  double *r = (double *)palloc(m * n * sizeof(double));

  vdTanh(m * n, a, r);

  // Prepare result as PostgreSQL array
  ArrayType *result = NULL;
  if (ndim == 1) {
    result =
        construct_array((Datum *)r, n, FLOAT8OID, sizeof(float8), true, 'd');
  } else {
    int dims[2] = {m, n};
    int lbs[2] = {1, 1};
    result = construct_md_array((Datum *)r, NULL, 2, dims, lbs, FLOAT8OID,
                                sizeof(float8), true, 'd');
  }

  // Free allocated memory
  pfree(r);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(matrix_softmax);
Datum matrix_softmax(PG_FUNCTION_ARGS) {
  ArrayType *mat_a = PG_GETARG_ARRAYTYPE_P(0);

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

  // Prepare result matrix
  double *r = (double *)palloc(m * n * sizeof(double));
  // double *exp = (double *)palloc(m * n * sizeof(double));
  vdExp(m * n, a, r);

  for (int i = 0; i < m; ++i) {
    double sum = cblas_dasum(n, &r[i * n], 1);
    cblas_daxpby(n, (double)1.0 / sum, &r[i * n], 1, 0.0, &r[i * n], 1);
  }

  // Prepare result as PostgreSQL array
  ArrayType *result = NULL;
  if (ndim == 1) {
    result =
        construct_array((Datum *)r, n, FLOAT8OID, sizeof(float8), true, 'd');
  } else {
    int dims[2] = {m, n};
    int lbs[2] = {1, 1};
    result = construct_md_array((Datum *)r, NULL, 2, dims, lbs, FLOAT8OID,
                                sizeof(float8), true, 'd');
  }

  // Free allocated memory
  pfree(r);

  // Return the resulting array
  PG_RETURN_ARRAYTYPE_P(result);
}
