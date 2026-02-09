import numpy as np

import einsql

A_mod = np.array(
    [[1, 0, 1, 0],
     [0, 1, 1, 1],
     [1, 1, 1, 0],
     [0, 1, 0, 1]])

X = np.array(
    [[1, 0, 1],
     [0, 1, 0],
     [1, 1, 0],
     [0, 0, 1]])

hidden_dim = 8
W_Q = np.arange(1, 1 + hidden_dim * X.shape[1]).reshape(X.shape[1], hidden_dim)
W_K = np.arange(10, 10 + hidden_dim * X.shape[1]).reshape(X.shape[1], hidden_dim)
XW_Q = np.einsum("ij,jk->ik", X, W_Q)
XW_K = np.einsum("ij,jk->ik", X, W_K)
print("XW_Q")
print(XW_Q)

print("XW_K")
print(XW_K)
temp = np.einsum("ik,ij->ijk", XW_Q, A_mod)
print("temp")
print(temp)
attention = np.einsum("ijk,jk->ij", temp, XW_K)
print("attention")
print(attention)

print(np.einsum("nd,dh,nm,mf,fh->nm", X, W_Q, A_mod, X, W_K))

A_mod = einsql.NumpyNDArray("A_mod", A_mod)
X = einsql.NumpyNDArray("X", X)
W_Q = einsql.NumpyNDArray("W_Q", W_Q)
W_K = einsql.NumpyNDArray("W_K", W_K)

r = einsql.einsum("nd,dh,nm,mf,fh->nm", X, W_Q, A_mod, X, W_K)
schemes = sorted(r.schemes.items(), key=lambda kv: kv[1].accumulated_cost)

base_name = "gan_sparse_test"

# print the best 3 schemes
for i in range(3):
    with open(f"{base_name}_scheme_{i}.txt", "w") as f:
        printed = set()
        working_list = [schemes[i][1]]
        while len(working_list) > 0:
            scheme = working_list.pop()
            tile_shape = scheme.tile_shape

            f.write(f"{scheme.node.name}: {tile_shape}\n")
            f.write(f"\tcost: {scheme.cost}\n")
            f.write(f"\tflops: {scheme.flops}\n")
            f.write(f"\ttotal_cost: {scheme.accumulated_cost}\n")
            f.write(f"\ttotal_flops: {scheme.accumulated_flops}\n")
            for j, s in enumerate(scheme.source):
                f.write(f"\tinput[{j}]: {s.node.name} {s.tile_shape}\n")
                if s not in printed:
                    working_list.append(s)
            f.write("\n")
            printed.add(scheme)

    prefix = f"{base_name}_{i}"
    einsql.SQLGen(r, prefix, schemes[i][1].tile_shape)
