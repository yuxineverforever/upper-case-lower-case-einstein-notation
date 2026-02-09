import torch_geometric.datasets as datasets
import scipy.sparse as sp
import numpy as np

import einsql

for dataset_name in ["Cora", "CiteSeer"]:
    dataset = datasets.Planetoid(root="data", name=dataset_name)

    # Unpack the data
    edges = dataset.edge_index  # Edge index (2 x E)

    # Convert edges to a sparse adjacency matrix
    N = dataset.x.shape[0]
    D = dataset.x.shape[1]
    E = dataset.edge_index.shape[1]
    C = dataset.num_classes

    A = sp.csr_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(N, N))

    A_mod = sp.coo_matrix(A + sp.eye(N))

    X = dataset.x.detach().cpu().numpy()  # Node features (N x D)
    y = dataset.y.detach().cpu().numpy()  # Node labels (N x 1)

    print(f"Planetoid.{dataset_name}")
    print(f"\tnum_nodes   N: {N}")
    print(f"\tnum_edges   E: {E}")
    print(f"\tnum_classes C: {C}")
    print(f"\tadjacent matrix A: {A_mod.shape}")
    print(f"\tnode features   X: {X.shape}")

    hidden_dim = 1024

    Attention = np.zeros((N, N))
    F = X.shape[1]
    W_Q = np.random.randn(F, hidden_dim)
    W_K = np.random.randn(F, hidden_dim)

    A_mod = einsql.NumpyNDArray("A_mod", A_mod.toarray())
    X = einsql.NumpyNDArray("X", X)
    W_Q = einsql.NumpyNDArray("W_Q", W_Q)
    W_K = einsql.NumpyNDArray("W_K", W_K)

    XW_Q = einsql.einsum_legacy("XW_Q", "ij", X["ik"], W_Q["kj"])
    XW_K = einsql.einsum_legacy("XW_K", "ij", X["ik"], W_K["kj"])

    temp = einsql.einsum_legacy("temp", "ijk", XW_Q["ik"], A_mod["ij"])
    r = einsql.einsum_legacy("attention", "ij", temp["ijk"], XW_K["jk"])
    schemes = sorted(r.schemes.items(), key=lambda kv: kv[1].accumulated_cost)

    base_name = f"gan_sparse_planetoid_{dataset_name}"

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
