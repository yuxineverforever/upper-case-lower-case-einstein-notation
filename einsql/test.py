"""
Unit tests for the EinSQL library.

Tests cover:
- Tensor shape inference
- Sparsity pattern detection
- Cost model validation
- SQL code generation
- Topological sorting
"""

import unittest

import torch

import einsql


class TestReduction(unittest.TestCase):
    """Tests for tensor reduction and Einstein notation."""

    def test_reduction_inference(self):
        """Test that reduction correctly infers output shape and keys."""
        a = einsql.Tensor("a", (4, 8))
        b = einsql.Tensor("b", (8, 16))
        s = einsql.einsum("s", "ij", a["ik"], b["kj"])

        self.assertEqual((4, 16), s.shape)

        f1 = [1, 2, 4]
        f2 = [1, 2, 4, 8, 16]
        self.assertEqual(len(f1) * len(f2), len(s.schemes))
        self.assertTrue((a, 0) in s.aggregation_keys)
        self.assertTrue((b, 1) in s.aggregation_keys)
        self.assertTrue(((a, 1), (b, 0)) in s.join_keys)


class TestSparseTensor(unittest.TestCase):
    """Tests for sparse tensor handling."""

    def test_sparse_tensor_sparsity(self):
        """Test sparsity computation for sparse tensors."""
        a = torch.tensor([[1.0, 2, 0], [3.0, 4, 0], [5.0, 0, 0]])
        a = a.to_sparse_coo()
        a = einsql.SparseTensor("a", a)

        self.assertEqual(4, len(a.schemes))

        expected = {
            (1, 1): (5, 3, 2),
            (1, 3): (3, 3, 1),
            (3, 1): (2, 1, 2),
            (3, 3): (1, 1, 1),
        }
        for _, scheme in a.schemes.items():
            self.assertIn(scheme.tile_shape, expected)
            ans = expected[scheme.tile_shape]
            self.assertEqual(ans[0], scheme.num_tuples)
            self.assertEqual(ans[1], scheme.value_count[0])
            self.assertEqual(ans[2], scheme.value_count[1])

    def test_sparse_identity_matrix(self):
        """Test sparsity computation for identity matrix."""
        a = einsql.SparseTensor("a", torch.eye(4).to_sparse_coo())

        expected = {
            (1, 1): (4, 4, 4),
            (1, 2): (4, 4, 2),
            (1, 4): (4, 4, 1),
            (2, 1): (4, 2, 4),
            (2, 2): (2, 2, 2),
            (2, 4): (2, 2, 1),
            (4, 1): (4, 1, 4),
            (4, 2): (2, 1, 2),
            (4, 4): (1, 1, 1),
        }
        for _, scheme in a.schemes.items():
            self.assertIn(scheme.tile_shape, expected)
            ans = expected[scheme.tile_shape]
            self.assertEqual(ans[0], scheme.num_tuples)
            self.assertEqual(ans[1], scheme.value_count[0])
            self.assertEqual(ans[2], scheme.value_count[1])


class TestCostModel(unittest.TestCase):
    """Tests for the cost model optimization."""

    def test_sparse_cost_model(self):
        """Test cost model selects optimal tiling for sparse matrices."""
        a = einsql.SparseTensor("a", torch.eye(4).to_sparse_coo())
        b = einsql.SparseTensor("b", torch.eye(4).to_sparse_coo())

        s = einsql.einsum("s", "ij", a["ik"], b["kj"])

        schemes = sorted(s.schemes.items(), key=lambda kv: kv[1].accumulated_cost)
        # Smallest tiles should be cheapest for sparse data
        self.assertEqual((1, 1), schemes[0][0])
        # Largest tiles should be most expensive
        self.assertEqual((4, 4), schemes[-1][0])

    def test_dense_cost_model(self):
        """Test cost model for dense matrices."""
        a = einsql.DenseTensor("a", torch.eye(4))
        b = einsql.DenseTensor("b", torch.eye(4))

        r = einsql.einsum("r", "ij", a["ik"], b["kj"])

        schemes = sorted(r.schemes.items(), key=lambda kv: kv[1].accumulated_cost)
        self.assertEqual((1, 1), schemes[0][0])
        self.assertEqual((4, 4), schemes[-1][0])


class TestDenseTensor(unittest.TestCase):
    """Tests for dense tensor operations."""

    def test_tile_value_generation(self):
        """Test that tile values are correctly generated."""
        a = einsql.DenseTensor("a", torch.arange(0, 64).reshape((4, 4, 4)))
        results = []
        a._gen_tile_values(results, (2, 2, 2), a.data, [])
        # Verify tiles are generated (8 tiles for 4x4x4 with 2x2x2 tiling)
        self.assertEqual(8, len(results))

    def test_sparsity_detection(self):
        """Test sparsity detection in dense tensors."""
        a = torch.tensor([[1.0, 2, 0], [3.0, 4, 0], [5.0, 0, 0]])
        a = einsql.DenseTensor("a", a)

        self.assertEqual(4, len(a.schemes))

        expected = {
            (1, 1): (5, 3, 2),
            (1, 3): (3, 3, 1),
            (3, 1): (2, 1, 2),
            (3, 3): (1, 1, 1),
        }
        for _, scheme in a.schemes.items():
            self.assertIn(scheme.tile_shape, expected)
            ans = expected[scheme.tile_shape]
            self.assertEqual(ans[0], scheme.num_tuples)
            self.assertEqual(ans[1], scheme.value_count[0])
            self.assertEqual(ans[2], scheme.value_count[1])


class TestSQLGeneration(unittest.TestCase):
    """Tests for SQL code generation."""

    def test_simple_matmul(self):
        """Test SQL generation for simple matrix multiplication."""
        a = einsql.DenseTensor("a", torch.arange(0, 4).reshape((2, 2)))
        b = einsql.DenseTensor("b", torch.arange(0, 4).reshape((2, 2)))
        s = einsql.einsum("s", "ij", a["ik"], b["kj"])

        # Should not raise
        einsql.SQLGenerator(s, "test_matmul_insert.sql", "test_matmul_query.sql")

    def test_topological_sorting(self):
        """Test that operations are correctly ordered."""
        a = einsql.DenseTensor("a", torch.arange(0, 16).reshape((4, 4)))
        b = einsql.DenseTensor("b", torch.arange(0, 16).reshape((4, 4)))

        s = einsql.einsum("s", "ij", a["ik"], b["kj"])
        s1 = einsql.einsum("s1", "ij", a["ik"], s["kj"])
        s2 = einsql.einsum("s2", "ij", s1["ik"], s["kj"])
        s3 = einsql.einsum("s3", "ij", s2["ik"], b["kj"])

        # Should not raise - verifies topological sort works
        einsql.SQLGenerator(s3, "test_chain_insert.sql", "test_chain_query.sql")


class TestUtilityFunctions(unittest.TestCase):
    """Tests for utility functions."""

    def test_index_to_subscript(self):
        """Test index to subscript conversion."""
        self.assertEqual("i", einsql.index_to_subscript(0))
        self.assertEqual("j", einsql.index_to_subscript(1))
        self.assertEqual("k", einsql.index_to_subscript(2))

    def test_find_all_factors(self):
        """Test factor finding."""
        self.assertEqual([1, 2, 3, 6], einsql.find_all_factors(6))
        self.assertEqual([1, 2, 4, 8], einsql.find_all_factors(8))
        self.assertEqual([1, 5], einsql.find_all_factors(5))

    def test_index_to_subscript_bounds(self):
        """Test that index_to_subscript raises for out-of-range indices."""
        with self.assertRaises(ValueError):
            einsql.index_to_subscript(20)  # Would exceed 'z'


if __name__ == "__main__":
    unittest.main()
