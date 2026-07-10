import unittest


try:
    import torch

    from sam3.perflib.fused import addmm_act
except ModuleNotFoundError:
    torch = None
    addmm_act = None


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TestPerflibFused(unittest.TestCase):
    def test_addmm_act_preserves_linear_weight_dtype_for_following_linear(self):
        fc1 = torch.nn.Linear(4, 8).eval()
        fc2 = torch.nn.Linear(8, 2).eval()
        x = torch.randn(3, 4, dtype=torch.float32)

        with torch.inference_mode():
            hidden = addmm_act(torch.nn.GELU, fc1, x)
            self.assertEqual(hidden.dtype, fc1.weight.dtype)
            out = fc2(hidden)

        self.assertEqual(out.dtype, fc2.weight.dtype)


if __name__ == "__main__":
    unittest.main()
