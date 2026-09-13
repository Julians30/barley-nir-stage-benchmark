"""Unit fixtures only; not experimental observations or benchmark results."""
import unittest
from types import SimpleNamespace
import numpy as np
from run_review_additions import vip


class VIPTests(unittest.TestCase):
    def test_closed_form_and_mean_square(self):
        model = SimpleNamespace(x_scores_=np.array([[1.,0.],[0.,1.]]),
            x_weights_=np.array([[1.,0.],[0.,1.],[0.,0.]]), y_loadings_=np.array([[2.,1.]]))
        np.testing.assert_allclose(vip(model), np.sqrt([12/5,3/5,0]))
        np.testing.assert_allclose(np.mean(vip(model)**2),1)

    def test_component_weight_normalization(self):
        model = SimpleNamespace(x_scores_=np.eye(2), x_weights_=np.diag([3.,7.]),
            y_loadings_=np.ones((1,2)))
        np.testing.assert_allclose(vip(model),np.ones(2))

    def test_zero_explained_response_rejected(self):
        model = SimpleNamespace(x_scores_=np.eye(2), x_weights_=np.eye(2),
            y_loadings_=np.zeros((1,2)))
        with self.assertRaises(ValueError):
            vip(model)


if __name__ == '__main__':
    unittest.main()
