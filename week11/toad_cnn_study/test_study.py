"""Scientific invariants and leakage guards for the revised study."""
import unittest
import numpy as np
import tensorflow as tf
from toad_simulator import simulate_toads, lag_summaries
from toad_study import (StudyConfig, simulate_group_bank, make_group_pairs,
                        distribution_mmd, singleton_kernel_distance,
                        fit_preprocessing, transform, build_encoder)


class ScientificChecks(unittest.TestCase):
    def test_no_return_models_agree(self):
        theta = np.array([1.7, 34., 0., 200.])
        paths = [simulate_toads(m, theta, 4, 12, np.random.default_rng(7)) for m in range(3)]
        for path in paths[1:]: np.testing.assert_allclose(path, paths[0])

    def test_constant_return_goes_to_origin(self):
        theta = np.array([1.7, 34., 1., 200.])
        for m in (0, 1):
            path = simulate_toads(m, theta, 4, 12, np.random.default_rng(7))
            np.testing.assert_array_equal(path, 0)

    def test_summaries_and_toad_order(self):
        path = np.arange(20)[:, None] * np.arange(1, 11)[None, :] * 20
        actual = lag_summaries(path)
        np.testing.assert_allclose(actual, lag_summaries(path[:, ::-1]))
        self.assertEqual(actual.shape, (4, 12))
        distances = np.abs(np.diff(path, axis=0)).ravel()
        expected = np.log(np.diff(np.quantile(distances, np.linspace(0, 1, 11))))
        np.testing.assert_allclose(actual[0, 2:], expected, rtol=1e-6)
        self.assertEqual(actual[0, 1], np.median(distances))

    def test_splits_pairs_and_training_only_scaler(self):
        cfg = StudyConfig(num_toads=4, num_days=12, evaluation_replicates=4)
        train = simulate_group_bank(cfg, 4, 1, 1)
        val = simulate_group_bank(cfg, 4, 2, 4)
        self.assertFalse(set(train['group_id']) & set(val['group_id']))
        a, b = make_group_pairs(train['model'], 4)
        self.assertTrue(np.all(a != b))
        np.testing.assert_array_equal(train['model'][a], train['model'][b])
        self.assertTrue(np.all(np.any(train['theta'][a, :3] != train['theta'][b, :3], axis=1)))
        self.assertEqual(len(a), 6)
        self.assertEqual(len(np.unique(np.r_[a, b])), 12)
        centre, scale = fit_preprocessing(train['raw'])
        before = centre.copy()
        self.assertTrue(np.isfinite(transform(val['raw'], centre, scale)).all())
        np.testing.assert_array_equal(centre, before)

    def test_mmd_sample_axis_and_scale_invariance(self):
        rng = np.random.default_rng(9)
        x = tf.constant(rng.normal(size=(2, 20, 4)), tf.float32)
        y = x + 4
        np.testing.assert_allclose(distribution_mmd(x, x), 0, atol=1e-6)
        self.assertTrue(np.all(distribution_mmd(x, y).numpy() > .1))
        np.testing.assert_allclose(distribution_mmd(x, y), distribution_mmd(x*.001, y*.001), atol=1e-5)
        one = x[:, :, :1]
        self.assertEqual(tuple(distribution_mmd(one, one).shape), (2,))
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = tf.reduce_mean(distribution_mmd(x, y))
        gradient = tape.gradient(loss, x).numpy()
        self.assertTrue(np.isfinite(gradient).all())
        self.assertGreater(np.abs(gradient).sum(), 0)

    def test_exact_clara_pair_count(self):
        models = np.repeat(np.arange(3), (33_334, 33_334, 33_332))
        left, right = make_group_pairs(models, 3991)
        self.assertEqual(len(left), 50_000)
        self.assertEqual(len(np.unique(np.r_[left, right])), 100_000)
        np.testing.assert_array_equal(models[left], models[right])
        x = tf.constant([[0.0, 0.0], [1.0, 1.0]])
        y = tf.constant([[0.0, 0.0], [3.0, 3.0]])
        distance = singleton_kernel_distance(x, y).numpy()
        self.assertAlmostEqual(distance[0], 0.0)
        self.assertGreater(distance[1], 0.0)

    def test_bottleneck_dimensions(self):
        for architecture in ('CNN', 'FFNN'):
            for d in (1, 2, 4, 8):
                encoder = build_encoder(architecture, d)
                output = encoder(np.ones((2, 4, 12), dtype='float32'))
                self.assertEqual(tuple(output['summary'].shape), (2, d))
                np.testing.assert_allclose(output['probabilities'].numpy().sum(1), 1, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
