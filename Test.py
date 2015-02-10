"""
Unittesting for tempotron
"""


import unittest
import numpy as np
from Tempotron import Tempotron


class TestTempotron(unittest.TestCase):
    """
    Test the functionality of the Tempotron class.
    """
    def setUp(self):
        # Change synaptic efficacies later!!
        self.tempotron = Tempotron(0, 10, 2.5, np.ones(10))
        pass

    def test_normalisation_tempotron(self):
        """
        Test if the tempotron normalisation is
        computed correctly.
        """
        V_computed = self.tempotron.V_norm
        V_man = 2.116534735957599
        self.assertEqual(V_computed, V_man)

    def test_membrane_potential0(self):
        """
        Test for tempotron.compute_membrane_potential
        for spike_times = {}
        """
        self.tempotron.efficacies = np.random.random(10) - 0.5
        spike_times = [set([])] * 10
        V = self.tempotron.compute_membrane_potential(10, spike_times)
        self.assertEqual(V, 0.0)

    def test_membrane_potential1(self):
        """
        Test 2 for tempotron.compute_membrane_potential
        for non empty spike_times
        """
        self.tempotron.efficacies = np.random.random(10)
        spike_times = np.array([[0], [0], [0], [], [], [], [], [], [], []])
        potential = self.tempotron.compute_membrane_potential(4.62, spike_times)
        potential_man = self.tempotron.efficacies[0:3].sum()
        self.assertAlmostEqual(potential, potential_man)

    def test_spike_contributions1(self):
        """
        Test 1 for tempotron.compute_spike_contributions
        Every neuron spikes once at a different time
        """
        spike_times = np.array([[0], [10], [20], [30], [40]])
        spike_contribs = self.tempotron.compute_spike_contributions(40, spike_times)
        spike_contribs_correct = np.array([0.03877, 0.1054, 0.2857, 0.7399, 0.0])
        self.assertTrue(np.allclose(spike_contribs, spike_contribs_correct, atol=1e-4))

    def test_spike_contributions2(self):
        """
        Test 2 for tempotron.compute_spike_contributions
        One neuron spikes twice
        """
        spike_times = np.array([[0, 10], [], [], [], []])
        spike_contribs = self.tempotron.compute_spike_contributions(20, spike_times)
        spike_contribs_correct = np.array([1.025596, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(spike_contribs, spike_contribs_correct, atol=1e-4))

    def test_compute_tmax1(self):
        """
        Test for tempotron.compute_tmax
        Only one spike at t=10
        """
        self.tempotron.efficacies = np.random.random(10)
        spike_times = ([[10], [], [], [], [], [], [], [], [], []])

        # maximum occurs 4.6 ms after spike
        self.assertAlmostEqual(self.tempotron.compute_tmax(spike_times), 14.62098120373297, places=7)

    def test_compute_tmax2(self):
        """
        Test for tempotron.compute_tmax
        Multiple spikes at a single time
        """
        self.tempotron.efficacies = np.random.random(10)
        spike_times = ([[10], [10], [10], [10], [], [], [], [], [], []])

        # maximum occurs 4.6 ms after spike
        self.assertAlmostEqual(self.tempotron.compute_tmax(spike_times), 14.62098120373297, places=7)

    def test_compute_tmax3(self):
        """
        Test for tempotron.compute_tmax
        Boundary case with non-existing derivative
        """
        tempotron1 = Tempotron(0, 15, 15/4, np.array([2, -1]))
        spike_times = np.array([[np.log(2)*15, 100], [np.log(3)*15, 101]])

        self.assertAlmostEqual(tempotron1.compute_tmax(spike_times), 16.43259988)


if __name__ == '__main__':
    unittest.main()
