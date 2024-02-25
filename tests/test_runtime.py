import unittest
import numpy as np
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from TheAlgorithm import TheAlgorithm
from main import download_data, split

class TestAlgorithmRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('Setting up class for runtime tests.')

    def setUp(self):
        print('Setup for runtime test')
        X, y = download_data()
        splitRatio = 60000
        self.X_train, self.y_train, self.X_test, self.y_test = split(X, y, splitRatio)

    def test_fit_runtime(self):
        np.random.seed(31337)
        ta = TheAlgorithm(self.X_train, self.y_train, self.X_test, self.y_test)

        # Start timing
        start_time = time.time()

        # Execute the fit() function
        ta.fit()

        # End timing
        end_time = time.time()

        # Calculate the actual duration
        actual_duration = end_time - start_time

        # Print the actual duration for visibility
        print(f"Actual runtime of fit() function: {actual_duration} seconds")

        # Define expected maximum runtime = 120% of a baseline runtime
        # Baseline runtime = 20 sec
        baseline_runtime = 20  #  baseline runtime in seconds
        runtime_limit = baseline_runtime * 1.2  # 120% of the baseline

        # Assert that the actual duration does not exceed the limit
        self.assertTrue(actual_duration <= runtime_limit, f"fit() function runtime exceeded the 120% limit of the baseline: {actual_duration} seconds; Limit was {runtime_limit} seconds")

if __name__ == '__main__':
    unittest.main()
