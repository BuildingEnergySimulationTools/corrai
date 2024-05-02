import numpy as np


class Sampler:
    def __init__(self):
        self.samples = None

    def add_samples(self, new_samples):
        """
        Add new samples to the existing sample set.

        Parameters:
        - new_samples (numpy.ndarray or list): New samples to add. Each row
            represents a sample.
        """
        if self.samples is None:
            self.samples = new_samples
        else:
            self.samples = np.vstack([self.samples, new_samples])

    def get_samples(self):
        """
        Get the current sample set.

        Returns:
        - samples (numpy.ndarray or None): The current sample set.
        """
        return self.samples

    def clear_samples(self):
        """
        Clear the current sample set.
        """
        self.samples = None
