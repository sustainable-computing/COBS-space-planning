import random

import numpy as np
import sklearn


def generate_uniform_around_centers(centers, variance):

    """
    The generate_uniform_around_centers function takes in a list of centers and a variance.
    It then randomly selects one of the centers from the list, adds to it a random number between
    -variance and +variance, and returns that new point.

    :param centers: Define the centers of the clusters
    :param variance: Control the spread of the points around each center
    :return: An array of two numbers
    """
    num_center = len(centers)

    return centers[np.random.choice(num_center)] + variance * np.random.uniform(-1, 1, 2)


def generate_cross(centers, variance):
    """
    The generate_cross function takes in two parameters, centers and variance.
    The centers parameter is a list of lists containing the x and y coordinates for each center point.
    The variance parameter is a float that determines how far away from the center points we want our cross to be.
    This function returns an array with 2 elements, which are the x and y coordinates of one point on our cross.

    :param centers: Generate the centers of the cross
    :param variance: Control the range of the random number
    :return: A point that is a linear combination of two points in the centers array
    """
    num_center = len(centers)
    x = variance * np.random.uniform(-1, 1)
    y = (np.random.randint(2) * 2 - 1) * x

    return centers[np.random.choice(num_center)] + [x, y]


def sample_data(dataset, batch_size, scale, var):

    """
    The sample_data function is used to generate data for the various datasets that we will be using.
    The function takes in three arguments: dataset, batch_size, and scale. The dataset argument specifies which
    dataset to use (e.g., "25gaussians", "swissroll", etc.). The batch_size argument specifies
    how many samples to return at a time from the generator (i.e., how many points are in each minibatch). Finally,
    the scale argument is used as a scaling factor for some of our datasets.

    :param dataset: Select the dataset to be used
    :param batch_size: Determine the number of samples to draw from the dataset
    :param scale: Scale the centers of the gaussians
    :param var: Control the variance of the gaussian distribution
    :return: A generator that can be used to generate batches of data
    """
    if dataset == "25gaussians":

        dataset = []
        for i in range(100000 // 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        np.random.shuffle(dataset)
        # dataset /= 2.828 # stdev
        while True:
            for i in range(len(dataset) / batch_size):
                yield dataset[i * batch_size: (i + 1) * batch_size]

    elif dataset == "swissroll":

        while True:
            data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=0.25)[0]
            data = data.astype("float32")[:, [0, 2]]
            # data /= 7.5 # stdev plus a little
            yield data

    elif dataset == "8gaussians":

        scale = scale
        variance = var
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for i in range(batch_size):
                point = np.random.randn(2) * variance
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "checker_board_five":

        scale = scale
        variance = var
        centers = scale * np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]])
        while True:
            dataset = []
            for i in range(batch_size):
                dataset.append(generate_uniform_around_centers(centers, variance))
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "checker_board_four":

        scale = scale
        variance = var
        centers = scale * np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        while True:
            dataset = []
            for i in range(batch_size):
                dataset.append(generate_uniform_around_centers(centers, variance))
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "simpleGaussian":

        while True:
            dataset = []
            for i in range(batch_size):
                point = np.random.randn(2)
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "unif_square":

        while True:
            dataset = []
            for i in range(batch_size):
                point = np.random.uniform(-var, var, 2)
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "simpletranslatedGaussian":

        while True:
            dataset = []
            for i in range(batch_size):
                point = scale * np.array([1.0, 1.0]) + np.random.randn(2)
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "simpletranslated_scaled_Gaussian":

        while True:
            dataset = []
            for i in range(batch_size):
                point = scale * np.array([1.0, 1.0]) + var * np.random.randn(2)
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "circle-S1":

        while True:
            dataset = []
            for i in range(batch_size):
                angle = np.random.rand() * 2 * np.pi
                point = scale * np.array([np.cos(angle), np.sin(angle)])
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            yield dataset

    elif dataset == "semi-circle-S1":

        while True:
            dataset = []
            for i in range(batch_size):
                angle = np.random.rand() * np.pi
                point = scale * np.array([np.cos(angle), np.sin(angle)])
                dataset.append(point)
            dataset = np.array(dataset, dtype="float32")
            yield dataset

    elif dataset == "checker_board_five_cross":

        scale = scale
        variance = var
        centers = scale * np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]])
        while True:
            dataset = []
            for i in range(batch_size):
                dataset.append(generate_cross(centers, variance))
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset

    elif dataset == "checker_board_five_expanded":

        scale = scale
        variance = 2 * var
        centers = scale * np.array([[0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]])
        while True:
            dataset = []
            for i in range(batch_size):
                dataset.append(generate_uniform_around_centers(centers, variance))
            dataset = np.array(dataset, dtype="float32")
            # dataset /= 1.414 # stdev
            yield dataset
