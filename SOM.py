
import simpsom as sps

from simpsom.plots import scatter_on_map
from pylettes import *


class SOMSimpsomInterface:
    """
    Wrapper class around SimpSOM providing an interface
    for training and visualizing a Self-Organizing Map (SOM).

    Supports:
    - Training the SOM (fit)
    - Visualization (class map)
    """

    def __init__(self, 
                 rows=20, cols=20, n_iterations=1000, learning_rate=0.5, sigma=None, 
                 neighborhood_function='gaussian', topology='rectangular', metric='cosine'
                ):
        """
        Initialize the SimpSOM wrapper.

        Parameters
        ----------
        rows : int
            Number of rows in the SOM map.
        cols : int
            Number of columns in the SOM map.
        n_iterations : int
            Number of training iterations.
        learning_rate : float
            Initial learning rate.
        sigma : float or None
            Initial neighborhood radius (if None, defaults to max(rows, cols) / 2).
        neighborhood_function : str
            Type of neighborhood kernel: 'gaussian', 'bubble', 'mexican_hat', etc.
        topology : str
            SOM topology, e.g. 'rectangular' or 'hexagonal'.
        metric : str
            Distance metric used in training (e.g., 'euclidean', 'cosine').
        """
        # Store SOM configuration parameters
        self.rows = rows
        self.cols = cols
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.sigma = sigma or max(rows, cols) / 2
        self.neighborhood_function = neighborhood_function
        self.metric = metric
        self.topology = topology
        self.net = None  # SimpSOM network object

    # ----------------------------------------------------------------------
    def fit(self, X, verbose=True):
        """
        Train the SOM using SimpSOM.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        verbose : bool
            If True, print training progress to console.
        """

        # Build and initialize the SOM network
        self.net = sps.SOMNet(
            self.cols, self.rows, X,
            init='PCA', # initialize weights using PCA
            metric=self.metric, # distance metric for training
            debug=True, # enable SimpSOM debug output
            neighborhood_fun=self.neighborhood_function, # neighborhood kernel
            topology=self.topology # map topology
        )

        if verbose:
            print(f"Training SOM ({self.cols}x{self.rows}) for {self.n_iterations} iterations...")

        # Train the SOM
        self.net.train(start_learning_rate=self.learning_rate, epochs=self.n_iterations, train_algo='batch', batch_size=-1)

        if verbose:
            print("Training complete.")

    def visualise(self, X, y):
        """
        Visualize the low-dimensional projection of data on the trained SOM.

        Parameters
        ----------
        X : np.ndarray
            Standardized input data (n_samples, n_features).
        y : np.ndarray
            Class labels (n_samples,).
        """

        # Project input data onto the SOM map (returns 2D coordinates)
        projection = self.net.project_onto_map(X)

        # Plot results — each class is displayed in a separate color
        scatter_on_map(
            [projection[y==i] for i in range(10)], # separate points by class 
            [[node.pos[0], node.pos[1]] for node in self.net.nodes_list], # neuron positions
            self.net.polygons, color_val=None,
            show=True, print_out=True, cmap=Tundra(reverse=True).cmap
        )
