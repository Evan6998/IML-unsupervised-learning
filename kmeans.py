import numpy as np
import argparse

# import matplotlib.pyplot as plt
# NOTE: you shouldn't need to use any other packages for this homework

#  DO NOT CHANGE THIS LINE
#########################
np.random.seed(10301601)
#########################

def kmpp_init(X, K):
    """ Perform K-Means++ Cluster Initialization.

        Input:
        X: a numpy ndarray with shape (N,M), where each row is a data point
        K: an int where K is the number of cluster centers 

        Output:
        C: a numpy ndarray with shape (K,M), where each row is a cluster center
    """

    # TODO: Initialize cluster centers for KMPP

    # Start by randomly initializing the first cluster center. 
    # We've already one this for you below.

    N = X.shape[0]
    C = []
    C.append(X[np.random.randint(N)])
    
    return 

def kmeans_loss(X, C, z):
    """ Compute the K-means loss.

        Input:
        X: a numpy ndarray with shape (N,M), where each row is a data point
        C: a numpy ndarray with shape (K,M), where each row is a cluster center
        z: a numpy ndarray with shape (N,) where the i-th entry is an int from {0..K-1}
            representing the cluster index for the i-th point in X

        Returns mean squared distance from each point to the center for its assigned cluster
    """

    # TODO: calculate the k-means loss
    
    return 

def kmeans(X, K: int, algo=0):
    """ Cluster data X into K converged clusters.
    
        X: an N-by-M numpy ndarray, where we want to assign each
            of the N data points to a cluster.

        K: an integer denoting the number of clusters.

        Returns a tuple of length two containing (C, z):
            C: a numpy ndarray with shape (K,M), where each row is an M-dimensional cluster center
            z: a numpy ndarray with shape (N,) where the i-th entry is an int from {0..K-1}
                representing the cluster index for the i-th point in X
    """
    N = X.shape[0]

    # Initialize K cluster centers based on the type of initialization. 
    # We gave you the random initialization below. **DO NOT CHANGE IT** 
    # otherwise we cannot guarantee that your solution will work with the autograder

    C = X[np.random.choice(N, size=K, replace=False)]
    if algo == 1:
        C = kmpp_init(X, K)

    # z is a vector represents the center for each data points. shape: (N, 1)
    z = np.zeros(N)
    
    # Write the k-means algorithm below
    while True:
        stacked_X = np.stack([X] * K, axis=1)
        distance = np.sum(np.square(stacked_X - C), axis=2)
        
        new_z = np.argmin(distance, axis=1)
        if np.all(new_z == z):
            break
        z = new_z
        
        C = np.array([np.mean(X[new_z == k], axis=0) for k in np.unique(new_z)])

    return C, z


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get K, you can use `args.K`.
    # You should not need to modify the main function
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("K", type=int, help='number of cluster centers')
    parser.add_argument("algorithm", type=int, choices=[0,1], help='algorithm; 0 = K-Means with random init, 1 = K-Means++')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')

    args = parser.parse_args()

    train_data = np.loadtxt(args.train_input, dtype=np.float32, delimiter = ',')

    K = args.K
    algo = args.algorithm

    C, z = kmeans(train_data, K, algo)

    np.savetxt(args.train_out, z, delimiter=",")


    # TODO: uncomment the following code to graph your cluster centers after you pass the autograder

    # algo_name = "rand" if algo==0 else "kmpp"
    # figures_directory = f'figures/{K}/{algo_name}'

    # os.makedirs(figures_directory, exist_ok=True)
    
    # for k in range(K):
    #     plt.imshow(C[k].reshape((28,28)))
    #     plt.savefig(f"{figures_directory}/kmeans_K_{K}_cluster_{k}_init_{algo}")
    #     plt.clf()
