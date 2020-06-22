from itertools import product
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment

NN_ALGS = ['knn', 'hungarian']

def nearest_neighbors(point_cloud_A, point_cloud_B, alg='knn'):
    """Find the nearest (Euclidean) neighbor in point_cloud_B (model) for each
    point in point_cloud_A (data).

    Parameters
    ----------
    point_cloud_A: Nx3 numpy array
        data points
    point_cloud_B: Mx3 numpy array
        model points

    Returns
    -------
    distances: (N, ) numpy array
        Euclidean distances from each point in
        point_cloud_A to its nearest neighbor in point_cloud_B.
    indices: (N, ) numpy array 
        indices in point_cloud_B of each
        point_cloud_A point's nearest neighbor - these are the c_i's
    """
    assert 3 == point_cloud_A.shape[1] and 3 == point_cloud_B.shape[1]
    n, m = point_cloud_A.shape[0], point_cloud_B.shape[0]
    assert n == m
    distances = np.zeros(n)
    indices = np.zeros(n)

    if alg == 'knn':
        nbrs = NearestNeighbors(n_neighbors=1).fit(point_cloud_B)
        d, ids = nbrs.kneighbors(point_cloud_A)
        distances = np.array(d).flatten()
        indices = np.array(ids).flatten()
    elif alg == 'hungarian':
        cost = np.zeros((n, m))
        for i, j in product(range(n), range(m)):
            cost[i,j] = norm(point_cloud_A[i,:]- point_cloud_B[j,:])
        row_ids, indices = linear_sum_assignment(cost)
        distances = cost[row_ids, indices]
    else:
        raise NotImplementedError('NN algorithm must be one of: {}'.format(NN_ALGS))
    
    return distances, indices


def least_squares_transform(point_cloud_A, point_cloud_B):
    """Calculates the least-squares best-fit transform that maps corresponding
    points point_cloud_A (data) to point_cloud_B (model).

    Parameters
    ----------
    point_cloud_A: Nx3 numpy array
        corresponding data (scene) points
    point_cloud_B: Nx3 numpy array
        corresponding model points

    Returns
    -------
    X_BA: 4x4 numpy array
        the homogeneous transformation matrix that maps
        point_cloud_A on to point_cloud_B such that

            X_BA x point_cloud_Ah ~= point_cloud_B,

        where point_cloud_Ah is a homogeneous version of point_cloud_A
    """
    # dimension
    m = 3
    X_BA = np.identity(4)

    # compute center of mass
    # make sure A and B have the same amount of points
    # assert(point_cloud_A.shape[0] == point_cloud_B.shape[0])
    Ns = point_cloud_A.shape[0]

    # data center
    mu_s = (1.0/Ns) * np.sum(point_cloud_A, axis=0)
    # model center
    mu_m = (1.0/Ns) * np.sum(point_cloud_B, axis=0)
    
    # construct W
    W = np.zeros([m,m])
    for i in range(0, Ns):
        # W += np.outer(point_cloud_B[i,:] - mu_m, point_cloud_A[i,:] - mu_s)
        W += np.outer(point_cloud_A[i,:] - mu_s, point_cloud_B[i,:] - mu_m)
    
    u, _, vh = np.linalg.svd(W, full_matrices=True)
    v = vh.T
    uh = u.T
    R = v.dot(np.diag([1., 1., np.linalg.det(v.dot(uh))]).dot(uh))

    t = mu_m.T - R.dot(mu_s.T)
    
    X_BA[0:m, 0:m] = R
    X_BA[0:m, -1] = t

    return X_BA


def icp(point_cloud_A, point_cloud_B,
        init_guess=None, max_iterations=20, tolerance=1e-3, nn_alg='knn'):
    """The Iterative Closest Point algorithm: finds best-fit transform that maps
        point_cloud_A(data) on to point_cloud_B(model)

    Parameters
    ----------
    point_cloud_A: Nx3 numpy array or a list of 3-element lists
        data points to be matched onto point_cloud_B
    point_cloud_B: Nx3 numpy array
        model points
    init_guess: 4x4 numpy array or a list of 3-element lists
        homogeneous transformation representing an initial guess
        of the transform. If one isn't provided, the 4x4 identity matrix
        will be used.
    max_iterations: int
        if the algorithm hasn't converged after max_iterations, 
        exit the algorithm
    tolerance: float
        the maximum difference in the error between two
        consecutive iterations before stopping
    
    Returns
    -------
    X_BA: 4x4 numpy array
        the homogeneous transformation matrix that
        maps point_cloud_A on to point_cloud_B such that

                    X_BA x point_cloud_Ah ~= point_cloud_B,

        where point_cloud_Ah is a homogeneous version of point_cloud_A
    mean_error: float
        mean of the Euclidean distances from each point in
        the transformed point_cloud_A to its nearest neighbor in
        point_cloud_B
    num_iters: int
        total number of iterations run
    """
    if type(point_cloud_A) != np.ndarray:
        point_cloud_A = np.array(point_cloud_A)
    if type(point_cloud_B) != np.ndarray:
        point_cloud_B = np.array(point_cloud_B)

    # Transform from point_cloud_B to point_cloud_A
    # Overwrite this with ICP results.
    X_BA = np.identity(4)

    mean_error = 0
    num_iters = 0

    # Number of dimensions
    dim = 3

    # Make homogeneous copies of boht point clouds
    point_cloud_Ah = np.ones((4, point_cloud_A.shape[0]))
    point_cloud_Bh = np.ones((4, point_cloud_B.shape[0]))
    point_cloud_Ah[:dim, :] = np.copy(point_cloud_A.T)
    point_cloud_Bh[:dim, :] = np.copy(point_cloud_B.T)

    # assert(point_cloud_A.shape[0] == point_cloud_B.shape[0])
    Ns = point_cloud_A.shape[0]
    
    if np.any(init_guess):
        X_BA = init_guess
    
    while True:
        if num_iters >= max_iterations:
            # print("ICP ends exceeding max_iterations.")
            break
        
        # use c to solve R and t
        point_cloud_Ah_new = X_BA.dot(point_cloud_Ah)

        # given R and t, calculate c
        indices = nearest_neighbors(point_cloud_Ah_new.T[:,0:dim], point_cloud_B, alg=nn_alg)[1]
        point_cloud_B_c = np.copy(point_cloud_B[indices])
        
        # transf for next iteration
        X_BA = least_squares_transform(point_cloud_A, point_cloud_B_c)
        
        # check error
        old_mean_error = mean_error
        mean_error = (1.0/Ns) * (np.linalg.norm(point_cloud_Ah_new - point_cloud_Bh[:,indices])**2)        
        if abs(mean_error-old_mean_error)<tolerance:
            break
            
        num_iters += 1

    return X_BA, mean_error, num_iters


def repeat_icp_until_good_fit(point_cloud_A,
                              point_cloud_B,
                              error_threshold,
                              max_tries,
                              init_guess=None,
                              max_iterations=20,
                              tolerance=0.001, nn_alg='knn'):
    """Run ICP until it converges to a "good" fit.

    Parameters
    ----------
    point_cloud_A: Nx3 numpy array
        data points to match to point_cloud_B
    point_cloud_B: Nx3 numpy array
        model points
    error_threshold: float
        maximum allowed mean ICP error before stopping
    max_tries: int
        stop running ICP after max_tries if it hasn't produced
        a transform with an error < error_threshold.
    init_guess: 4x4 numpy array 
        homogeneous transformation representing an initial guess
        of the transform. If one isn't provided, the 4x4 identity matrix
        will be used.
    max_iterations: int
        if the algorithm hasn't converged after
        max_iterations, exit the algorithm
    tolerance: float
        the maximum difference in the error between two
        consecutive iterations before stopping

    Returns
    -------
    X_BA: 4x4 numpy array
        the homogeneous transformation matrix that
        maps point_cloud_A on to point_cloud_B such that

                    X_BA x point_cloud_Ah ~= point_cloud_B,

        where point_cloud_Ah is a homogeneous version of point_cloud_A
    mean_error: float
        mean of the Euclidean distances from each point in
        the transformed point_cloud_A to its nearest neighbor in
        point_cloud_B
    num_runs: int
        total number of times ICP ran - not the total number of
        ICP iterations.
    """
    # Transform from point_cloud_B to point_cloud_A
    # Overwrite this with ICP results.
    X_BA = np.identity(4)
    
    if np.any(init_guess)!=True:
        init_guess=np.identity(4)

    mean_error = 1e8
    num_runs = 0
    transf_dict = {}
    
    while True:
        if num_runs >= max_tries:
            # print("repeat_ICP exceeds max_iterations, exit.")
            break
        X_BA, mean_error, num_iters = \
            icp(point_cloud_A, point_cloud_B, init_guess, max_iterations, tolerance, nn_alg)
        
        transf_dict[mean_error] = X_BA
        # print("iter %d, mean error %0.8f, inside iters %d \n"%(num_runs, mean_error, num_iters))
        
        init_guess[:3, :3] = Rotation.random().as_matrix()
        if mean_error < error_threshold:
            break
        num_runs += 1

    s_keys = sorted(transf_dict.keys())
    mean_error = s_keys[0]
    X_BA = transf_dict[mean_error]
    
    return X_BA, mean_error, num_runs