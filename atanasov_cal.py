import numpy as np 

def atanasov_calibration_method(ps: np.ndarray):
    """
    Nikolay's method for laser calibration.
    Inputs:
     - ps: the laser points
    Output: the 5-vector of the laser parameters, with the first 3 being the orientation,
            and the final two being the x and y coordinates of the laser origin
    """
    avg_alpha = np.zeros((3,))
    params = np.zeros((5,))
    for i in range(ps.shape[0]):
        for j in range(ps.shape[0]):
            if i != j:
                v = ps[i] - ps[j]
                if v[2] < 0:
                    v = -v
                avg_alpha += v

    avg_alpha /= np.linalg.norm(avg_alpha)
    if avg_alpha[2] < 0:
        avg_alpha = -avg_alpha

    centroid = np.mean(ps, axis=0)
    scale_factor = centroid[2] / avg_alpha[2]
    params[0:3] = avg_alpha
    params[3:5] = centroid[0:2] - scale_factor * avg_alpha[0:2]
    return params

if __name__ == '__main__': 
    test_arr = np.array([[0,1,2], [4,5,6], [0,3,2], [0,4,5]])
    print(atanasov_calibration_method(test_arr))