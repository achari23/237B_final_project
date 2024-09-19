
__kernel void convolution2D(
    __global float * inputData, __global float * outputData){
    //@@ Insert code to implement matrix multiplication here

    //get id for given laser
    int laser_id = get_global_id(0); 

    //instance local arrays to = 0
    __local float avg_alpha[3] = {0,0,0};
    __local float v[3] = {0,0,0};

    input_shape = inputData[laser_id].shape[0];
    for (int i = 0; i< input_shape; i++) {
        for (int j = 0; j < input_shape; j++ ) {
            if (i != j) {
                //compute vector distance
                for (int k = 0; k< 3; k++) {
                    v[k] = inputData[laser_id][i][k] - inputData[laser_id][j][k];
                }
                //invert vector if z is negative
                if (v[2] < 0)  {
                    for (int k = 0; k< 3; k++) {
                        v[k] = -1 * v[k];
                    }
                }
                //add vector to average alpha
                for (int k = 0; k< 3; k++) {
                    avg_alpha[k] += v[k];
                }
            }
        }
    }
    flaot norm = 0;
    for(int k = 0; k < 3; k++) {
        norm += avg_alpha[k] * avg_alpha[k]; 
    }
    
  
}
 /**
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
    */
