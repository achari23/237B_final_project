
//#include <math.h>
__kernel void atanasov(
    __global float * inputData, __global float * outputData, const unsigned int num_vectors, const unsigned int num_inputs){
    //@@ Insert code to implement matrix multiplication here

    //get id for given laser
    int laser_id = get_global_id(0); 

    __local float output[5];
    __local float avg_alpha[3];
    __local float v[3];
    int input_shape = num_vectors; 
    for (int i = 0; i< input_shape; i++) {
        for (int j = 0; j < input_shape; j++ ) {
            if (i != j) {
                //compute vector distance
                for (int k = 0; k< 3; k++) {
                    v[k] = inputData[laser_id*num_vectors*3 + i + k] - inputData[laser_id*num_vectors*3 + j + k];
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
   
    float norm = 0;
    for(int k = 0; k < 3; k++) {
        norm += avg_alpha[k] * avg_alpha[k]; 
    }
    norm = sqrt(norm);

    //normalize avg_alpha
    
    for (int k = 0; k< 3; k++) {
        avg_alpha[k] /= norm;
    }
    
    //if z component negative invert whole vector
    if (avg_alpha[2] < 0)  {
        for (int k = 0; k< 3; k++) {
            avg_alpha[k] = -1 * avg_alpha[k];
        }
    }
    //now we need to find the mean accross the columns of the input_data
    __local float centroid[3];
    for(int i = 0; i< 3; i++) {
        for (int k = 0; k < num_vectors; k++)  {
            centroid[i] += inputData[laser_id*num_vectors*3 + k*3 + i];
        }
        centroid[i] /= num_vectors; 
    }
    
    float scale_factor = centroid[2] / avg_alpha[2];
    for (int i = 0; i < 3; i++) {
        output[i] = avg_alpha[i];
    }
    output[3] = centroid[0] - scale_factor * avg_alpha[0];
    output[4] = centroid[1] - scale_factor * avg_alpha[1];

    for (int i = 0; i < 5; i++) {
        outputData[i + 5*laser_id] = output[i]; 
    }
    barrier(CLK_LOCAL_MEM_FENCE); 
  
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
