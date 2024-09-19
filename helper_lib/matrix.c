#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrix.h"

cl_int LoadMatrix(const char *path, Matrix *matrix)
{
    FILE *data_file;

    data_file = fopen(path, "r");
    if (!data_file) // Error opening file
        return CL_INVALID_VALUE;

    unsigned int rows = 0;
    unsigned int cols = 0;
    if (fscanf(data_file, "# (%u, %u)\n", &rows, &cols) == EOF)
        return CL_INVALID_VALUE; // Error parsing dimensions

    if (rows == 0)
        rows = 1;
    if (cols == 0)
        cols = 1;

    matrix->shape[0] = rows;
    matrix->shape[1] = cols;

    matrix->data = malloc(sizeof(float) * rows * cols);
    if (!matrix->data) // Error mallocing matrix data
        return CL_OUT_OF_HOST_MEMORY;

    int n = 0;
    while (fscanf(data_file, "%f", &(matrix->data[n++])) != EOF)
        ;

    fclose(data_file);

    return CL_SUCCESS;
}

cl_int SaveMatrix(const char *path, Matrix *matrix)
{
    FILE *data_file;

    data_file = fopen(path, "w");
    if (!data_file) // Error opening file
        return CL_INVALID_VALUE;

    unsigned int rows = matrix->shape[0];
    unsigned int cols = matrix->shape[1];
    if (fprintf(data_file, "# (%u, %u)\n", rows, cols) < 0)
        return CL_INVALID_VALUE; // Error parsing dimensions
    
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            if (fprintf(data_file, "%.2f ", roundf(matrix->data[cols * r + c] * 100) / 100) < -1)
                return CL_INVALID_VALUE; // Error parsing dimensions
        }
        fprintf(data_file, "\n");
    }

    fclose(data_file);

    return CL_SUCCESS;
}

cl_int CheckMatrix(Matrix *truth, Matrix *student)
{
    if (truth->shape[0] != student->shape[0] || truth->shape[1] != student->shape[1])
    {
        printf("!!SOLUTION IS NOT CORRECT!!\n");
        return CL_INVALID_VALUE;
    }

    int count = truth->shape[0] * truth->shape[1];
    for (int i = 0; i < count; i++)
    {
        float epsilon = fabs(truth->data[i]) * 0.1f;
        float diff = fabs(truth->data[i] - student->data[i]);
        if (diff > epsilon)
        {
            printf("!!SOLUTION IS NOT CORRECT!!\n");
            return CL_INVALID_VALUE;
        }
    }

    printf("!!SOLUTION IS CORRECT!!\n");
    return CL_SUCCESS;
}

void PrintMatrix(Matrix *matrix)
{
    int rows, cols;
    rows = matrix->shape[0];
    cols = matrix->shape[1];

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            printf("%.2f ", roundf(matrix->data[r * cols + c] * 100) / 100);
        }
        printf("\n");
    }
}
