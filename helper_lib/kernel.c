#include <stdio.h>
#include <stdlib.h>

#include "kernel.h"

char *OclLoadKernel(const char *path)
{
    FILE *kernel_file;
    char *kernel_source;
    long kernel_size;

    kernel_file = fopen(path, "r");
    if (!kernel_file) // Error opening file
        return NULL;

    // Get kernel size
    if (fseek(kernel_file, 0L, SEEK_END) != 0)
        return NULL;

    kernel_size = ftell(kernel_file) + 1; // null terminator
    rewind(kernel_file);                  // Reset file position

    kernel_source = (char *)malloc(kernel_size);
    if (!kernel_size) // Not enough host memory
        return NULL;

    fread(kernel_source, 1, kernel_size, kernel_file);
    kernel_source[kernel_size - 1] = '\0'; // Add null terminator

    fclose(kernel_file);
    return kernel_source;
}