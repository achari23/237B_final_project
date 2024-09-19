#include <assert.h>
#include <ctype.h>
#include <stdbool.h>
#include <string.h>

#include <stdlib.h>

#include "device.h"

/**
 * @brief Reads an OpenCL platform's property.
 * The caller is responsible for freeing *val.
 *
 * @param platform_id The target OpenCL platform.
 * @param prop_name The OpenCL Platform property to read.
 * @param val The destination pointer for the OpenCL Platform property.
 *
 * @return CL_SUCCESS if and only if the platform property read is successful.
 */
static cl_int OclGetProperty(const cl_platform_id platform_id, const cl_uint prop_name,
                             const char **val)
{
    cl_int status;
    size_t prop_size;

    status = clGetPlatformInfo(platform_id, prop_name, 0, NULL, &prop_size);
    if (status != CL_SUCCESS)
        return status;

    char *temp_val = (char *)malloc(prop_size);
    if (!temp_val)
        return CL_OUT_OF_HOST_MEMORY;

    status = clGetPlatformInfo(platform_id, prop_name, prop_size, temp_val, NULL);
    if (status != CL_SUCCESS)
        return status;

    *val = temp_val;

    return CL_SUCCESS;
}

/**
 * @brief Reads an OpenCL Devices's property.
 * The caller is responsible for freeing *val.
 *
 * @param device_id The target OpenCL device.
 * @param param The OpenCL Device property to read.
 * @param val The destination pointer for the OpenCL Device property.
 *
 * @return CL_SUCCESS if and only if the platform property read is successful.
 */
static cl_int OclGetInfo(const cl_device_id device_id, const cl_device_info param,
                         const void **val)
{
    cl_int status;
    size_t param_size;

    status = clGetDeviceInfo(device_id, param, 0, NULL, &param_size);
    if (status != CL_SUCCESS)
        return status;

    void *temp_val = (void *)malloc(param_size);
    if (!temp_val)
        return CL_OUT_OF_HOST_MEMORY;

    status = clGetDeviceInfo(device_id, param, param_size, temp_val, NULL);
    if (status != CL_SUCCESS)
        return status;

    *val = temp_val;

    return CL_SUCCESS;
}

const char *OclDeviceTypeString(cl_device_type type)
{
    switch (type)
    {
    case CL_DEVICE_TYPE_CPU:
        return "CPU";
    case CL_DEVICE_TYPE_GPU:
        return "GPU";
    case CL_DEVICE_TYPE_ACCELERATOR:
        return "Accelerator";
    case CL_DEVICE_TYPE_CUSTOM:
        return "Custom";
    default:
        return "Default";
    }
}

cl_int OclFindDevices(const cl_platform_id platform_id, const OclDeviceProp **devices,
                      cl_uint *num_devices)
{
    cl_uint num_found_devices;
    cl_device_id *device_ids;
    cl_int status;

    // Find number of OpenCL devices
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_found_devices);
    if (status != CL_SUCCESS)
        return status;

    *num_devices = num_found_devices;

    // Exit early if no devices found
    if (num_found_devices == 0)
        return CL_SUCCESS;

    device_ids = (cl_device_id *)malloc(num_found_devices * sizeof(cl_device_id));
    if (!device_ids)
        return CL_OUT_OF_HOST_MEMORY;

    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_found_devices,
                            device_ids, NULL);
    if (status != CL_SUCCESS)
        return status;

    OclDeviceProp *temp_devices =
        (OclDeviceProp *)malloc(num_found_devices * sizeof(OclDeviceProp));
    if (!temp_devices)
        return CL_OUT_OF_HOST_MEMORY;

    for (cl_uint i = 0; i < num_found_devices; i++)
    {
        temp_devices[i].device_id = device_ids[i];

        status = OclGetInfo(device_ids[i], CL_DEVICE_NAME,
                            (const void **)&temp_devices[i].name);
        if (status != CL_SUCCESS)
            return status;
        status = OclGetInfo(device_ids[i], CL_DEVICE_TYPE,
                            (const void **)&temp_devices[i].type);
        if (status != CL_SUCCESS)
            return status;
        status = OclGetInfo(device_ids[i], CL_DEVICE_MAX_COMPUTE_UNITS,
                            (const void **)&temp_devices[i].max_compute_units);
        if (status != CL_SUCCESS)
            return status;
        status = OclGetInfo(device_ids[i], CL_DEVICE_GLOBAL_MEM_SIZE,
                            (const void **)&temp_devices[i].global_mem_size);
        if (status != CL_SUCCESS)
            return status;
        status = OclGetInfo(device_ids[i], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                            (const void **)&temp_devices[i].max_constant_buffer_size);
        if (status != CL_SUCCESS)
            return status;
        status = OclGetInfo(device_ids[i], CL_DEVICE_LOCAL_MEM_SIZE,
                            (const void **)&temp_devices[i].local_mem_size);
        if (status != CL_SUCCESS)
            return status;
        status = OclGetInfo(device_ids[i], CL_DEVICE_MAX_WORK_ITEM_SIZES,
                            (const void **)&temp_devices[i].max_work_item_sizes);
        if (status != CL_SUCCESS)
            return status;
        status = OclGetInfo(device_ids[i], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            (const void **)&temp_devices[i].max_work_group_size);
        if (status != CL_SUCCESS)
            return status;
        status = OclGetInfo(device_ids[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                            (const void **)&temp_devices[i].max_work_item_dimensions);
        if (status != CL_SUCCESS)
            return status;
    }

    *devices = temp_devices;

    return CL_SUCCESS;
}

cl_int OclFindPlatforms(const OclPlatformProp **platforms, cl_uint *num_platforms)
{
    cl_uint num_found_platforms;
    cl_platform_id *platform_ids;
    cl_int status;

    // Find number of OpenCL platforms
    status = clGetPlatformIDs(0, NULL, &num_found_platforms);
    if (status != CL_SUCCESS)
        return status;

    *num_platforms = num_found_platforms;

    // Exit early if no platforms found
    if (num_found_platforms == 0)
        return CL_SUCCESS;

    platform_ids = (cl_platform_id *)malloc(num_found_platforms * sizeof(cl_platform_id));
    if (!platform_ids)
        return CL_OUT_OF_HOST_MEMORY;

    status = clGetPlatformIDs(num_found_platforms, platform_ids, NULL);
    if (status != CL_SUCCESS)
        return status;

    OclPlatformProp *temp_platforms =
        (OclPlatformProp *)malloc(num_found_platforms * sizeof(OclPlatformProp));
    if (!temp_platforms)
        return CL_OUT_OF_HOST_MEMORY;

    for (cl_uint i = 0; i < num_found_platforms; i++)
    {
        temp_platforms[i].platform_id = platform_ids[i];

        status = OclGetProperty(platform_ids[i], CL_PLATFORM_NAME,
                                (const char **)(&temp_platforms[i].name));
        if (status != CL_SUCCESS)
            return status;
        status = OclGetProperty(platform_ids[i], CL_PLATFORM_VERSION,
                                (const char **)(&temp_platforms[i].version));
        if (status != CL_SUCCESS)
            return status;
        status = OclGetProperty(platform_ids[i], CL_PLATFORM_PROFILE,
                                (const char **)(&temp_platforms[i].profile));
        if (status != CL_SUCCESS)
            return status;
        status = OclGetProperty(platform_ids[i], CL_PLATFORM_VENDOR,
                                (const char **)(&temp_platforms[i].vendor));
        if (status != CL_SUCCESS)
            return status;
        status = OclGetProperty(platform_ids[i], CL_PLATFORM_EXTENSIONS,
                                (const char **)(&temp_platforms[i].extensions));
        if (status != CL_SUCCESS)
            return status;

        status = OclFindDevices(platform_ids[i],
                                (const OclDeviceProp **)&temp_platforms[i].devices,
                                &temp_platforms[i].num_devices);
        if (status != CL_SUCCESS)
            return status;
    }

    *platforms = temp_platforms;

    return CL_SUCCESS;
}

cl_int OclFreeDeviceProp(OclDeviceProp *device)
{
    free(device->name);
    free(device->max_compute_units);
    free(device->global_mem_size);
    free(device->max_constant_buffer_size);
    free(device->local_mem_size);
    free(device->max_work_item_sizes);
    free(device->max_work_group_size);
    free(device->max_work_item_dimensions);

    return CL_SUCCESS;
}

cl_int OclFreePlatformProp(OclPlatformProp *platform)
{
    free(platform->name);
    free(platform->version);
    free(platform->profile);
    free(platform->vendor);
    free(platform->extensions);
    for (unsigned int i = 0; i < platform->num_devices; i++)
    {
        OclFreeDeviceProp(&platform->devices[i]);
    }
    free(platform->devices);

    return CL_SUCCESS;
}