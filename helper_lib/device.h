#pragma once

#include <assert.h>
#include <ctype.h>
#include <stdbool.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#define CL_TARGET_OPENCL_VERSION 300 // Use OpenCL 3.0
#include <CL/cl.h>
#endif

/**
 * @brief Struct for storing OpenCL Device information.
 * All device parameters can be found here:
 * https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceInfo.html
 */
typedef struct _OclDeviceProp
{
    char *name;
    cl_device_type *type;
    cl_uint *max_compute_units;
    cl_ulong *global_mem_size;
    cl_ulong *max_constant_buffer_size;
    cl_ulong *local_mem_size;
    size_t *max_work_item_sizes;
    size_t *max_work_group_size;
    cl_uint *max_work_item_dimensions;
    cl_device_id device_id;
} OclDeviceProp;

/**
 * @brief Struct for storing OpenCL Platform and Device information.
 * All platform properties can be found here:
 * https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetPlatformInfo.html
 */
typedef struct _OclPlatformProp
{
    char *name;
    char *version;
    char *profile;
    char *vendor;
    char *extensions;
    cl_platform_id platform_id;
    cl_uint num_devices;
    OclDeviceProp *devices;
} OclPlatformProp;

/**
 * @brief Finds all OpenCL platforms and devices, and get their respective properties.
 * Internally calls OclFindDevices.
 * The caller is responsible for freeing *platforms.
 *
 * @param platforms The array of OpenCL Platform properties.
 * @param num_platforms The number of OpenCL Platforms found.
 *
 * @return CL_SUCCESS if and only if platforms and platform property reads are successful.
 */
cl_int OclFindPlatforms(const OclPlatformProp **platforms, cl_uint *num_platforms);

/**
 * @brief Finds all OpenCL devices on a platform, and their respective properties.
 * The caller is responsible for freeing *devices.
 *
 * @param platform_id The ID for the target platform.
 * @param devices The array of OpenCL Device properties.
 * @param num_devices The number of OpenCL devices found.
 *
 * @return CL_SUCCESS if and only if platforms and platform property reads are successful.
 */
cl_int OclFindDevices(const cl_platform_id platform_id, const OclDeviceProp **devices,
                      cl_uint *num_devices);

/**
 * @brief Converts cl_device_type to human-readable string,
 *
 * @param type An OpenCL device type.
 *
 * @return Constant human-readable string corresponding to OpenCL device type.
 */
const char *OclDeviceTypeString(cl_device_type type);

/**
 * @brief Frees OpenCL platfornm property struct.
 * Internally calls OclFreeDeviceProp.
 *
 * @param device An OpenCL platform property struct.
 *
 * @return CL_SUCCESS if and only if struct is successfully freed.
 */
cl_int OclFreePlatformProp(OclPlatformProp *platform);

/**
 * @brief Frees OpenCL device property struct,
 *
 * @param device An OpenCL device property struct.
 *
 * @return CL_SUCCESS if and only if struct is successfully freed.
 */
cl_int OclFreeDeviceProp(OclDeviceProp *device);
