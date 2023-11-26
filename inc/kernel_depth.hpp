//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_KERNEL_DEPTH_HPP
#define CUTRACE_KERNEL_DEPTH_HPP

#include "gpu_array.hpp"
#include "gpu_types.hpp"
#include <vector>

/**
 * @brief Main namespace for GPU-related code
 */
namespace cutrace::gpu {
/**
 * @brief Performs a single ray-cast operation.
 * @param [in] scene The scene to cast a ray through
 * @param [in] finder The ray to cast through the scene
 * @param [in] min_dist The minimal parametric distance along the ray to consider an intersection
 * @param [out] distance The distance to the first hit, if any
 * @param [out] hit_id The index of the object hit first, if any
 * @param [out] hit_point The point in space where the first object was hit, if any
 * @param [out] normal The normal in at the hit point, if any
 * @param [in] ignore_transparent Whether or not to ignore objects with transparency (for shadow rays)
 * @return Whether or not an object was hit
 */
__device__ bool cast_ray(
        const gpu_scene *scene, const ray *finder, float min_dist, float *distance,
        size_t *hit_id, vector *hit_point, vector *normal, bool ignore_transparent
);

/**
 * @brief Type alias for a grid of an arbitrary type (an `std::vector` of `std::vector`).
 */
template <typename T>
using grid = std::vector<std::vector<T>>;

/**
 * @brief Renders a single frame, with depth and normal maps
 * @param [in] cam The camera settings for the frame
 * @param [in] scene The scene to render
 * @param [out] max The maximum, non-infinite depth among all the traced rays
 * @param [out] depth The output depth map, will be reassigned, row-major
 * @param [out] color The output colored frame, will be reassigned, row-major
 * @param [out] normals The output normal map for the frame, will be reassigned, row-major
 *
 * This function will reallocate the given grids to make sure that they are rectangular, with `cam.h` rows of `cam.w`
 * values. After that, it allocates memory on the GPU for the output, and prints out the scene details in short, after
 * which it calls the (internal) rendering kernel and copies all output from GPU to CPU. Finally it frees the GPU memory
 * and computes the maximal depth.
 *
 * Per pixel, a single ray is fired on a separate thread.
 */
__host__ void
render(cam cam, gpu_scene scene, float &max, grid<float> &depth, grid<vector> &color,
       grid<vector> &normals
);

/**
 * @brief Cleans up all GPU memory used by the scene.
 * @param [in,out] scene The scene to clear
 */
__host__ void cleanup(gpu_scene scene);
}

#endif //CUTRACE_KERNEL_DEPTH_HPP
