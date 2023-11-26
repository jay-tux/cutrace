//
// Created by jay on 11/20/23.
//

#ifndef CUTRACE_SHADING_HPP
#define CUTRACE_SHADING_HPP

#include "gpu_types.hpp"

/**
 * @brief Main namespace for GPU-related code.
 */
namespace cutrace::gpu {
/**
 * @brief Performs Phong-shading for a single hit, without taking reflexivity (mirror) or transparency (translucency) into
 * account.
 * @param [in] scene The scene in which the ray was fired
 * @param [in] ray The ray that was fired
 * @param [in] hit The first point where the ray hit something
 * @param [in] hit_id The index of the object the ray hit first
 * @param [in] normal The normal of the object in the point where the ray hit it
 * @return The color of the ray, according to the Phong lighting model
 *
 * This function does consider all lights in the scene, as well as shadows. (Partially) transparent object do not cast
 * shadows.
 */
__device__ vector phong(const gpu_scene *scene, const ray *ray, const vector *hit,
                        size_t hit_id, const vector *normal);

/**
 * @brief Computes the color for a ray, for 10 bounces.
 * @param scene The scene in which to fire the ray
 * @param incoming The incoming ray from the camera to color
 * @param min_t The minimal parametric distance to consider a hit
 * @return The color for the ray
 *
 * This function depends heavily on \ref cutrace::gpu::cast_ray and \ref cutrace::gpu::phong. In contract to the Phong
 * function, it does take reflexivity (mirror) into account (by using recursion), as well as transparency
 * (translucency).
 */
__device__ vector ray_color(const gpu_scene *scene, const ray *incoming, float min_t);
}

#endif //CUTRACE_SHADING_HPP
