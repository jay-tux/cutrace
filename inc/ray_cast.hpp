//
// Created by jay on 12/2/23.
//

#ifndef CUTRACE_RAY_CAST_HPP
#define CUTRACE_RAY_CAST_HPP

#include "vector.hpp"
#include "gpu_types.hpp"

/**
 * @brief Main namespace for GPU-related code.
 */
namespace cutrace::gpu {
/**
 * Performs a single ray-cast operation.
 * @tparam S The type of the GPU scene
 * @param scene The GPU scene
 * @param finder The ray to use for intersections
 * @param min_dist The minimal distance for an intersection to be valid
 * @param[out] distance The distance of the hit, or infinity if no hit was found
 * @param[out] hit_id The object index of the hit, undefined if no hit was found
 * @param[out] hit_point The 3D-coordinate of the hit, undefined if no hit was found
 * @param[out] normal The normal at the point of the hit, undefined if no hit was found
 * @param[out] tex_coords The texture coordinates at the point of the hit, undefined if no hit was found
 * @param ignore_transparent Whether or not to ignore transparent objects
 * @return True if a hit was found, false otherwise
 */
template <typename S> requires(is_gpu_scene<S>)
__device__ bool ray_cast(const S *scene, const ray *finder, float min_dist, float *distance, size_t *hit_id, vector *hit_point, vector *normal, uv *tex_coords, bool ignore_transparent) {
  *distance = INFINITY;
  vector hit{}, nrm{};
  uv tc{};
  float dist;
  bool was_hit = false;

  for(size_t i = 0; i < scene->objects.size; i++) {
    const auto &obj = scene->objects[i];
    const auto &mat = scene->materials[get_mat_idx(obj)];
    if(ignore_transparent && is_transparent(mat)) continue; // ignore

    if(get_intersect(obj, finder, min_dist, &hit, &dist, &nrm, &tc)) {
      if (dist > min_dist && dist < *distance) {
        *distance = dist;
        *hit_id = i;
        *hit_point = hit;
        *normal = nrm;
        *tex_coords = tc;
        was_hit = true;
      }
    }
  }

  return was_hit;
}
}

#endif //CUTRACE_RAY_CAST_HPP
