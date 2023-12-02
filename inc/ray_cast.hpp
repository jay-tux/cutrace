//
// Created by jay on 12/2/23.
//

#ifndef CUTRACE_RAY_CAST_HPP
#define CUTRACE_RAY_CAST_HPP

#include "vector.hpp"
#include "gpu_types.hpp"

namespace cutrace::gpu {
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

    if(get_intersect(obj, finder, min_dist, &hit, &dist, &nrm, &tc)
        && dist > min_dist && dist < *distance) {
      *distance = dist;
      *hit_id = i;
      *hit_point = hit;
      *normal = nrm;
      *tex_coords = tc;
      was_hit = true;
    }
  }

  return was_hit;
}
}

#endif //CUTRACE_RAY_CAST_HPP
