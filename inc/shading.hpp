//
// Created by jay on 11/20/23.
//

#ifndef CUTRACE_SHADING_HPP
#define CUTRACE_SHADING_HPP

#include "gpu_types.hpp"

namespace cutrace::gpu {
__device__ vector phong(const gpu_scene *scene, const ray *ray, const vector *hit,
                        size_t hit_id, const vector *normal);

__device__ vector ray_color(const gpu_scene *scene, const ray *incoming, float min_t);
}

#endif //CUTRACE_SHADING_HPP
