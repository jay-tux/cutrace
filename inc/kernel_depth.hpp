//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_KERNEL_DEPTH_HPP
#define CUTRACE_KERNEL_DEPTH_HPP

#include "gpu_array.hpp"
#include "gpu_types.hpp"
#include <vector>

namespace cutrace::gpu {
struct cam {
  vector pos;
  vector up;
  vector forward;
  vector right;
  float near, far;
  size_t w, h;

  __host__ void look_at(const vector &v);
};

__device__ bool cast_ray(
        const gpu_scene *scene, const ray *finder, float min_dist, float *distance,
        size_t *hit_id, vector *hit_point, vector *normal, bool ignore_transparent
);

template <typename T>
using grid = std::vector<std::vector<T>>;

__host__ void
render(cam cam, gpu_scene scene, float &max, grid<float> &depth, grid<vector> &color,
       grid<vector> &normals
);

__host__ void cleanup(gpu_scene scene);
}

#endif //CUTRACE_KERNEL_DEPTH_HPP
