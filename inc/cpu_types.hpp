//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_CPU_TYPES_HPP
#define CUTRACE_CPU_TYPES_HPP

#include <vector>
#include <variant>
#include "gpu_types.hpp"
#include "kernel_depth.hpp"

namespace cutrace::cpu {
struct triangle {
  vector p1, p2, p3;
  size_t mat_idx;

  __host__ gpu::triangle to_gpu() const;
};

struct triangle_set {
  std::vector<triangle> tris;
  size_t mat_idx;

  __host__ gpu::triangle_set to_gpu() const;
};

struct plane {
  vector point;
  vector normal;
  size_t mat_idx;

  __host__ gpu::plane to_gpu() const;
};

struct sphere {
  vector center;
  float radius;
  size_t mat_idx;

  __host__ gpu::sphere to_gpu() const;
};

using cpu_object = std::variant<triangle, triangle_set, plane, sphere>;

__host__ gpu::gpu_object to_gpu(const cpu_object &cpu);
__host__ gpu::gpu_array<gpu::gpu_object> to_gpu(const std::vector<cpu_object> &cpus);

struct sun {
  vector direction;
  vector color;

  __host__ gpu::sun to_gpu() const;
};

struct point_light {
  vector point;
  vector color;

  __host__ gpu::point_light to_gpu() const;
};

using cpu_light = std::variant<sun, point_light>;
__host__ gpu::gpu_light to_gpu(const cpu_light &cpu);
__host__ gpu::gpu_array<gpu::gpu_light> to_gpu(const std::vector<cpu_light>& cpus);

struct cpu_mat {
  vector color;
  float specular, reflexivity, phong_exp, transparency;

  [[nodiscard]] __host__ gpu::gpu_mat to_gpu() const;
};

__host__ gpu::gpu_array<gpu::gpu_mat> to_gpu(const std::vector<cpu_mat> &cpus);

struct cpu_scene {
  gpu::cam camera;
  std::vector<cpu_object> objects;
  std::vector<cpu_light> lights;
  std::vector<cpu_mat> materials;

  [[nodiscard]] __host__ gpu::gpu_scene to_gpu() const;
};
}

#endif //CUTRACE_CPU_TYPES_HPP
