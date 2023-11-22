//
// Created by jay on 11/18/23.
//

#include "cuda.hpp"
#include "cpu_types.hpp"

using namespace cutrace;
using namespace cutrace::cpu;

template <typename T, typename GT, typename Fun>
concept t_to_gpu_fun = requires(Fun &&f, const T &t) {
  { f(t) } -> std::same_as<GT>;
};

template <typename T, typename GT, typename Fun> requires(t_to_gpu_fun<T, GT, Fun>)
gpu::gpu_array<GT> cpu_to_gpu(const std::vector<T> &cpus, Fun &&f) {
  gpu::gpu_array<GT> res { nullptr, 0 };

  cudaCheck(cudaMallocManaged(&res.buffer, cpus.size() * sizeof(GT)))

  GT mapped[cpus.size()];
  for(size_t i = 0; i < cpus.size(); i++) {
    mapped[i] = f(cpus[i]);
  }

  cudaCheck(cudaMemcpy(res.buffer, mapped, cpus.size() * sizeof(GT), cudaMemcpyHostToDevice))
  res.size = cpus.size();

  return res;
}

__host__ gpu::triangle triangle::to_gpu() const {
  return { p1.to_gpu(), p2.to_gpu(), p3.to_gpu(), mat_idx };
}

__host__ gpu::triangle_set triangle_set::to_gpu() const {
  gpu::triangle_set set{
    .triangles = { nullptr, 0 },
    .mat_idx = mat_idx
  };

  cudaCheck(cudaMallocManaged(&set.triangles.buffer, tris.size() * sizeof(gpu::triangle)))

  gpu::triangle mapped[tris.size()];
  for(size_t i = 0; i < tris.size(); i++) {
    mapped[i] = tris[i].to_gpu();
  }

  cudaCheck(cudaMemcpy(set.triangles.buffer, mapped, tris.size() * sizeof(gpu::triangle), cudaMemcpyHostToDevice));
  set.triangles.size = tris.size();
  return set;
}

__host__ gpu::plane plane::to_gpu() const {
  return {
    .point = point.to_gpu(),
    .normal = normal.to_gpu(),
    .mat_idx = mat_idx
  };
}

__host__ gpu::sphere sphere::to_gpu() const {
  return {
    .center = center.to_gpu(),
    .radius = radius,
    .mat_idx = mat_idx
  };
}

__host__ gpu::gpu_object cpu::to_gpu(const cpu_object &cpu) {
  gpu::gpu_object res;
  std::visit([&res](const auto &it) {
    res.set(it.to_gpu());
  }, cpu);
  return res;
}

__host__ gpu::gpu_array<gpu::gpu_object> cpu::to_gpu(const std::vector<cpu_object> &cpus) {
  return cpu_to_gpu<cpu_object, gpu::gpu_object>(cpus, [](const cpu_object &cpu) {
    return to_gpu(cpu);
  });
}

gpu::sun sun::to_gpu() const {
  return { .direction = direction.to_gpu(), .color = color.to_gpu() };
}

gpu::point_light point_light::to_gpu() const {
  return { .point = point.to_gpu(), .color = color.to_gpu() };
}

gpu::gpu_light cpu::to_gpu(const cpu_light &cpu) {
  gpu::gpu_light res;
  std::visit([&res](const auto &it) {
    res.set(it.to_gpu());
  }, cpu);
  return res;
}

gpu::gpu_array<gpu::gpu_light> cpu::to_gpu(const std::vector<cpu_light> &cpus) {
  return cpu_to_gpu<cpu_light, gpu::gpu_light>(cpus, [](const cpu_light &cpu) {
    return to_gpu(cpu);
  });
}

gpu::gpu_mat cpu_mat::to_gpu() const {
  return {
    .color = color.to_gpu(),
    .specular = specular,
    .reflexivity = reflexivity,
    .phong_exp = phong_exp,
    .transparency = transparency
  };
}

gpu::gpu_array<gpu::gpu_mat> cpu::to_gpu(const std::vector<cpu_mat> &cpus) {
  return cpu_to_gpu<cpu_mat, gpu::gpu_mat>(cpus, [](const cpu_mat &cpu) {
    return cpu.to_gpu();
  });
}

gpu::gpu_scene cpu_scene::to_gpu() const {
  return {
          .objects = ::to_gpu(objects),
          .lights = ::to_gpu(lights),
          .materials = ::to_gpu(materials)
  };
}