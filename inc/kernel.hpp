//
// Created by jay on 12/2/23.
//

#ifndef CUTRACE_KERNEL_HPP
#define CUTRACE_KERNEL_HPP

#include <chrono>
#include "cuda.hpp"
#include "grid.hpp"
#include "shading.hpp"

namespace cutrace::gpu {
template <typename S, size_t bounces>
__global__ void render_kernel(const S scene, float fudge, float *depth, vector *color, vector *normals) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto cam = scene->cam;
  size_t w, h;
  cam.get_bounds(&w, &h);
  size_t max = w * h;
  if(tid >= max) return;

  size_t x_id = tid % w;
  size_t y_id = tid / w;

  float dist = INFINITY;
  ray r = cam.get_ray(x_id, y_id);
  size_t hit_id = scene.objects.size;
  vector hit_point{}, normal{};
  uv tc{};
  bool did_hit = ray_cast(&scene, &r, fudge, &dist, &hit_id, &hit_point, &normal, &tc, false);

  size_t px_id = y_id + w * x_id;
  depth[px_id] = dist;
  normals[px_id] = vector{0.5f, 0.5f, 0.5f} + 0.5f * normal;
  color[px_id] = did_hit ? ray_color<S, bounces>(&scene, &r, fudge, cam.get_ambient())
                         : cam.get_ambient() * vector{1.0f, 1.0f, 1.0f};
}

template <typename S, size_t tpb = 256> requires(is_gpu_scene<S>)
__host__ void render(const S &scene, float fudge, float &max, grid<float> &depth_map, grid<vector> &color_map, grid<vector> &normal_map, size_t &render_ms, size_t &total_ms) {
  auto start = std::chrono::high_resolution_clock::now();
  auto cam = scene.cam;
  depth_map.resize(cam.w, cam.h);
  color_map.resize(cam.w, cam.h);
  normal_map.resize(cam.w, cam.h);

  float *gpu_depth = nullptr, *gpu_col = nullptr, *gpu_normal = nullptr;
  cudaCheck(cudaMallocManaged(&gpu_depth, sizeof(float) * cam.w * cam.h))
  cudaCheck(cudaMallocManaged(&gpu_col, sizeof(vector) * cam.w * cam.h))
  cudaCheck(cudaMallocManaged(&gpu_normal, sizeof(vector) * cam.w * cam.h))

  size_t bpg = (cam.w * cam.h) / tpb + 1;

  auto start_render = std::chrono::high_resolution_clock::now();
  // TODO run kernel
  cudaCheck(cudaDeviceSynchronize())
  auto end_render = std::chrono::high_resolution_clock::now();

  for(size_t i = 0; i < cam.h; i++) {
    cudaCheck(cudaMemcpy(depth_map.data(i), &gpu_depth[i * cam.w], sizeof(float) * cam.w, cudaMemcpyDeviceToHost))
    cudaCheck(cudaMemcpy(color_map.data(i), &gpu_col[i * cam.w], sizeof(vector) * cam.w, cudaMemcpyDeviceToHost))
    cudaCheck(cudaMemcpy(normal_map.data(i), &gpu_normal[i * cam.w], sizeof(vector) * cam.w, cudaMemcpyDeviceToHost))
  }

  cudaCheck(cudaFree(gpu_depth))
  cudaCheck(cudaFree(gpu_col))
  cudaCheck(cudaFree(gpu_normal))

  max = 0.0f;
  for(const auto &row: depth_map) {
    for(const auto &f: row) {
      if(std::isfinite(f) && f > max) max = f;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  render_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_render - start_render).count();
  total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}
}

#endif //CUTRACE_KERNEL_HPP

