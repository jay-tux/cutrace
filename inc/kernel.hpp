//
// Created by jay on 12/2/23.
//

#ifndef CUTRACE_KERNEL_HPP
#define CUTRACE_KERNEL_HPP

#include <chrono>
#include "cuda.hpp"
#include "grid.hpp"
#include "shading.hpp"
#include "gpu_types.hpp"

/**
 * @brief Main namespace for GPU-related code.
 */
namespace cutrace::gpu {
/**
 * @brief Runs the render kernel.
 * @tparam S The GPU scene type
 * @tparam bounces The maximal amount of bounces
 * @param scene The GPU scene
 * @param fudge The minimal distance to the camera for a hit to be considered valid
 * @param[out] depth A pointer to the depth map buffer
 * @param[out] color A pointer to the color map buffer
 * @param[out] normals A pointer to the normal map buffer
 *
 * All three buffers should be at least of size `w*h`, where `w` and `h` are the width and height of the camera,
 * respectively.
 *
 * It is recommended to use `cutrace::render` to render a scene, as it sets up all required buffers correctly.
 *
 * @see cutrace::render
 */
template <typename S, size_t bounces>
__global__ void render_kernel(const S scene, float fudge, float *depth, vector *color, vector *normals) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto cam = scene.cam;
  size_t w, h;
  cam.get_bounds(&w, &h);
  size_t max = w * h;
  if(tid >= max) { return; }

  size_t x_id = tid % w;
  size_t y_id = tid / w;

  float dist = INFINITY;
  ray r = cam.get_ray(x_id, y_id);
  size_t hit_id = scene.objects.size;
  vector hit_point{}, normal{0,0,0};
  uv tc{};
  bool did_hit = ray_cast(&scene, &r, fudge, &dist, &hit_id, &hit_point, &normal, &tc, false);

  size_t px_id = y_id * w + x_id;
  depth[px_id] = dist;
  normals[px_id] = normal;
//  color[px_id] = did_hit ? ray_color<S, bounces>(&scene, &r, fudge, cam.get_ambient())
//                         : cam.get_ambient() * vector{1.0f, 1.0f, 1.0f};
  color[px_id] = ray_color<S, bounces>(&scene, &r, fudge, cam.get_ambient());
}

/**
 * @brief Runs the render kernel with all setup and teardown.
 * @tparam S The type of the GPU scene
 * @tparam bounces The maximal amount of bounces
 * @tparam tpb The amount of threads per block
 * @param scene The GPU scene
 * @param fudge The minimal distance to consider a hit valid
 * @param max The maximal non-infinite depth among all rays
 * @param[out] depth_map A grid for the depth map
 * @param[out] color_map A grid for the color map
 * @param[out] normal_map A grid for the normal map
 * @param[out] render_ms The amount of time spent in the render kernel
 * @param[out] total_ms The amount of time spent in total
 *
 * This function does, in order:
 *  1. Sets up the CPU-side output buffers (`depth_map`, `color_map`, `normal_map`) and clears them
 *  2. Sets up the GPU-side output buffers
 *  3. Runs the render kernel
 *  4. Copies the GPU-side output to CPU
 *  5. Clear up the GPU-side output buffers
 *  6. Find the maximal value in the depth buffer
 *
 * @see cutrace::render_kernel
 */
template <typename S, size_t bounces = 10, size_t tpb = 256> requires(impl::is_gpu_scene<S>)
__host__ void render(const S &scene, float fudge, float &max, grid<float> &depth_map, grid<vector> &color_map, grid<vector> &normal_map, size_t &render_ms, size_t &total_ms) {
  auto start = std::chrono::high_resolution_clock::now();

  size_t w, h;
  scene.cam.get_bounds(&w, &h);

  depth_map.resize(w, h);
  color_map.resize(w, h);
  normal_map.resize(w, h);

  float *gpu_depth = nullptr;
  vector *gpu_col = nullptr, *gpu_normal = nullptr;
  cudaCheck(cudaMallocManaged(&gpu_depth, sizeof(float) * w * h))
  cudaCheck(cudaMallocManaged(&gpu_col, sizeof(vector) * w * h))
  cudaCheck(cudaMallocManaged(&gpu_normal, sizeof(vector) * w * h))

  size_t bpg = (w * h) / tpb + 1;

  auto start_render = std::chrono::high_resolution_clock::now();
  render_kernel<S, bounces><<<bpg, tpb>>>(scene, fudge, gpu_depth, gpu_col, gpu_normal);
  cudaCheck(cudaDeviceSynchronize())
  auto end_render = std::chrono::high_resolution_clock::now();

  for(size_t i = 0; i < h; i++) {
    cudaCheck(cudaMemcpy(depth_map.data(i), &gpu_depth[i * w], sizeof(float) * w, cudaMemcpyDeviceToHost))
    cudaCheck(cudaMemcpy(color_map.data(i), &gpu_col[i * w], sizeof(vector) * w, cudaMemcpyDeviceToHost))
    cudaCheck(cudaMemcpy(normal_map.data(i), &gpu_normal[i * w], sizeof(vector) * w, cudaMemcpyDeviceToHost))
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

//template <typename S> requires(is_gpu_scene<S>)
//__host__ void cleanup(S &scene) {
//  for(auto &object: scene.objects) gpu_clean(object);
//  for(auto &light: scene.lights) gpu_clean(light);
//  for(auto &mat: scene.materials) gpu_clean(mat);
//  scene.cam.gpu_clean();
//
//  cudaCheck(cudaFree(scene.objects.buffer))
//  cudaCheck(cudaFree(scene.lights.buffer))
//  cudaCheck(cudaFree(scene.materials.buffer))
//}

/**
 * @brief Prints some basic scene details
 * @tparam S The type of the GPU scene
 * @param scene The GPU scene
 * @see cutrace::dump_scene
 */
template <typename S> requires(impl::is_gpu_scene<S>)
__global__ void dump_scene_kernel(S scene) {
  printf(" -> Have %-4llu objects:\n", scene.objects.size);
  for(size_t i = 0; i < scene.objects.size; i++) {
    printf("  -> Object   #%-4llu has type #%-2llu\n", i, scene.objects[i].get_idx());
  }

  printf(" -> Have %-4llu lights:\n", scene.lights.size);
  for(size_t i = 0; i < scene.lights.size; i++) {
    printf("  -> Light    #%-4llu has type #%-2llu\n", i, scene.lights[i].get_idx());
  }

  printf(" -> Have %-4llu materials:\n", scene.materials.size);
  for(size_t i = 0; i < scene.materials.size; i++) {
    printf("  -> Material #%-4llu has type #%-2llu\n", i, scene.materials[i].get_idx());
  }
}

/**
 * @brief Prints some basic scene details using the kernel from CPU-side
 * @tparam S The type of the GPU scene
 * @param scene The GPU scene
 * @see cutrace::dump_scene_kernel
 */
template <typename S> requires(impl::is_gpu_scene<S>)
__host__ void dump_scene(const S &scene) {
  dump_scene_kernel<<<1, 1>>>(scene);
  cudaCheck(cudaDeviceSynchronize())
}
}

#endif //CUTRACE_KERNEL_HPP

