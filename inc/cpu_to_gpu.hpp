//
// Created by jay on 11/30/23.
//

#ifndef CUTRACE_CPU_TO_GPU_HPP
#define CUTRACE_CPU_TO_GPU_HPP

#include "cuda.hpp"
#include "cpu_types.hpp"
#include "gpu_types.hpp"
#include "gpu_array.hpp"

/**
 * @brief Namespace containing all code for converting scenes from CPU to GPU.
 */
namespace cutrace::cpu2gpu {
/**
 * @brief Concept relating what it means for `C` to be gpu-convertible to `G`.
 * @tparam C The CPU type
 * @tparam G The GPU type
 *
 * To satisfy this concept, `c.to_gpu()` must yield a `G` for any `const C &c`.
 */
template <typename C, typename G>
concept to_gpu_convertible = requires(const C &c) {
  { c.to_gpu() } -> std::same_as<G>;
};

/**
 * @brief Concept relating what it means to be a CPU-GPU object pair.
 * @tparam C The CPU object
 * @tparam G The GPU object
 */
template <typename C, typename G>
concept cpu_gpu_object_pair = gpu::is_object<G> && to_gpu_convertible<C, G>;

/**
 * @brief Concept relating what it means to be a CPU-GPU light pair.
 * @tparam C The CPU light
 * @tparam G The GPU light
 */
template <typename C, typename G>
concept cpu_gpu_light_pair = gpu::is_light<G> && to_gpu_convertible<C, G>;

/**
 * @brief Concept relating what it means to be a CPU-GPU material pair.
 * @tparam C The CPU material
 * @tparam G The GPU material
 */
template <typename C, typename G>
concept cpu_gpu_material_pair = gpu::is_material<G> && to_gpu_convertible<C, G>;

/**
 * @brief Concept relating what it means to be a CPU-GPU camera pair.
 * @tparam C The CPU camera
 * @tparam G The GPU camera
 */
template <typename C, typename G>
concept cpu_gpu_camera_pair = gpu::is_camera<G> && to_gpu_convertible<C, G>;

/**
 * @brief Helper function to convert a list of CPU values to GPU.
 * @tparam C The type of the CPU values
 * @tparam G The type of the GPU values
 * @param cpus The CPU values
 * @return An array containing all converted values in the same order
 * @see cutrace::cpu2gpu::vec_to_gpu_converting
 */
template <typename C, typename G> requires(to_gpu_convertible<C, G>)
gpu::gpu_array<G> vec_to_gpu(const std::vector<C> &cpus) {
  gpu::gpu_array<G> res { nullptr, 0 };

  cudaCheck(cudaMallocManaged(&res.buffer, cpus.size() * sizeof(G)))

  G *mapped = new G[cpus.size()];
  for(size_t i = 0; i < cpus.size(); i++) {
    mapped[i] = cpus[i].to_gpu();
  }

  cudaCheck(cudaMemcpy(res.buffer, mapped, cpus.size() * sizeof(G), cudaMemcpyHostToDevice))
  res.size = cpus.size();

  delete [] mapped;

  return res;
}

/**
 * @brief Helper function to convert a list of CPU values to GPU using a conversion function.
 * @tparam C The type of the CPU values
 * @tparam G The type of the GPU values
 * @tparam Fun The type of the conversion function
 * @param f The conversion function
 * @param cpus The CPU values
 * @return An array containing all converted values in the same order
 * @see cutrace::cpu2gpu::vec_to_gpu
 */
template <typename C, typename G, typename Fun> requires(std::invocable<Fun, const C &> && std::same_as<std::invoke_result_t<Fun, const C &>, G>)
gpu::gpu_array<G> vec_to_gpu_converting(Fun &&f, const std::vector<C> &cpus) {
  gpu::gpu_array<G> res { nullptr, 0 };

  cudaCheck(cudaMallocManaged(&res.buffer, cpus.size() * sizeof(G)))

  G *mapped = new G[cpus.size()];
  for(size_t i = 0; i < cpus.size(); i++) {
    mapped[i] = f(cpus[i]);
  }

  cudaCheck(cudaMemcpy(res.buffer, mapped, cpus.size() * sizeof(G), cudaMemcpyHostToDevice))
  res.size = cpus.size();

  delete [] mapped;

  return res;
}

/**
 * @brief Helper struct to convert a CPU scene to GPU.
 * @tparam CPUS The type of CPU scene
 * @tparam GPUS The type of GPU scene
 */
template <typename CPUS, typename GPUS>
struct cpu_to_gpu;

/**
 * @brief Helper struct to convert a CPU scene to GPU.
 * @tparam COs The type of CPU objects
 * @tparam GOs The type of GPU objects
 * @tparam CLs The type of CPU lights
 * @tparam GLs The type of GPU lights
 * @tparam CMs The type of CPU materials
 * @tparam GMs The type of GPU materials
 * @tparam CC The CPU type of the camera
 * @tparam GC The GPU type of the camera
 */
template <typename ... COs, typename ... GOs, typename ... CLs, typename ... GLs, typename ... CMs, typename ... GMs, typename CC, typename GC>
requires((cpu_gpu_object_pair<COs, GOs> && ...) && (cpu_gpu_light_pair<CLs, GLs> && ...) && (cpu_gpu_material_pair<CMs, GMs> && ...) && cpu_gpu_camera_pair<CC, GC>)
struct cpu_to_gpu<
        cpu::cpu_scene<cpu::cpu_object_set<COs...>, cpu::cpu_light_set<CLs...>, cpu::cpu_material_set<CMs...>, CC>,
        gpu::gpu_scene_<gpu::gpu_object_set<GOs...>, gpu::gpu_light_set<GLs...>, gpu::gpu_material_set<GMs...>, GC>
> {
  using cpu_object_t = cpu::cpu_object_set<COs...>; //!< The type of CPU objects
  using cpu_light_t = cpu::cpu_light_set<CLs...>; //!< The type of CPU lights
  using cpu_material_t = cpu::cpu_material_set<CMs...>; //!< The type of CPU materials
  using cpu_cam_t = CC; //!< The type of the CPU camera
  using gpu_object_t = gpu::gpu_object_set<GOs...>; //!< The type of GPU objects
  using gpu_light_t = gpu::gpu_light_set<GLs...>; //!< The type of GPU lights
  using gpu_material_t = gpu::gpu_material_set<GMs...>; //!< The type of GPU materials
  using gpu_cam_t = GC; //!< The type of the GPU camera
  using cpu_t = cpu::cpu_scene<cpu_object_t, cpu_light_t, cpu_material_t, cpu_cam_t>; //!< The type of the CPU scene
  using gpu_t = gpu::gpu_scene_<gpu_object_t, gpu_light_t, gpu_material_t, gpu_cam_t>; //!< The type of the GPU scene

  /**
   * @brief Converts a single, arbitrary CPU object to GPU
   * @param cpu The CPU object to convert
   * @return The converted GPU object
   */
  static gpu_object_t convert(const cpu_object_t &cpu) {
    const auto lambda = [](const auto &cpu_v) { return gpu_object_t { cpu_v.to_gpu() }; };
    return std::visit(lambda, cpu);
  }

  /**
   * @brief Converts a single, arbitrary CPU light to GPU
   * @param cpu The CPU light to convert
   * @return The converted GPU light
   */
  static gpu_light_t convert(const cpu_light_t &cpu) {
    const auto lambda = [](const auto &cpu_v) { return gpu_light_t { cpu_v.to_gpu() }; };
    return std::visit(lambda, cpu);
  }

  /**
   * @brief Converts a single, arbitrary CPU material to GPU
   * @param cpu The CPU material to convert
   * @return The converted GPU material
   */
  static gpu_material_t convert(const cpu_material_t &cpu) {
    const auto lambda = [](const auto &cpu_v) { return gpu_material_t { cpu_v.to_gpu() }; };
    return std::visit(lambda, cpu);
  }

  /**
   * @brief Converts a CPU scene to GPU.
   * @param cpu The CPU scene
   * @return The converted GPU scene
   */
  static gpu_t convert(const cpu_t &cpu) {
    auto lambda_object = [](const cpu_object_t &cpu) -> gpu_object_t { return convert(cpu); };
    auto lambda_light = [](const cpu_light_t &cpu) -> gpu_light_t { return convert(cpu); };
    auto lambda_material = [](const cpu_material_t &cpu) -> gpu_material_t { return convert(cpu); };
    return {
      .objects = vec_to_gpu_converting<cpu_object_t, gpu_object_t, decltype(lambda_object)>(std::move(lambda_object), cpu.objects),
      .lights = vec_to_gpu_converting<cpu_light_t, gpu_light_t, decltype(lambda_light)>(std::move(lambda_light), cpu.lights),
      .materials = vec_to_gpu_converting<cpu_material_t, gpu_material_t, decltype(lambda_material)>(std::move(lambda_material), cpu.materials),
      .cam = cpu.cam.to_gpu()
    };
  }
};
}

#endif //CUTRACE_CPU_TO_GPU_HPP
