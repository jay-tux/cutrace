//
// Created by jay on 11/25/23.
//

#ifndef CUTRACE_KERNEL_PREPARE_HPP
#define CUTRACE_KERNEL_PREPARE_HPP

#include "gpu_types.hpp"
#include "cpu_types.hpp"

/**
 * @brief Main namespace for GPU-related code.
 */
namespace cutrace::gpu {
/**
 * @brief Prepares the GPU scene from the CPU scene, by precomputing the mesh bounding boxes.
 * @param [in, out] scene The GPU scene to update
 * @param [in] cpu The CPU scene to gather mesh indexes from
 *
 * This is a wrapper function around \ref cutrace::gpu::calc_model_bounding_boxes.
 */
__host__ void prepare_scene(gpu_scene scene, const cpu::cpu_scene &cpu);

/**
 * @brief Kernel function to calculate the bounding boxes for a set of models.
 * @param [in, out] scene The GPU scene to update
 * @param [in] models Which objects (by index) are meshes
 *
 * This function launches a single thread per model to compute the bounding box for. Specifying the index of a non-model
 * will result in undefined behavior.
 */
__global__ void calc_model_bounding_boxes(gpu_scene scene, gpu_array<size_t> models);
}

#endif //CUTRACE_KERNEL_PREPARE_HPP
