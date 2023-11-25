//
// Created by jay on 11/25/23.
//

#ifndef CUTRACE_KERNEL_PREPARE_HPP
#define CUTRACE_KERNEL_PREPARE_HPP

#include "gpu_types.hpp"
#include "cpu_types.hpp"

namespace cutrace::gpu {
__host__ void prepare_scene(gpu_scene scene, const cpu::cpu_scene &cpu);

__global__ void calc_model_bounding_boxes(gpu_scene scene, gpu_array<size_t> models);
}

#endif //CUTRACE_KERNEL_PREPARE_HPP
