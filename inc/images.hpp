//
// Created by jay on 11/19/23.
//

#ifndef CUTRACE_IMAGES_HPP
#define CUTRACE_IMAGES_HPP

#include <string>
#include <vector>
#include "kernel_depth.hpp"

namespace cutrace {
void write_depth_map(const std::string &file, const gpu::grid<float> &depth_map, float max_d);
void write_normal_map(const std::string &file, const gpu::grid<gpu::vector> &normal_map);
void write_colorized(const std::string &file, const gpu::grid<gpu::vector> &colorized);
}

#endif //CUTRACE_IMAGES_HPP
