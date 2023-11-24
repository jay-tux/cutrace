//
// Created by jay on 11/24/23.
//

#ifndef CUTRACE_MESH_LOADER_HPP
#define CUTRACE_MESH_LOADER_HPP

#include <string>
#include <vector>
#include "cpu_types.hpp"

namespace cutrace::cpu {
std::vector<triangle_set> load_mesh(const std::string &file, size_t mat_idx);
}

#endif //CUTRACE_MESH_LOADER_HPP
