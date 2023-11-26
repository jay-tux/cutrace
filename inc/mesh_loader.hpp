//
// Created by jay on 11/24/23.
//

#ifndef CUTRACE_MESH_LOADER_HPP
#define CUTRACE_MESH_LOADER_HPP

#include <string>
#include <vector>
#include "cpu_types.hpp"

/**
 * @brief Main namespace for CPU-related code.
 */
namespace cutrace::cpu {
/**
 * @brief Uses ASSIMP to load all meshes from a file.
 * @param [in] file The model file
 * @param [in] mat_idx The material index, should be specified separately
 * @return A list of all meshes found in the file
 *
 * This function will ignore all materials and textures, as well as lights found in the file, as those should be
 * specified separately in the scene's JSON file.
 *
 * @see cutrace::loader::load
 */
std::vector<triangle_set> load_mesh(const std::string &file, size_t mat_idx);
}

#endif //CUTRACE_MESH_LOADER_HPP
