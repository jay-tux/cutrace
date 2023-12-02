//
// Created by jay on 11/19/23.
//

#ifndef CUTRACE_IMAGES_HPP
#define CUTRACE_IMAGES_HPP

#include <string>
#include <vector>
#include "grid.hpp"
#include "vector.hpp"

/**
 * @brief Namespace containing all of cutrace’s code.
 */
namespace cutrace {
/**
 * @brief Writes a depth map to a JPG image (single-channel).
 * @param file The name of the file to write to
 * @param depth_map The depth map data
 * @param max_d The maximal value in the depth map
 */
void write_depth_map(const std::string &file, const grid<float> &depth_map, float max_d);
/**
 * @brief Writes a normal map to a JPG image (3 colors).
 * @param file The name of the file to write to
 * @param normal_map The normal map data
 */
void write_normal_map(const std::string &file, const grid<vector> &normal_map);
/**
 * @brief Writes a frame to a JPG image (3 colors).
 * @param file The name of the file to write to
 * @param colorized The pixel colors
 */
void write_colorized(const std::string &file, const grid<vector> &colorized);
}

#endif //CUTRACE_IMAGES_HPP
