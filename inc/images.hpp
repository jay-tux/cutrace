//
// Created by jay on 11/19/23.
//

#ifndef CUTRACE_IMAGES_HPP
#define CUTRACE_IMAGES_HPP

#include <string>
#include <vector>
#include "grid.hpp"
#include "vector.hpp"
#include "stb_image_write.h"

/**
 * @brief Namespace containing all of cutrace’s code.
 */
namespace cutrace {
using byte = unsigned char;

/**
 * @brief Writes a depth map to a JPG image (single-channel).
 * @param file The name of the file to write to
 * @param depth_map The depth map data
 * @param max_d The maximal value in the depth map
 */
inline void write_depth_map(const std::string &file, const grid<float> &depth_map, float max_d) {
  static const auto intensity = [](float v, float m){
    return std::isfinite(v) ? (byte)(255 * (m - v) / m) : 0;
  };

  byte *data = new byte[3 * depth_map.elems()];
  for(size_t px = 0; px < depth_map.elems(); px++) {
    byte v = intensity(depth_map.raw(px), max_d);
    data[3 * px + 0] = v;
    data[3 * px + 1] = v;
    data[3 * px + 2] = v;
  }

  stbi_write_jpg(file.c_str(), (int)depth_map.cols(), (int)depth_map.rows(), 3, data, 90);
  delete [] data;
}
/**
 * @brief Writes a normal map to a JPG image (3 colors).
 * @param file The name of the file to write to
 * @param normal_map The normal map data
 */
inline void write_normal_map(const std::string &file, const grid<vector> &normal_map) {
  static const auto clamp = [](vector v){
    if(v.norm() <= 1e-6) {
      return std::tuple<byte, byte, byte>{0,0,0};
    }
    auto n = vector{0.5f, 0.5f, 0.5f} + 0.5f * v.normalized();
    return std::tuple<byte, byte, byte>{ (byte)(255 * n.x), (byte)(255 * n.y), (byte)(255 * n.z) };
  };

  byte *data = new byte[3 * normal_map.elems()];
  for(size_t px = 0; px < normal_map.elems(); px++) {
    auto [r, g, b] = clamp(normal_map.raw(px));
    data[3 * px + 0] = r;
    data[3 * px + 1] = g;
    data[3 * px + 2] = b;
  }

  stbi_write_jpg(file.c_str(), (int)normal_map.cols(), (int)normal_map.rows(), 3, data, 90);
  delete [] data;
}
/**
 * @brief Writes a frame to a JPG image (3 colors).
 * @param file The name of the file to write to
 * @param colorized The pixel colors
 */
inline void write_colorized(const std::string &file, const grid<vector> &colorized) {
  static const auto clamp = [](vector v){
    vector n = { std::min(1.0f, std::max(0.0f, v.x)), std::min(1.0f, std::max(0.0f, v.y)), std::min(1.0f, std::max(0.0f, v.z)) };
    return std::tuple<byte, byte, byte>{ (byte)(255 * n.x), (byte)(255 * n.y), (byte)(255 * n.z) };
  };

  byte *data = new byte[3 * colorized.elems()];
  for(size_t px = 0; px < colorized.elems(); px++) {
    auto [r, g, b] = clamp(colorized.raw(px));
    data[3 * px + 0] = r;
    data[3 * px + 1] = g;
    data[3 * px + 2] = b;
  }

  stbi_write_jpg(file.c_str(), (int)colorized.cols(), (int)colorized.rows(), 3, data, 90);
  delete [] data;
}
}

#endif //CUTRACE_IMAGES_HPP
