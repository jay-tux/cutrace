//
// Created by jay on 11/19/23.
//

#include <tuple>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "images.hpp"

using namespace cutrace;
using namespace cutrace::gpu;

using byte = unsigned char;

byte intensity(float v, float max_v) {
  if(std::isfinite(v)) {
    return (byte)(255 * (max_v - v) / max_v);
  }
  return 0;
}

std::tuple<byte, byte, byte> clamped(const vector &color) {
  constexpr auto range_clamp = [](float f) { return std::min(1.0f, std::max(0.0f, f)); };
  constexpr auto to_byte = [](float f) { return (byte)(f * 255); };

  return {
    to_byte(range_clamp(color.x)),
    to_byte(range_clamp(color.y)),
    to_byte(range_clamp(color.z))
  };
}

std::tuple<byte, byte, byte> normal_clamped(const vector &normal) {
  auto n = normal.normalized();
  constexpr auto to_byte = [](float f) { return (byte)(f * 255); };
  return {
    to_byte(n.x), to_byte(n.y), to_byte(n.z)
  };
}

void cutrace::write_depth_map(const std::string &file, const gpu::grid<float> &depth_map, float max_d) {
  byte *data = new byte[3 * depth_map.size() * depth_map[0].size()];
  for(size_t row = 0; row < depth_map.size(); row++) {
    for(size_t col = 0; col < depth_map[0].size(); col++) {
      byte v = intensity(depth_map[row][col], max_d);
      data[3 * (row * depth_map[0].size() + col) + 0] = v;
      data[3 * (row * depth_map[0].size() + col) + 1] = v;
      data[3 * (row * depth_map[0].size() + col) + 2] = v;
    }
  }

  stbi_write_jpg(file.c_str(), (int)depth_map[0].size(), (int)depth_map.size(), 3, data, 90);
  delete [] data;
}

void cutrace::write_colorized(const std::string &file, const gpu::grid<gpu::vector> &color_map) {
  byte *data = new byte[3 * color_map.size() * color_map[0].size()];
  for(size_t row = 0; row < color_map.size(); row++) {
    for(size_t col = 0; col < color_map[0].size(); col++) {
      auto [r,g,b] = clamped(color_map[row][col]);
      data[3 * (row * color_map[0].size() + col) + 0] = r;
      data[3 * (row * color_map[0].size() + col) + 1] = g;
      data[3 * (row * color_map[0].size() + col) + 2] = b;
    }
  }

  stbi_write_jpg(file.c_str(), (int)color_map[0].size(), (int)color_map.size(), 3, data, 90);
  delete [] data;
}

void cutrace::write_normal_map(const std::string &file, const gpu::grid<gpu::vector> &normal_map) {
  byte *data = new byte[3 * normal_map.size() * normal_map[0].size()];
  for(size_t row = 0; row < normal_map.size(); row++) {
    for(size_t col = 0; col < normal_map[0].size(); col++) {
      auto [r,g,b] = normal_clamped(normal_map[row][col]);
      data[3 * (row * normal_map[0].size() + col) + 0] = r;
      data[3 * (row * normal_map[0].size() + col) + 1] = g;
      data[3 * (row * normal_map[0].size() + col) + 2] = b;
    }
  }

  stbi_write_jpg(file.c_str(), (int)normal_map[0].size(), (int)normal_map.size(), 3, data, 90);
  delete [] data;
}