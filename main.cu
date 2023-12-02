#include <iostream>
#include <iomanip>
#include "loader.hpp"
#include "images.hpp"
#include "default_schema.hpp"
#include "kernel.hpp"
#include "grid.hpp"

int main(int argc, const char **argv) {
  if(argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <scene file>\n";
    return -1;
  }

  auto scene = cutrace::cpu::schema::default_schema::load_file(argv[1]);
  auto gpu_scene = cutrace::cpu::schema::default_to_gpu(scene);

  float max_d;
  cutrace::grid<float> depth_map;
  cutrace::grid<cutrace::vector> color_map;
  cutrace::grid<cutrace::vector> normal_map;
//  cutrace::gpu::render(scene.camera, gpu_scene, max_d, depth_map, color_map, normal_map);
  cutrace::write_depth_map("./depth_map.jpg", depth_map, max_d);
  cutrace::write_normal_map("./normal_map.jpg", normal_map);
  cutrace::write_colorized("./frame.jpg", color_map);

//  cutrace::gpu::cleanup(gpu_scene);
  return 0;
}
