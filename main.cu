#include <iostream>
#include "default_schema.hpp"
#include "loader.hpp"
#include "images.hpp"
#include "kernel.hpp"
#include "grid.hpp"

int main(int argc, const char **argv) {
  if(argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <scene file>\n";
    return -1;
  }

  auto scene = cutrace::cpu::schema::default_schema::load_file(argv[1]);

  if(!cutrace::cpu::schema::default_schema::last_was_success) {
    cutrace::cpu::schema::default_schema_viewer::dump_schema();
    return -2;
  }

  auto gpu_scene = cutrace::cpu::schema::default_to_gpu(scene);

  cutrace::gpu::dump_scene(gpu_scene);

  float max_d;
  cutrace::grid<float> depth_map;
  cutrace::grid<cutrace::vector> color_map;
  cutrace::grid<cutrace::vector> normal_map;
  size_t render, total;
  cutrace::gpu::render<decltype(gpu_scene), 5, 256>(gpu_scene, 1e-3, max_d, depth_map, color_map, normal_map, render, total);

  std::cout << "Render time was " << render << " ms; kernel time with setup/teardown was " << total << " ms.\n";

  cutrace::write_depth_map("./depth_map.jpg", depth_map, max_d);
  cutrace::write_normal_map("./normal_map.jpg", normal_map);
  cutrace::write_colorized("./frame.jpg", color_map);

//  for(const auto &row : depth_map) {
//    for(const auto &elem : row) {
//      std::cout << elem << "  ";
//    }
//    std::cout << "\n";
//  }

//  cutrace::gpu::cleanup(gpu_scene);
  return 0;
}
