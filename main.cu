#include <iostream>
#include <iomanip>
#include "loader.hpp"
#include "kernel_depth.hpp"
#include "images.hpp"
#include "kernel_prepare.hpp"

int main(int argc, const char **argv) {
  if(argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <scene file>\n";
    return -1;
  }

  auto scene = cutrace::loader::load(argv[1]);
  const auto &cam = scene.camera;

  std::cout << "Camera: \n"
               "-> pos = v3(" << cam.pos.x << ", " << cam.pos.y << ", " << cam.pos.z << ")\n"
               "-> up = v3(" << cam.up.x << ", " << cam.up.y << ", " << cam.up.z << ")\n"
               "-> right = v3(" << cam.right.x << ", " << cam.right.y << ", " << cam.up.z << ")\n"
               "-> forward = v3(" << cam.forward.x << ", " << cam.forward.y << ", " << cam.forward.z << ")\n";

  auto gpu_scene = scene.to_gpu();
  cutrace::gpu::prepare_scene(gpu_scene, scene);

  float max_d;
  cutrace::gpu::grid<float> depth_map;
  cutrace::gpu::grid<cutrace::gpu::vector> color_map;
  cutrace::gpu::grid<cutrace::gpu::vector> normal_map;
  cutrace::gpu::render(scene.camera, gpu_scene, max_d, depth_map, color_map, normal_map);
  cutrace::write_depth_map("./depth_map.jpg", depth_map, max_d);
  cutrace::write_normal_map("./normal_map.jpg", normal_map);
  cutrace::write_colorized("./frame.jpg", color_map);

  cutrace::gpu::cleanup(gpu_scene);

//  for(const auto &row : depth_map) {
//    for(const auto &elem : row) {
//      std::cout << std::setw(7) << std::setprecision(3) << elem << "  ";
//    }
//    std::cout << "\n";
//  }

  return 0;
}
