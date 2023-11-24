#include <iostream>
#include <iomanip>
#include "loader.hpp"
#include "kernel_depth.hpp"
#include "images.hpp"

int main(int argc, const char **argv) {
  if(argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <scene file>\n";
    return -1;
  }

  auto scene = cutrace::loader::load(argv[1]);

//  "eye": [1,0,2],
//  "up": [0,1,0],
//  "look": [-0.92388,0,-0.38268],

  cutrace::gpu::cam cam {
    .pos = { 1, 0, 2 }, // { 0, 3, 1 },
    .up = { 0, 1, 0 }, // { 0, 0, -1 },
    .forward = { 0, 0, -1 }, // { 0, -1, 0 }
  };
  cam.look_at({ -0.92388,0,-0.38268 });

  std::cout << "Camera: \n"
               "-> pos = v3(" << cam.pos.x << ", " << cam.pos.y << ", " << cam.pos.z << ")\n"
               "-> up = v3(" << cam.up.x << ", " << cam.up.y << ", " << cam.up.z << ")\n"
               "-> right = v3(" << cam.right.x << ", " << cam.right.y << ", " << cam.up.z << ")\n"
               "-> forward = v3(" << cam.forward.x << ", " << cam.forward.y << ", " << cam.forward.z << ")\n";

  auto gpu_scene = scene.to_gpu();
  float max_d;
  cutrace::gpu::grid<float> depth_map;
  cutrace::gpu::grid<cutrace::gpu::vector> color_map;
  cutrace::gpu::grid<cutrace::gpu::vector> normal_map;
  cutrace::gpu::render(cam, gpu_scene, 640, 640, max_d, depth_map, color_map, normal_map);
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
