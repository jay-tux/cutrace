//
// Created by jay on 11/25/23.
//

#include "cuda.hpp"
#include "kernel_prepare.hpp"

using namespace cutrace;
using namespace cutrace::gpu;

__host__ void gpu::prepare_scene(gpu_scene scene, const cpu::cpu_scene &cpu) {
  auto models = cpu::find_model_indexes(cpu);

  int tpb = 256;
  size_t bpg = models.size / tpb + 1;
  calc_model_bounding_boxes<<<bpg, tpb>>>(scene, models);
  cudaCheck(cudaDeviceSynchronize())

  cudaCheck(cudaFree(models.buffer))
}

__device__ bound box_for(const triangle *t) {
#define METRIC(s, fun) fun(t->p1.s, fun(t->p2.s, t->p3.s))
  return {
          .min = {
                  METRIC(x, fminf), METRIC(y, fminf), METRIC(z, fminf),
          },
          .max = {
                  METRIC(x, fmaxf), METRIC(y, fmaxf), METRIC(z, fmaxf),
          }
  };
#undef METRIC
}

__global__ void gpu::calc_model_bounding_boxes(gpu_scene scene, gpu_array<size_t> models) {
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if(idx >= models.size) return;

  bound box{
          .min = { INFINITY, INFINITY, INFINITY },
          .max = { -INFINITY, -INFINITY, -INFINITY }
  };
  auto *model = scene.objects[models.buffer[idx]].get<triangle_set>();
  for(const auto &tri : model->triangles) {
    box.merge(box_for(&tri));
  }

  model->bounding_box = box;
}