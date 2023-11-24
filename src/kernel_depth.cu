//
// Created by jay on 11/18/23.
//

#include <chrono>
#include "cuda.hpp"
#include "kernel_depth.hpp"
#include "shading.hpp"

using namespace cutrace;
using namespace cutrace::gpu;

__host__ void cam::look_at(const vector &v) {
  forward = (v - pos).normalized();
  right = forward.cross(up).normalized(); // perpendicular to plane formed by forward & up
  up = right.cross(forward).normalized(); // make them all perpendicular to each other
}

__device__ vector project_to_y0(ray r) {
  float fac = -r.start.y / r.dir.y;
  return r.start + r.dir * fac;
}

__device__ bool gpu::cast_ray(const gpu_scene *scene, const ray *finder, float min_dist, float *distance, size_t *hit_id, vector *hit_point, vector *normal, bool ignore_transparent) {
  *distance = INFINITY;
  vector hit{};
  float dist;
  vector nrm{};
  bool was_hit = false;
  for(size_t i = 0; i < scene->objects.size; i++) {
    const auto &obj = scene->objects[i];
    auto lambda = [](const auto *obj) { return obj->mat_idx; };
    auto mat = scene->materials[visit(&lambda, &scene->objects[i])];
    if(ignore_transparent && mat.transparency > 0.0f) continue;

    if(intersects(finder, &obj, 1e-6, &hit, &dist, &nrm)) {
      if(dist > min_dist && dist < *distance) {
        *distance = dist;
        *hit_id = i;
        *hit_point = hit;
        *normal = nrm;
        was_hit = true;
      }
    }
  }
  return was_hit;
}

__global__ void kernel(cam cam, const gpu_scene scene, float *depth, vector *color, vector *normals, size_t w, size_t h) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t max = w * h;
  if(tid >= max) return;

  size_t x_id = tid % w;
  size_t y_id = tid / w;

  float dist = INFINITY;

  ray r{
    .start = cam.pos,
    .dir = (cam.forward + ((float)x_id / (float)w - 0.5f) * cam.right + (0.5f - (float)y_id / (float)h) * cam.up).normalized()
  };

  size_t hit_id = scene.objects.size;
  vector hit_point{};
  vector normal{};
  cast_ray(&scene, &r, 0.1f, &dist, &hit_id, &hit_point, &normal, false);

  depth[y_id * w + x_id] = dist;
  normals[y_id * w + x_id] = normal;
  color[y_id * w + x_id] = ray_color(&scene, &r, 0.1f);
}

struct printer {
  __device__ inline void operator()(const triangle *t) {
    printf("-> triangle{ .p1 = v3(%04f, %04f, %04f), .p2 = v3(%04f, %04f, %04f), .p3 = v3(%04f, %04f, %04f), .mat = %d }\n",
           t->p1.x, t->p1.y, t->p1.z, t->p2.x, t->p2.y, t->p2.z, t->p3.x, t->p3.y, t->p3.z, (int)t->mat_idx);
  }

  __device__ inline void operator()(const triangle_set *s) {
    printf(" -> triangle_set{ .tris = [%d], .mat = %d }\n",
           (int)s->triangles.size, (int)s->mat_idx);
  }

  __device__ inline void operator()(const sphere *s) {
    printf(" -> sphere{ .center = v3(%04f, %04f, %04f), .radius = %04f, .mat = %d }\n",
           s->center.x, s->center.y, s->center.z, s->radius, (int)s->mat_idx);
  }

  __device__ inline void operator()(const plane *p) {
    printf(" -> plane{ .point = v3(%04f, %04f, %04f), .normal = v3(%04f, %04f, %04f), .mat = %d }\n",
           p->point.x, p->point.y, p->point.z, p->normal.x, p->normal.y, p->normal.z, (int)p->mat_idx);
  }

  __device__ inline void operator()(const sun *s) {
    printf(" -> sun{ .direction = v3(%04f, %04f, %04f), .color = v3(%04f, %04f, %04f) }\n",
           s->direction.x, s->direction.y, s->direction.z, s->color.x, s->color.y, s->color.z);
  }

  __device__ inline void operator()(const point_light *p) {
    printf(" -> point_light{ .point = v3(%04f, %04f, %04f), .color = v3(%04f, %04f, %04f) }\n",
           p->point.x, p->point.y, p->point.z, p->color.x, p->color.y, p->color.z);
  }

  __device__ inline void operator()(const gpu_mat &m) {
    printf(" -> material{ .color = v3(%04f, %04f, %04f), .specular = %04f, .reflect = %04f, .phong_exp = %.04f, .transparency = %04f }\n",
           m.color.x, m.color.y, m.color.z, m.specular, m.reflexivity, m.phong_exp, m.reflexivity);
  }
};

__global__ void scene_dump(gpu_scene scene) {
  printer p{};
  for(const auto &obj: scene.objects) {
    visit(&p, &obj);
  }
  for(const auto &light: scene.lights) {
    visit(&p, &light);
  }
  for(const auto &m : scene.materials) {
    p(m);
  }
}

__host__ void
gpu::render(cam cam, gpu_scene scene, size_t w, size_t h, float &max, grid<float> &depth_map, grid<vector> &colors, grid<vector> &normals) {
  depth_map.resize(h, {});
  for(auto &row: depth_map) {
    row.resize(w, 0.0f);
  }

  colors.resize(h, {});
  for(auto &row: colors) {
    row.resize(w, {});
  }

  normals.resize(h, {});
  for(auto &row: normals) {
    row.resize(w, {});
  }

  float *gpu_dm = nullptr;
  cudaCheck(cudaMallocManaged(&gpu_dm, sizeof(float) * w * h))

  vector *gpu_col = nullptr;
  cudaCheck(cudaMallocManaged(&gpu_col, sizeof(vector) * w * h))

  vector *gpu_nrm = nullptr;
  cudaCheck(cudaMallocManaged(&gpu_nrm, sizeof(vector) * w * h))

  scene_dump<<<1, 1>>>(scene);
  cudaCheck(cudaDeviceSynchronize())

  auto time = std::chrono::high_resolution_clock::now();
  int tpb = 256;
  size_t bpg = (w * h) / tpb + 1;
  kernel<<<bpg, tpb>>>(cam, scene, gpu_dm, gpu_col, gpu_nrm, w, h);
  cudaChecked(nullptr)

  cudaCheck(cudaDeviceSynchronize())
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Rendering a single frame took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - time).count() << "ms.\n";

  for(size_t i = 0; i < depth_map.size(); i++) {
    cudaCheck(cudaMemcpy(depth_map[i].data(), &gpu_dm[i * w], sizeof(float) * w, cudaMemcpyDeviceToHost))
    cudaCheck(cudaMemcpy(colors[i].data(), &gpu_col[i * w], sizeof(vector) * w, cudaMemcpyDeviceToHost))
    cudaCheck(cudaMemcpy(normals[i].data(), &gpu_nrm[i * w], sizeof(vector) * w, cudaMemcpyDeviceToHost))
  }

  cudaCheck(cudaFree(gpu_dm))
  cudaCheck(cudaFree(gpu_col))
  cudaCheck(cudaFree(gpu_nrm))

  max = 0.0f;
  for(const auto &row: depth_map) {
    for(const auto &f: row) {
      if(std::isfinite(f) && f > max) max = f;
    }
  }
}

__host__ void gpu::cleanup(gpu_scene scene) {
  for(auto &e: scene.objects) {
    if(e.holds<triangle_set>()) cudaCheck(cudaFree(e.get<triangle_set>()->triangles.buffer))
  }
  cudaCheck(cudaFree(scene.objects.buffer))
  cudaCheck(cudaFree(scene.materials.buffer))
  cudaCheck(cudaFree(scene.lights.buffer))
}
