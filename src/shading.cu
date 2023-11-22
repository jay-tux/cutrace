//
// Created by jay on 11/20/23.
//

#include "shading.hpp"
#include "kernel_depth.hpp"

using namespace cutrace;
using namespace cutrace::gpu;

#define I_A 0.1
#define FUDGE 1e-3
#define BOUNCES 10

struct visitor {
  __device__ inline size_t operator()(const triangle *t) { return t->mat_idx; }
  __device__ inline size_t operator()(const triangle_set *t) { return t->mat_idx; }
  __device__ inline size_t operator()(const plane *t) { return t->mat_idx; }
  __device__ inline size_t operator()(const sphere *t) { return t->mat_idx; }
};

struct light_visit {
  __device__ inline vector operator()(const sun *s) { return s->color; }
  __device__ inline vector operator()(const point_light *p) { return p->color; }
};

__device__ vector gpu::phong(const gpu_scene *scene, const ray *incoming, const vector *hit,
                  size_t hit_id, const vector *normal) {
  const auto *obj = &scene->objects[hit_id];
  visitor fun;
  const auto &mat = scene->materials[visit(&fun, obj)];

  vector final_color = mat.color * I_A;

  vector direction{};
  float distance;
  float shadow_dist;
  size_t h_;
  vector unused{};

  for(const auto &l: scene->lights) {
    direction_to(hit, &l, &direction, &distance);
    ray shadow {
      .start = *hit,
      .dir = direction
    };

    light_visit color_visitor{};
    auto color = visit(&color_visitor, &l);
    auto nn = normal->normalized();
    auto nd = direction.normalized();
    cast_ray(scene, &shadow, FUDGE, &shadow_dist, &h_, &unused, &unused, true);
    if(distance < shadow_dist) {
      auto fd = max(0.0f, nn.dot(nd));
      auto ld = mat.color * color;

      auto h = ((-1 * incoming->dir.normalized()) + nd).normalized();
      auto fs = pow(max(0.0f, nn.dot(h)), mat.phong_exp); // / (rd * rd);
      auto ls = mat.specular * mat.color * color;

      final_color += fd * ld + fs * ls;
    }
  }

  return final_color;
}

template <size_t bounces>
__device__ inline vector ray_color(const gpu_scene *scene, const ray *incoming, float min_t) {
  size_t id;
  vector normal{};
  vector rgb{0.0f, 0.0f, 0.0f};
  float distance;
  vector hit{};

  if(cast_ray(scene, incoming, FUDGE, &distance, &id, &hit, &normal, false)) {
    rgb = phong(scene, incoming, &hit, id, &normal);

    if constexpr(bounces != 0) {
      auto nd = incoming->dir.normalized();
      auto nn = normal.normalized();
      ray reflection {
        .start = incoming->start + distance * incoming->dir,
        .dir = reflect(nd, nn)
      };

      auto lambda = [](const auto *obj) { return obj->mat_idx; };
      auto mat = scene->materials[visit(&lambda, &scene->objects[id])];
      float factor = mat.reflexivity;
      auto rgb2 = ray_color<bounces - 1>(scene, &reflection, FUDGE);
      rgb += factor * rgb2;

      ray passthrough {
        .start = incoming->start + distance * incoming->dir,
        .dir = incoming->dir
      };
      auto rgb3 = ray_color<bounces - 1>(scene, &passthrough, FUDGE);
      rgb = (1.0f - mat.transparency) * rgb + mat.transparency * rgb3;
    }
  }

  return rgb;
}

__device__ vector gpu::ray_color(const gpu_scene *scene, const ray *incoming, float min_t) {
  return ::ray_color<BOUNCES>(scene, incoming, min_t);
}