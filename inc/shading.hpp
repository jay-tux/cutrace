//
// Created by jay on 11/20/23.
//

#ifndef CUTRACE_SHADING_HPP
#define CUTRACE_SHADING_HPP

#include "ray_cast.hpp"

/**
 * @brief Main namespace for GPU-related code.
 */
namespace cutrace::gpu {
template <typename S> requires (is_gpu_scene<S>)
__device__ vector phong(const S *scene, const ray *incoming, const vector *hit, size_t hit_id, const vector *normal, const uv *tex_coords, float ambient) {
  const auto &obj = scene->objects[hit_id];
  const auto &mat = scene->materials[get_mat_idx(obj)];
  vector diffuse{}, specular{};
  float reflect, translucent, phong_exp;
  get_phong_params(mat, normal, tex_coords, &diffuse, &specular, &reflect, &translucent, &phong_exp);

  vector final = diffuse * ambient;

  vector direction{}, unused{};
  float distance, shadow_dist_raw;
  size_t h_;
  uv tc{};

  for(const auto &light : scene->lights) {
    get_direction_to(light, hit, &direction, &distance);
    ray shadow { .start = *hit, .dir = direction };
    float light_dist = distance * direction.norm();
    vector color = get_color(light);
    vector nn = normal->normalized(), nd = direction.normalized();

    if(ray_cast(scene, shadow, 1e-3, &shadow_dist_raw, &h_, &unused, &unused, &tc, true)) {
      float shadow_dist = shadow_dist_raw * shadow.dir.norm();
      if(light_dist < shadow_dist) {
        float fd = max(0.0f, nn.dot(nd));
        vector ld = diffuse * color;

        vector h = ((-1.0f * incoming->dir.normalized()) + nd).normalized();
        float fs = pow(max(0.0f, nn.dot(h)), phong_exp);
        vector ls = specular * color;

        final += fd * ld + fs * ls;
      }
    }
  }

  return final;
}

template <typename S, size_t bounces> requires(is_gpu_scene<S>)
__device__ vector ray_color(const S *scene, const ray *incoming, float min_t, float ambient) {
  size_t id;
  vector normal{}, rgb{0.0f, 0.0f, 0.0f}, hit{};
  float distance;
  uv tc{};

  if(ray_cast(scene, incoming, min_t, &distance, &id, &hit, &normal, &tc, false)) {
    rgb = phong(scene, incoming, &hit, id, &normal, &tc, ambient);

    if constexpr(bounces != 0) {
      float reflective, translucent;
      auto mat = scene->materials[get_mat_idx(scene->objects[id])];
      get_bounce_params(mat, &normal, &tc, &reflective, &translucent);
      if(reflective >= 1e-6) {
        vector nd = incoming->dir.normalized(), nn = normal.normalized();
        ray reflection {
          .start = incoming->start + distance * incoming->dir,
          .dir = reflect(nd, nn)
        };

        auto r_rgb = ray_color<S, bounces - 1>(scene, &reflection, min_t, ambient);
        rgb += reflective * r_rgb;
      }

      if(translucent >= 1e-6) {
        ray pass{
          .start = incoming->start + distance * incoming->dir,
          .dir = incoming->dir
        };

        auto t_rgb = ray_color<S, bounces - 1>(scene, &pass, min_t, ambient);
        rgb = (1.0f - translucent) * rgb + translucent * t_rgb;
      }
    }
  }

  return rgb;
}
}

#endif //CUTRACE_SHADING_HPP
