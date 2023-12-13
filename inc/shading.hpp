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
template <typename S> requires(impl::is_gpu_scene<S>)
__device__ float shadow_intensity(const S *scene, const ray *shadow_ray, float max_dist) {
  float intensity = 0.0f;
  float last_hit = 0.0f;
  ray check{ .start = shadow_ray->start, .dir = shadow_ray->dir };
  // ray_cast(scene, &shadow, 1e-3, &shadow_dist_raw, &h_, &unused, &unused, &tc, true);
  float dist, trans, ref;
  size_t h;
  vector hit{}, normal{};
  uv uvs{};

  while(ray_cast(scene, &check, last_hit + 1e-3, &dist, &h, &hit, &normal, &uvs, false) && dist < max_dist) {
    const auto &mat = scene->materials[get_mat_idx(scene->objects[h])];
    get_bounce_params(mat, &normal, &uvs, &ref, &trans);
    intensity += (1.0f - trans);

    if(intensity >= 1.0f) {
      return 1.0f; // overload
    }

    last_hit = dist;
  }

  return intensity;
}

/**
 * @brief Performs Phong-shading.
 * @tparam S The type of the GPU scene
 * @param scene The GPU scene
 * @param incoming The incoming ray
 * @param hit The point of the hit
 * @param hit_id The index of the object that was hit
 * @param normal The normal at the point of the hit
 * @param tex_coords The texture coordinates at the point of the hit
 * @param ambient The ambient lighting factor
 * @return The color of the light according to the material settings and the Phong lighting model
 *
 * This function takes into account all lights, and whether or not there are shadows cast on this point.
 * However, it does not apply reflection or transparency.
 *
 * @see cutrace::gpu::ray_color
 */
template <typename S> requires (impl::is_gpu_scene<S>)
__device__ vector phong(const S *scene, const ray *incoming, const vector *hit, size_t hit_id, const vector *normal, const uv *tex_coords, float ambient) {
  const auto obj = &scene->objects[hit_id];
  size_t mat_i = get_mat_idx(*obj);
  const auto &mat = scene->materials[mat_i];
  vector diffuse{}, specular{};
  float reflect, translucent, phong_exp;
  get_phong_params(mat, normal, tex_coords, &diffuse, &specular, &reflect, &translucent, &phong_exp);

  vector final = diffuse * ambient;

  vector direction{};
  float distance = INFINITY;
  
  for(const auto &light : scene->lights) {
    get_direction_to(light, hit, &direction, &distance);
    ray shadow{.start = *hit, .dir = direction.normalized()};
    float light_dist = distance * direction.norm();
    vector color = get_color(light);
    vector nn = normal->normalized(), nd = direction.normalized();

    /*bool did_hit = ray_cast(scene, &shadow, 1e-3, &shadow_dist_raw, &h_, &unused, &unused, &tc, true);
    float shadow_dist = shadow_dist_raw * shadow.dir.norm();
    if (!(did_hit && shadow_dist < light_dist))*/
    float shadow_fac = shadow_intensity(scene, &shadow, light_dist);
    if(shadow_fac < 1.0f){
      float fd = max(0.0f, nn.dot(nd));
      vector ld = diffuse * color;

      vector h = ((-1.0f * incoming->dir.normalized()) + nd).normalized();
      float fs = pow(max(0.0f, nn.dot(h)), phong_exp);
      vector ls = specular * color;

      final += (1 - shadow_fac) * (fd * ld + fs * ls);
    }
  }

  return final;
}

/**
 * @brief Gets the color of a ray.
 * @tparam S The type of the GPU scene
 * @tparam bounces The maximal amount of bounces
 * @param scene The GPU scene
 * @param incoming The incoming ray
 * @param min_t The minimal distance for a hit to be valid
 * @param ambient The ambient lighting factor
 * @return The color that the given ray should have
 *
 * This function relies on the `cutrace::gpu::phong` function to perform basic shading, then uses recursion to apply
 * reflections and shading.
 *
 * @see cutrace::gpu::phong
 */
template <typename S, size_t bounces> requires(impl::is_gpu_scene<S>)
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
