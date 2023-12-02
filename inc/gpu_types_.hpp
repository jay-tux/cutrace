//
// Created by jay on 11/30/23.
//

#ifndef CUTRACE_GPU_TYPES__HPP
#define CUTRACE_GPU_TYPES__HPP

#include "gpu_variant.hpp"
#include "vector.hpp"

namespace cutrace::gpu {
/**
 * @brief Struct representing a ray.
 */
struct ray {
  vector start; //!< The starting point of the ray
  vector dir; //!< The direction of the ray
};

/**
 * @brief Concept relating what it means to be a renderable object.
 * @tparam T The type to check
 *
 * For a type to construct objects that can be rendered, it needs to support the following (on a `const T &t`):
 *  - `t.intersect(const cutrace::gpu::ray *, float, cutrace::gpu::vector *, float *, cutrace::gpu::vector *) -> bool`, and
 *  - `t.mat_idx` (a public field of type `size_t`).
 */
template <typename T>
concept is_object = requires(const T &t, const ray *r, float min_t, vector *p, float *dist, vector *normal) {
  { t.intersect(r, min_t, p, dist, normal) } -> std::same_as<bool>;
  { t.mat_idx } -> std::same_as<const size_t &>;
};

template <typename ... Ts> requires(is_object<Ts> && ...)
using gpu_object_set = gpu_variant<Ts...>;

/**
 * @brief Concept relating what it means to be a light to render with.
 * @tparam T The type to check
 *
 * For a type to construct objects that are lights, it needs to support the following (on a `const T &t`):
 *  - `t.direction_to(const cutrace::gpu::vector *, cutrace::gpu::vector *, float *) -> void`, and
 *  - `t.color` (a public field of type `cutrace::gpu::vector`).
 */
template <typename T>
concept is_light = requires(const T &t, const vector *point, vector *dir, float *dist) {
  { t.direction_to(point, dir, dist) } -> std::same_as<void>;
  { t.color } -> std::same_as<const vector &>;
};

template <typename ... Ts> requires(is_light<Ts> && ...)
using gpu_light_set = gpu_variant<Ts...>;

// TODO: concept is_material

template <typename ... Ts>
using gpu_material_set = gpu_variant<Ts...>;

template <typename O, typename L, typename M, typename C> struct gpu_scene_;

// TODO: concept is_camera

template <typename ... Os, typename ... Ls, typename ... Ms, typename C>
struct gpu_scene_<gpu_object_set<Os...>, gpu_light_set<Ls...>, gpu_material_set<Ms...>, C> {
  using object = gpu_object_set<Os...>;
  using light = gpu_light_set<Ls...>;
  using material = gpu_material_set<Ms...>;
  using camera = C;

  gpu_array<object> objects;
  gpu_array<light> lights;
  gpu_array<material> materials;
  camera cam;
};
}

#endif //CUTRACE_GPU_TYPES__HPP
