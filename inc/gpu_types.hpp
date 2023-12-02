//
// Created by jay on 11/30/23.
//

#ifndef CUTRACE_GPU_TYPES_HPP
#define CUTRACE_GPU_TYPES_HPP

#include "gpu_variant.hpp"
#include "vector.hpp"
#include "gpu_array.hpp"

namespace cutrace::gpu {
/**
 * @brief Struct representing a ray.
 */
struct ray {
  vector start; //!< The starting point of the ray
  vector dir; //!< The direction of the ray
};

template <typename T>
concept is_gpu_cleanable = requires(T &t) {
  { t.gpu_clean() } -> std::same_as<void>;
};

template <typename ... Ts> requires(is_gpu_cleanable<Ts> && ...)
inline __host__ void gpu_clean(gpu_variant<Ts...> &v) {
  auto visitor = []<typename T>(T *obj) { obj->gpu_clean(); };
  visit(&visitor, &v);
}

/**
 * @brief Concept relating what it means to be a renderable object.
 * @tparam T The type to check
 *
 * For a type to construct objects that can be rendered, it needs to support the following (on a `const T &t`):
 *  - `t.intersect(const cutrace::gpu::ray *, float, cutrace::gpu::vector *, float *, cutrace::gpu::vector *) -> bool`, and
 *  - `t.mat_idx` (a public field of type `size_t`).
 */
template <typename T>
concept is_object = is_gpu_cleanable<T> && requires(const T &t, const ray *r, float min_t, vector *p, float *dist, vector *normal, uv *tex_coords) {
  { t.intersect(r, min_t, p, dist, normal, tex_coords) } -> std::same_as<bool>;
  { t.mat_idx } -> std::same_as<const size_t &>;
};

template <typename ... Ts> requires(is_object<Ts> && ...)
using gpu_object_set = gpu_variant<Ts...>;

template <typename ... Ts>
inline __device__ bool get_intersect(const gpu_object_set<Ts...> &o, const ray *r, float min_t, vector *p, float *dist, vector *normal, uv *tex_coords) {
  auto visitor = [r, min_t, p, dist, normal, tex_coords]<typename T>(const T &v) -> bool {
    return v.intersect(r, min_t, p, dist, normal, tex_coords);
  };
  return visit(&visitor, &o);
}

template <typename ... Ts>
inline __host__ __device__ size_t get_mat_idx(const gpu_object_set<Ts...> &o) {
  auto visitor = []<typename T>(const T &v) -> size_t { return v.mat_idx; };
  return visit(&visitor, &o);
}

/**
 * @brief Concept relating what it means to be a light to render with.
 * @tparam T The type to check
 *
 * For a type to construct objects that are lights, it needs to support the following (on a `const T &t`):
 *  - `t.direction_to(const cutrace::gpu::vector *, cutrace::gpu::vector *, float *) -> void`, and
 *  - `t.color` (a public field of type `cutrace::gpu::vector`).
 */
template <typename T>
concept is_light = is_gpu_cleanable<T> && requires(const T &t, const vector *point, vector *dir, float *dist) {
  { t.direction_to(point, dir, dist) } -> std::same_as<void>;
  { t.color } -> std::same_as<const vector &>;
};

template <typename ... Ts> requires(is_light<Ts> && ...)
using gpu_light_set = gpu_variant<Ts...>;

template <typename ... Ts>
inline __device__ void get_direction_to(const gpu_light_set<Ts...> &l, const vector *point, vector *direction, float *distance) {
  auto visitor = [point, direction,distance]<typename T>(const T &v) {
    v.direction_to(point, direction, distance);
  };
  visit(&visitor, &l);
}

template <typename ... Ts>
constexpr __host__ __device__ const vector &get_color(const gpu_light_set<Ts...> &l) {
  auto visitor = []<typename T>(const T &v) { return v.color; };
  return visit(&visitor, &l);
}

template <typename T>
concept is_material = is_gpu_cleanable<T> && requires(const T &t, const vector *normal, const uv *tc, vector *col, vector *spec, float *ref, float *tran, float *phong) {
  { t.get_phong_params(normal, tc, col, spec, ref, tran, phong) } -> std::same_as<void>;
  { t.is_transparent() } -> std::same_as<bool>;
  { t.get_bounce_params(normal, tc, ref, tran) } -> std::same_as<void>;
};

template <typename ... Ts>
using gpu_material_set = gpu_variant<Ts...>;

template <typename ... Ts>
constexpr __device__ void get_phong_params(const gpu_material_set<Ts...> &m, const vector *normal, const uv *tc, vector *col, vector *spec, float *ref, float *trans, float *phong) {
  auto visitor = [normal, tc, col, spec, ref, trans, phong]<typename T>(const T &v) {
    v.get_phong_params(normal, tc, col, spec, ref, trans, phong);
  };
  visit(&visitor, &m);
}

template <typename ... Ts>
constexpr __device__ bool is_transparent(const gpu_material_set<Ts...> &m) {
  auto visitor = []<typename T>(const T &v) { return v.is_transparent(); };
  return visit(&visitor, &m);
}

template <typename ... Ts>
constexpr __device__ bool is_reflecting(const gpu_material_set<Ts...> &m) {
  auto visitor = []<typename T>(const T &v) { return v.is_reflecting(); };
  return visit(&visitor, &m);
}

template <typename ... Ts>
constexpr __device__ void get_bounce_params(const gpu_material_set<Ts...> &m, const vector *normal, const uv *tc, float *ref, float *trans) {
  auto visitor = [normal, tc, ref, trans]<typename T>(const T &v) {
    v.get_bounce_params(normal, tc, ref, trans);
  };
  visit(&visitor, &m);
}

template <typename T>
concept is_camera = is_gpu_cleanable<T> && requires(const T &t, size_t x, size_t y, size_t *w, size_t *h) {
  { t.get_bounds(w, h) } -> std::same_as<void>;
  { t.get_ray(x, y) } -> std::same_as<ray>;
  { t.get_ambient() } -> std::same_as<float>;
};

template <typename O, typename L, typename M, typename C> struct gpu_scene_;

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

template <typename T> struct is_gpu_scene_t : std::bool_constant<false> {};
template <typename ... Os, typename ... Ls, typename ... Ms, typename C>
struct is_gpu_scene_t<gpu_scene_<gpu_object_set<Os...>, gpu_light_set<Ls...>, gpu_material_set<Ms...>, C>> :
        std::bool_constant<true> {};

template <typename T>
concept is_gpu_scene = is_gpu_scene_t<T>::value;
}

#endif //CUTRACE_GPU_TYPES_HPP
