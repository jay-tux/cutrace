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

/**
 * @brief Concept relating what it means to be cleanable.
 * @tparam T The type to check
 *
 * A cleanable type supports the `t.gpu_clean()` method.
 */
template <typename T>
concept is_gpu_cleanable = requires(T &t) {
  { t.gpu_clean() } -> std::same_as<void>;
};

/**
 * @brief Cleans up an arbitrary cleanable GPU variant.
 * @tparam Ts The types of the variant
 */
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
 *  - `t.intersect(const cutrace::gpu::ray *, float, cutrace::vector *, float *, cutrace::vector *) -> bool`, and
 *  - `t.mat_idx` (a public field of type `size_t`).
 */
template <typename T>
concept is_object = is_gpu_cleanable<T> && requires(const T &t, const ray *r, float min_t, vector *p, float *dist, vector *normal, uv *tex_coords) {
  { t.intersect(r, min_t, p, dist, normal, tex_coords) } -> std::same_as<bool>;
  { t.mat_idx } -> std::same_as<const size_t &>;
};

/**
 * @brief Type alias for a variant of GPU objects.
 * @tparam Ts The GPU object types
 */
template <typename ... Ts> requires(is_object<Ts> && ...)
using gpu_object_set = gpu_variant<Ts...>;

/**
 * @brief Performs intersection detection on an arbitrary GPU object.
 * @tparam Ts The types of possible objects
 * @param o The object variant
 * @param r The incoming ray
 * @param min_t The minimal parametric distance for an intersection to be valid
 * @param[out] p The hit point, undefined if no hit took place
 * @param[out] dist The parametric distance to the hit point, undefined if no hit took place
 * @param[out] normal The normal vector in the hit point, undefined if no hit took place
 * @param[out] tex_coords The texture coordinates at the hit point, undefined if no hit took place
 * @return True if a hit took place, false otherwise
 */
template <typename ... Ts>
inline __device__ bool get_intersect(const gpu_object_set<Ts...> &o, const ray *r, float min_t, vector *p, float *dist, vector *normal, uv *tex_coords) {
  auto visitor = [r, min_t, p, dist, normal, tex_coords]<typename T>(const T *v) -> bool {
    return v->intersect(r, min_t, p, dist, normal, tex_coords);
  };
  return visit(&visitor, &o);
}
/**
 * @brief Gets the material index of an arbitrary GPU object.
 * @tparam Ts The types of possible objects
 * @param o The object variant
 * @return The object's material index
 */
template <typename ... Ts>
inline __host__ __device__ size_t get_mat_idx(const gpu_object_set<Ts...> &o) {
  auto visitor = []<typename T>(const T *v) -> size_t { return v->mat_idx; };
  return visit(&visitor, &o);
}

/**
 * @brief Concept relating what it means to be a light to render with.
 * @tparam T The type to check
 *
 * For a type to construct objects that are lights, it needs to support the following (on a `const T &t`):
 *  - `t.direction_to(const cutrace::vector *, cutrace::vector *, float *) -> void`, and
 *  - `t.color` (a public field of type `cutrace::vector`).
 */
template <typename T>
concept is_light = is_gpu_cleanable<T> && requires(const T &t, const vector *point, vector *dir, float *dist) {
  { t.direction_to(point, dir, dist) } -> std::same_as<void>;
  { t.color } -> std::same_as<const vector &>;
};

/**
 * @brief Type alias for a variant of GPU lights.
 * @tparam Ts The GPU light types
 */
template <typename ... Ts> requires(is_light<Ts> && ...)
using gpu_light_set = gpu_variant<Ts...>;

/**
 * @brief Gets the direction and distance from a point to an arbitrary light.
 * @tparam Ts The possible light types
 * @param l The light variant
 * @param point The point of intersection
 * @param[out] direction The direction from the point to the light
 * @param[out] distance The parametric distance from the point to the light
 * 
 * The distance is parametric relative to the direction vector.
 */
template <typename ... Ts>
inline __device__ void get_direction_to(const gpu_light_set<Ts...> &l, const vector *point, vector *direction, float *distance) {
  auto visitor = [point, direction,distance]<typename T>(const T *v) {
    v->direction_to(point, direction, distance);
  };
  visit(&visitor, &l);
}

/**
 * @brief Gets the color of an arbitrary light.
 * @tparam Ts The types of possible lights
 * @param l The light variant
 * @return The light color
 */
template <typename ... Ts>
constexpr __host__ __device__ vector get_color(const gpu_light_set<Ts...> &l) {
  auto visitor = []<typename T>(const T *v) { return v->color; };
  return visit(&visitor, &l);
}

/**
 * @brief Concept relating what it means to be a material.
 * @tparam T The type to check
 * 
 * To be a material, a type must support (for `const T &t`):
 *  - `t.get_phong_params(const cutrace::vector *, const cutrace::uv *, cutrace::vector *, cutrace::vector *, float *, float *) -> void`
 *  - `t.is_transparent() -> bool`
 *  - `t.is_reflecting() -> bool`
 *  - `t.get_bounce_params(const cutrace::vector *, const cutrace::uv *, float *, float *) -> void`
 */
template <typename T>
concept is_material = is_gpu_cleanable<T> && requires(const T &t, const vector *normal, const uv *tc, vector *col, vector *spec, float *ref, float *tran, float *phong) {
  { t.get_phong_params(normal, tc, col, spec, ref, tran, phong) } -> std::same_as<void>;
  { t.is_transparent() } -> std::same_as<bool>;
  { t.is_reflecting() } -> std::same_as<bool>;
  { t.get_bounce_params(normal, tc, ref, tran) } -> std::same_as<void>;
};

/**
 * @brief Type alias for a variant of GPU materials.
 * @tparam Ts The GPU material types
 */
template <typename ... Ts>
using gpu_material_set = gpu_variant<Ts...>;

/**
 * @brief Gets the Phong parameters of an arbitrary material
 * @tparam Ts The possible material types
 * @param m The material variant
 * @param normal The normal at the point of intersection
 * @param tc The texture coordinates at the point of intersection
 * @param[out] col The diffuse color of the material at the point of intersection
 * @param[out] spec The specular color of the material at the point of intersection
 * @param[out] ref The reflexivity factor of the material at the point of intersection
 * @param[out] trans The translucency factor of the material at the point of intersection
 * @param[out] phong The Phong-exponent of the material at the point of intersection
 */
template <typename ... Ts>
constexpr __device__ void get_phong_params(const gpu_material_set<Ts...> &m, const vector *normal, const uv *tc, vector *col, vector *spec, float *ref, float *trans, float *phong) {
  auto visitor = [normal, tc, col, spec, ref, trans, phong]<typename T>(const T *v) {
    v->get_phong_params(normal, tc, col, spec, ref, trans, phong);
  };
  visit(&visitor, &m);
}

/**
 * @brief Detects whether an arbitrary material is transparent.
 * @tparam Ts The possible material types
 * @param m The material variant
 * @return True if the material is transparent, false otherwise
 */
template <typename ... Ts>
constexpr __device__ bool is_transparent(const gpu_material_set<Ts...> &m) {
  auto visitor = []<typename T>(const T *v) { return v->is_transparent(); };
  return visit(&visitor, &m);
}

/**
 * @brief Detects whether an arbitrary material has any mirror-like reflections.
 * @tparam Ts The possible material types
 * @param m The material variant
 * @return True if the material has mirror-like reflections, false otherwise
 */
template <typename ... Ts>
constexpr __device__ bool is_reflecting(const gpu_material_set<Ts...> &m) {
  auto visitor = []<typename T>(const T *v) { return v->is_reflecting(); };
  return visit(&visitor, &m);
}

/**
 * @brief Gets the bounce parameters (reflexivity, translucency) of an arbitrary material.
 * @tparam Ts The possible material types
 * @param m The material variant
 * @param normal The normal of the material at the point of intersection
 * @param tc The texture coordinates of the material at the point of intersection
 * @param[out] ref The reflexivity factor of the material at the point of intersection
 * @param[out] trans The translucency factor of the material at the point of intersection
 */
template <typename ... Ts>
constexpr __device__ void get_bounce_params(const gpu_material_set<Ts...> &m, const vector *normal, const uv *tc, float *ref, float *trans) {
  auto visitor = [normal, tc, ref, trans]<typename T>(const T *v) {
    v->get_bounce_params(normal, tc, ref, trans);
  };
  visit(&visitor, &m);
}

/**
 * @brief Concept relating what it means to be a camera.
 * @tparam T The type to check
 *
 * To be a camera, a type needs to support (for `const T &t`):
 *  - `t.get_bounds(size_t *, size_t *) -> void`
 *  - `t.get_ray(size_t, size_t) -> cutrace::ray`
 *  - `t.get_ambient() -> float`
 */
template <typename T>
concept is_camera = is_gpu_cleanable<T> && requires(const T &t, size_t x, size_t y, size_t *w, size_t *h) {
  { t.get_bounds(w, h) } -> std::same_as<void>;
  { t.get_ray(x, y) } -> std::same_as<ray>;
  { t.get_ambient() } -> std::same_as<float>;
};

/**
 * @brief Structure representing a scene on GPU.
 * @tparam O The object variant type
 * @tparam L The light variant type
 * @tparam M The material variant type
 * @tparam C The camera type
 */
template <typename O, typename L, typename M, typename C> struct gpu_scene_;

/**
 * @brief Structure representing a scene on GPU.
 * @tparam Os The object types
 * @tparam Ls The light types
 * @tparam Ms The material types
 * @tparam C The camera type
 */
template <typename ... Os, typename ... Ls, typename ... Ms, typename C>
struct gpu_scene_<gpu_object_set<Os...>, gpu_light_set<Ls...>, gpu_material_set<Ms...>, C> {
  using object = gpu_object_set<Os...>; //!< Type alias for the object variant type
  using light = gpu_light_set<Ls...>; //!< Type alias for the light variant type
  using material = gpu_material_set<Ms...>; //!< Type alias for the material variant type
  using camera = C; //!< Type alias for the camera type

  gpu_array<object> objects; //!< All objects in the scene
  gpu_array<light> lights; //!< All lights in the scene
  gpu_array<material> materials; //!< All materials in the scene
  camera cam; //!< The camera to render the scene with
};

namespace impl {
template<typename T>
struct is_gpu_scene_t : std::bool_constant<false> {
};
template<typename ... Os, typename ... Ls, typename ... Ms, typename C>
struct is_gpu_scene_t<gpu_scene_<gpu_object_set<Os...>, gpu_light_set<Ls...>, gpu_material_set<Ms...>, C>> :
        std::bool_constant<true> {
};

template<typename T>
concept is_gpu_scene = is_gpu_scene_t<T>::value;
}
}

#endif //CUTRACE_GPU_TYPES_HPP
