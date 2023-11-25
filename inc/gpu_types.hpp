//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_GPU_TYPES_HPP
#define CUTRACE_GPU_TYPES_HPP

#include "vector.hpp"
#include "gpu_variant.hpp"
#include "gpu_array.hpp"
#include "scene_subdiv.hpp"

/**
 * @brief Main namespace for GPU-related code
 */
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
concept object = requires(const T &t, const ray *r, float min_t, vector *p, float *dist, vector *normal) {
  { t.intersect(r, min_t, p, dist, normal) } -> std::same_as<bool>;
  { t.mat_idx } -> std::same_as<const size_t &>;
};

/**
 * @brief A struct representing a single triangle. Triangle corners are expected to be counter-clockwise.
 */
struct triangle {
  vector p1, //!< The first point of the triangle
         p2, //!< The second point of the triangle
         p3; //!< The third point of the triangle
  size_t mat_idx; //!< The index of the material to render the triangle with

  /**
   * @brief (Device) function to check if a ray intersects this triangle.
   * @param[in] r The ray
   * @param[in] min_t The minimal required parametric distance from the ray's origin to consider the intersection
   * @param[out] hit The coordinates of the hit, if any
   * @param[out] dist The parametric distance of the hit, if any
   * @param[out] normal The normal at the point of the hit, if any
   * @return True if there's an intersection, otherwise false.
   */
  __device__ bool intersect(const ray *r, float min_t, vector *hit, float *dist, vector *normal) const;
};

struct triangle_set {
  gpu_array<triangle> triangles;
  size_t mat_idx;
  bound bounding_box;

  __device__ bool intersect(const ray *r, float min_t, vector *hit, float *dist, vector *normal) const;
};

struct plane {
  vector point;
  vector normal;
  size_t mat_idx;

  __device__ bool intersect(const ray *r, float min_t, vector *hit, float *dist, vector *n) const;
};

struct sphere {
  vector center;
  float radius;
  size_t mat_idx;

  __device__ bool intersect(const ray *r, float min_t, vector *hit, float *dist, vector *normal) const;
};

static_assert(object<triangle>);
static_assert(object<triangle_set>);
static_assert(object<plane>);
static_assert(object<sphere>);

using gpu_object = gpu_variant<triangle, triangle_set, plane, sphere>;

struct intersect {
  const ray *r;
  vector out;
  const float min_t;
  float dist;
  vector normal;

  __device__ inline intersect(const ray *r, const float min_t) noexcept :
    r{r}, min_t{min_t}, out{0,0,0}, dist{0.0f}, normal{0.0f, 0.0f, 0.0f} {}

  __device__ inline bool operator()(const triangle *o) {
    return o->intersect(r, min_t, &out, &dist, &normal);
  }

  __device__ inline bool operator()(const triangle_set *o) {
    return o->intersect(r, min_t, &out, &dist, &normal);
  }

  __device__ inline bool operator()(const plane *o) {
    return o->intersect(r, min_t, &out, &dist, &normal);
  }

  __device__ inline bool operator()(const sphere *o) {
    return o->intersect(r, min_t, &out, &dist, &normal);
  }
};

__device__ inline bool intersects(const ray *r, const gpu_object *obj, float min_t, vector *out, float *dist, vector *normal_at) {
  intersect functor(r, min_t);
  auto res = visit(&functor, obj);
  *out = functor.out;
  *dist = functor.dist;
  *normal_at = functor.normal;
  return res;
}

struct sun {
  vector direction;
  vector color;

  __device__ void direction_to(const vector *point, vector *direction, float *distance) const;
};

struct point_light {
  vector point;
  vector color;

  __device__ void direction_to(const vector *point, vector *direction, float *distance) const;
};

using gpu_light = gpu_variant<sun, point_light>;

struct director {
  const vector *point;
  vector direction;
  float distance;

  __device__ inline void operator()(const sun *l) {
    return l->direction_to(point, &direction, &distance);
  }

  __device__ inline void operator()(const point_light *l) {
    return l->direction_to(point, &direction, &distance);
  }
};

__device__ inline void direction_to(const vector *point, const gpu_light *light, vector *direction, float *distance) {
  director functor(point, {}, 0.0f);
  visit(&functor, light);
  *direction = functor.direction;
  *distance = functor.distance;
}

struct gpu_mat {
  vector color;
  float specular, reflexivity, phong_exp, transparency;
};

struct cam {
  vector pos;
  vector up;
  vector forward;
  vector right;
  float near, far;
  size_t w, h;

  __host__ void look_at(const vector &v);
};

struct gpu_scene {
  gpu_array<gpu_object> objects;
  gpu_array<gpu_light> lights;
  gpu_array<gpu_mat> materials;
};
}

#endif //CUTRACE_GPU_TYPES_HPP
