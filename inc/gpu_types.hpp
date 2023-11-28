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
concept is_object = requires(const T &t, const ray *r, float min_t, vector *p, float *dist, vector *normal) {
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
   * @brief Function to check if a ray intersects this triangle.
   * @param[in] r The ray
   * @param[in] min_t The minimal required parametric distance from the ray's origin to consider the intersection
   * @param[out] hit The coordinates of the hit, if any
   * @param[out] dist The parametric distance of the hit, if any
   * @param[out] normal The normal at the point of the hit, if any
   * @return True if there's an intersection, otherwise false.
   */
  __device__ bool intersect(const ray *r, float min_t, vector *hit, float *dist, vector *normal) const;
};

/**
 * @brief A struct representing a set of triangles (usually loaded as a model).
 * @warning Candidate to be renamed to `model`
 */
struct triangle_set {
  gpu_array<triangle> triangles; //!< The triangles
  size_t mat_idx; //!< The index of the material to render the model with
  bound bounding_box; //!< The bounding box of the model, required for optimization

  /**
   * @brief Function to check if a ray intersects any triangle of this model.
   * @param[in] r The ray
   * @param[in] min_t The minimal required parametric distance from the ray's origin to consider the intersection
   * @param[out] hit The coordinates of the hit, if any
   * @param[out] dist The parametric distance of the hit, if any
   * @param[out] normal The normal at the point of the hit, if any
   * @return True if there's an intersection, otherwise false.
   */
  __device__ bool intersect(const ray *r, float min_t, vector *hit, float *dist, vector *normal) const;
};

/**
 * @brief Struct representing an infinite plane.
 */
struct plane {
  vector point; //!< A point of this plane
  vector normal; //!< The normal direction of this plane
  size_t mat_idx; //!< The material index to render this plane with

  /**
   * @brief Function to check if a ray intersects this plane.
   * @param[in] r The ray
   * @param[in] min_t The minimal required parametric distance from the ray's origin to consider the intersection
   * @param[out] hit The coordinates of the hit, if any
   * @param[out] dist The parametric distance of the hit, if any
   * @param[out] normal The normal at the point of the hit, if any
   * @return True if there's an intersection, otherwise false.
   */
  __device__ bool intersect(const ray *r, float min_t, vector *hit, float *dist, vector *n) const;
};

/**
 * @brief Struct representing a sphere.
 */
struct sphere {
  vector center; //!< The center point of the sphere
  float radius; //!< The radius of the sphere
  size_t mat_idx; //!< The index of the material to render this sphere with

  /**
   * @brief Function to check if a ray intersects this sphere.
   * @param[in] r The ray
   * @param[in] min_t The minimal required parametric distance from the ray's origin to consider the intersection
   * @param[out] hit The coordinates of the hit, if any
   * @param[out] dist The parametric distance of the hit, if any
   * @param[out] normal The normal at the point of the hit, if any
   * @return True if there's an intersection, otherwise false.
   */
  __device__ bool intersect(const ray *r, float min_t, vector *hit, float *dist, vector *normal) const;
};

static_assert(is_object<triangle>);
static_assert(is_object<triangle_set>);
static_assert(is_object<plane>);
static_assert(is_object<sphere>);

/**
 * @brief Type alias for a gpu object: either a @ref{cutrace::gpu::triangle}, @ref{cutrace::gpu::triangle_set}, @ref{cutrace::gpu::plane}, or @ref{cutrace::gpu::sphere}.
 */
using gpu_object = gpu_variant<triangle, triangle_set, plane, sphere>;

/**
 * @brief Visitor struct to check for intersections (to be replaced).
 * @warning Will be deprecated in favor of a lambda function inside the @ref{cutrace::gpu::intersects} function.
 */
struct intersect {
  const ray *r; //!< The input ray
  vector out; //!< The output hit point, if any
  const float min_t; //!< The input minimal distance
  float dist; //!< The output parametric distance to the hit point, if any
  vector normal; //!< The output normal vector in the hit point, if any

  /**
   * @brief Creates a new intersection visitor.
   * @param r The ray to intersect with
   * @param min_t The minimal parametric distance
   */
  __device__ inline intersect(const ray *r, const float min_t) noexcept :
    r{r}, min_t{min_t}, out{0,0,0}, dist{0.0f}, normal{0.0f, 0.0f, 0.0f} {}

  /**
   * @brief Performs the intersection detection on a \ref cutrace::gpu::triangle.
   * @param o The triangle to intersect with
   * @return True if there is an intersection, false otherwise
   * @see cutrace::gpu::triangle::intersect
   */
  __device__ inline bool operator()(const triangle *o) {
    return o->intersect(r, min_t, &out, &dist, &normal);
  }

  /**
   * @brief Performs the intersection detection on a \ref cutrace::gpu::triangle_set.
   * @param o The model to intersect with
   * @return True if there is an intersection, false otherwise
   * @see cutrace::gpu::triangle_set::intersect
   */
  __device__ inline bool operator()(const triangle_set *o) {
    return o->intersect(r, min_t, &out, &dist, &normal);
  }

  /**
   * @brief Performs the intersection detection on a \ref cutrace::gpu::plane.
   * @param o The plane to intersect with
   * @return True if there is an intersection, false otherwise
   * @see cutrace::gpu::plane::intersect
   */
  __device__ inline bool operator()(const plane *o) {
    return o->intersect(r, min_t, &out, &dist, &normal);
  }

  /**
   * @brief Performs the intersection detection on a \ref cutrace::gpu::sphere.
   * @param o The sphere to intersect with
   * @return True if there is an intersection, false otherwise
   * @see cutrace::gpu::sphere::intersect
   */
  __device__ inline bool operator()(const sphere *o) {
    return o->intersect(r, min_t, &out, &dist, &normal);
  }
};

/**
 * @brief Function which performs intersection detection on a \ref cutrace::gpu::gpu_object.
 * @param[in] r The ray
 * @param[in] obj The object to check for intersections with
 * @param[in] min_t The minimal required parametric distance from the ray's origin to consider the intersection
 * @param[out] out The coordinates of the hit, if any
 * @param[out] dist The parametric distance of the hit, if any
 * @param[out] normal_at The normal at the point of the hit, if any
 * @return True if there's an intersection, otherwise false.
 */
__device__ inline bool intersects(const ray *r, const gpu_object *obj, float min_t, vector *out, float *dist, vector *normal_at) {
  intersect functor(r, min_t);
  auto res = visit(&functor, obj);
  *out = functor.out;
  *dist = functor.dist;
  *normal_at = functor.normal;
  return res;
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
concept is_light = requires(const T &t, const vector *point, vector *dir, float *dist) {
  { t.direction_to(point, dir, dist) } -> std::same_as<void>;
  { t.color } -> std::same_as<const vector &>;
};

/**
 * @brief Struct representing a sun (directional light).
 */
struct sun {
  vector direction; //!< The direction of the light
  vector color; //!< The color of the light

  /**
   * @brief Gets the direction from a point towards the light, as well as the parametric distance.
   * @param [in] point The query point
   * @param [out] direction The direction from the point towards the light
   * @param [out] distance The parametric distance
   *
   * The actual distance can be computed by multiplying the output parametric distance by the direction's norm.
   * In this case, the direction is always `-this->direction` and the distance is always considered to be infinite.
   */
  __device__ void direction_to(const vector *point, vector *direction, float *distance) const;
};

/**
 * @brief Struct representing a point light.
 */
struct point_light {
  vector point; //!< The point where the light is shining from
  vector color; //!< The color of the light

  /**
   * @brief Gets the direction from a point towards the light, as well as the parametric distance.
   * @param [in] point The query point
   * @param [out] direction The direction from the point towards the light
   * @param [out] distance The parametric distance
   *
   * The actual distance can be computed by multiplying the output parametric distance by the direction's norm.
   * In this case, the direction is always the difference `*point - *this->point` normalized, and the distance is the
   * non-normalized norm of the difference above.
   */
  __device__ void direction_to(const vector *point, vector *direction, float *distance) const;
};

static_assert(is_light<sun>);
static_assert(is_light<point_light>);

/**
 * @brief Type alias for a gpu light: either a \ref cutrace::gpu::sun, or a \ref cutrace::gpu::point_light.
 */
using gpu_light = gpu_variant<sun, point_light>;

/**
 * @brief Visitor struct to get the direction towards a light (to be replaced).
 * @warning Will be deprecated in favor of a lambda function inside the \ref cutrace::gpu::direction_to function.
 */
struct director {
  const vector *point; //!< The input point
  vector direction; //!< The output direction
  float distance; //!< The output parametric distance

  /**
   * @brief Gets the direction towards a \ref cutrace::gpu::sun.
   * @param l The sun/directional light to get the direction towards
   * @see cutrace::gpu::sun::direction_to
   */
  __device__ inline void operator()(const sun *l) {
    return l->direction_to(point, &direction, &distance);
  }

  /**
   * @brief Gets the direction towards a \ref cutrace::gpu::point_light.
   * @param l The sun/directional light to get the direction towards
   * @see cutrace::gpu::point_light::direction_to
   */
  __device__ inline void operator()(const point_light *l) {
    return l->direction_to(point, &direction, &distance);
  }
};

/**
 * @brief Function to get the direction to a light.
 * @param [in] point The query point
 * @param [in] light The light to get the direction towards
 * @param [out] direction The direction from the point towards the light
 * @param [out] distance The parametric distance along the direction
 */
__device__ inline void direction_to(const vector *point, const gpu_light *light, vector *direction, float *distance) {
  director functor(point, {}, 0.0f);
  visit(&functor, light);
  *direction = functor.direction;
  *distance = functor.distance;
}

/**
 * @brief Struct representing a material.
 */
struct gpu_mat {
  vector color; //!< The base color of the material
  float specular, //!< The specular factor for the material (how smooth/shiny it is)
        reflexivity, //!< The reflexivity factor for the material (how much it reflects/mirrors)
        phong_exp, //!< The Phong lighting exponent for the material
        transparency; //!< The transparency/translucency factor for the material
};

/**
 * @brief Struct representing a camera
 */
struct cam {
  vector pos = { 0.0f, 0.0f, 0.0f }; //!< Eye position of the camera
  vector up = { 0.0f, 1.0f, 0.0f }; //!< Up direction for the camera
  vector forward = { 0.0f, 0.0f, 1.0f }; //!< Forward direction for the camera (look-at)
  vector right = { 1.0f, 0.0f, 0.0f }; //!< Right direction for the camera
  float near = 0.1f, //!< Distance to the near plane (unused)
        far = 100.0f; //!< Distance to the far plane (unused)
  size_t w = 1920, //!< The width of the image to be rendered
         h = 1080; //!< The height of the image to be rendered

  /**
   * @brief Computes all directions, given a point to look at.
   * @param [in] v The point to look at
   *
   * This function requires an estimate of the up direction, and computes (in this order):
   *  - The forward direction,
   *  - The right direction (by using a cross-product between forward and up),
   *  - The (correct) up direction (by using a cross-product between up and right).
   */
  __host__ void look_at(const vector &v);
};

/**
 * @brief Struct containing all information to render a scene on GPU.
 */
struct gpu_scene {
  gpu_array<gpu_object> objects; //!< The objects to be rendered
  gpu_array<gpu_light> lights; //!< The lights to render with
  gpu_array<gpu_mat> materials; //!< All used materials in the scene
};
}

#endif //CUTRACE_GPU_TYPES_HPP
