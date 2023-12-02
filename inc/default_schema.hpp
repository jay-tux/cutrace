//
// Created by jay on 11/29/23.
//

#ifndef CUTRACE_DEFAULT_SCHEMA_HPP
#define CUTRACE_DEFAULT_SCHEMA_HPP

#include <vector>
#include "vector.hpp"
#include "loader.hpp"
#include "gpu_types_.hpp"
#include "cpu_to_gpu.hpp"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include "ray_cast.hpp"

namespace cutrace::gpu::schema {
/**
 * @brief A struct representing a single triangle. Triangle corners are expected to be counter-clockwise.
 */
struct triangle {
  vector p1, //!< The first point of the triangle
         p2, //!< The second point of the triangle
         p3; //!< The third point of the triangle
  size_t mat_idx; //!< The index of the material to render the triangle with

  __device__ constexpr uv uv_for(const vector *point) const {
    vector p2p1 = p2 - p1, p3p1 = p3 - p1, xp1 = *point - p1;
    vector proj_u = xp1.dot(p2p1) / p2p1.dot(p2p1) * p2p1,
           proj_v = xp1.dot(p3p1) / p3p1.dot(p3p1) * p3p1;

    return {
      .u = proj_u.norm() / p2p1.norm(),
      .v = proj_v.norm() / p3p1.norm()
    };
  }

  /**
   * @brief Function to check if a ray intersects this triangle.
   * @param[in] r The ray
   * @param[in] min_t The minimal required parametric distance from the ray's origin to consider the intersection
   * @param[out] hit The coordinates of the hit, if any
   * @param[out] dist The parametric distance of the hit, if any
   * @param[out] normal The normal at the point of the hit, if any
   * @return True if there's an intersection, otherwise false.
   */
  __device__ constexpr bool intersect(const ray *r, float min_t, vector *hit, float *dist, vector *normal, uv *tex_coords) const {
    auto a = p1 - p2, b = p2 - p3,
         c = r->dir, d =p2 - r->start;

    matrix A{{a,b,d}}, B{{a,b,c}},
           A1{{d,b,c}}, A2{{a,d,c}};

    float alpha = B.determinant();
    float beta = A1.determinant() / alpha;
    float gamma = A2.determinant() / alpha;
    float t0 = A.determinant() / alpha;

    if(beta >= 0 && gamma >= 0 && beta + gamma <= 1 && isfinite(t0) && min_t <= t0) {
      *dist = t0;
      *hit = r->start + *dist * r->dir;
      *normal = -1.0f * (p2 - p3).cross(p1 - p3).normalized();
      *tex_coords = uv_for(hit);
      return true;
    }

    return false;
  }

  __host__ inline void gpu_clean() {}
};

/**
 * @brief A struct representing a set of triangles (usually loaded as a model).
 */
struct mesh {
  gpu_array<triangle> triangles; //!< The triangles
  size_t mat_idx; //!< The index of the material to render the model with
  bound bounding_box; //!< The bounding box of the model, required for optimization

  __device__ constexpr bool bound_intersects(const ray *r) const {
    // from https://tavianator.com/2022/ray_box_boundary.html
    float tmin = 0.0, tmax = INFINITY;

    vector r_inv = { 1.0f / r->dir.x, 1.0f / r->dir.y, 1.0f / r->dir.z };

    for (int d = 0; d < 3; ++d) {
      float t1 = (bounding_box.min[d] - r->start[d]) * r_inv[d];
      float t2 = (bounding_box.max[d] - r->start[d]) * r_inv[d];

      tmin = min(max(t1, tmin), max(t2, tmin));
      tmax = max(min(t1, tmax), min(t2, tmax));
    }

    return tmin <= tmax;
  }

  /**
   * @brief Function to check if a ray intersects any triangle of this model.
   * @param[in] r The ray
   * @param[in] min_t The minimal required parametric distance from the ray's origin to consider the intersection
   * @param[out] hit The coordinates of the hit, if any
   * @param[out] dist The parametric distance of the hit, if any
   * @param[out] normal The normal at the point of the hit, if any
   * @return True if there's an intersection, otherwise false.
   */
  __device__ constexpr bool intersect(const ray *r, float min_t, vector *hit, float *dist, vector *normal, uv *tex_coords) const {
    if(!bound_intersects(r)) return false;

    vector h{}, n{};
    uv unused{};
    float t;
    *dist = INFINITY;

    for(const auto &tri: triangles) {
      if(tri.intersect(r, min_t, &h, &t, &n, &unused) && t < *dist) {
        *dist = t;
        *hit = h;
        *normal = n;
        tex_coords->u = hit->x;
        tex_coords->v = hit->y;
      }
    }

    return *dist != INFINITY;
  }

  __host__ inline void gpu_clean() {
    cudaCheck(cudaFree(triangles.buffer))
    triangles.buffer = nullptr;
    triangles.size = 0;
  }
};

/**
 * @brief Struct representing an infinite plane.
 */
struct plane {
  vector point; //!< A point of this plane
  vector normal; //!< The normal direction of this plane
  size_t mat_idx; //!< The material index to render this plane with

  __device__ constexpr uv uv_for(const vector *p) const {
    vector ax1 = vector{ normal.y, -normal.x, 0.0f }.normalized();
    vector ax2 = normal.cross(ax1);
    vector mod_pt = point - *p;

    return {
      .u = ax1.dot(mod_pt),
      .v = ax2.dot(mod_pt)
    };
  }

  /**
   * @brief Function to check if a ray intersects this plane.
   * @param[in] r The ray
   * @param[in] min_t The minimal required parametric distance from the ray's origin to consider the intersection
   * @param[out] hit The coordinates of the hit, if any
   * @param[out] dist The parametric distance of the hit, if any
   * @param[out] normal The normal at the point of the hit, if any
   * @return True if there's an intersection, otherwise false.
   */
  __device__ constexpr bool intersect(const ray *r, float min_t, vector *hit, float *dist, vector *n, uv *tex_coords) const {
    float t0 = (point - r->start).dot(normal) / r->dir.dot(normal);

    if(isfinite(t0) && min_t <= t0) {
      *dist = t0;
      *hit = r->start + t0 * r->dir;
      *n = this->normal;
      *tex_coords = uv_for(hit);
      return true;
    }

    return false;
  }

  __host__ inline void gpu_clean() {}
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
  __device__ constexpr bool intersect(const ray *r, float min_t, vector *hit, float *dist, vector *normal, uv *tex_coords) const {
    auto d = r->dir.normalized(), c = center, e = r->start;
    float R = radius;

    float dec = -d.dot(e - c);
    float sub = dec * dec - d.dot(d) * ((e - c).dot(e - c) * R * R);

    float t0 = (dec - sqrt(sub)) / d.dot(d), t1 = (dec + sqrt(sub)) / d.dot(d);
    bool t0v = isfinite(t0) && min_t <= t0, t1v = isfinite(t1) && min_t <= t1;
    int condition = (t0v ? 2 : 0) + (t1v ? 1 : 0);

    switch(condition) {
      case 0: return false;
      case 1: *dist = t1; break;
      case 2: *dist = t0; break;
      case 3: *dist = min(t0, t1); break;
      default: break; // impossible
    }

    *hit = r->start + *dist * r->dir.normalized();
    *normal = (*hit - c).normalized();
    vector delta = (*hit - center).normalized();
    tex_coords->u = 0.5f + (atan2(delta.z, delta.x) / (2.0f * (float)M_PI));
    tex_coords->v = 0.5f + (asin(delta.y) / (float)M_PI);
    return true;
  }

  __host__ inline void gpu_clean() {}
};

static_assert(is_object<triangle>);
static_assert(is_object<mesh>);
static_assert(is_object<plane>);
static_assert(is_object<sphere>);

/**
 * @brief Struct representing a sun (directional light).
 */
struct sun {
  vector direction; //!< The direction of the light
  vector color; //!< The color of the light

  /**
   * @brief Gets the direction from a point towards the light, as well as the parametric distance.
   * @param [in] point The query point
   * @param [out] out_dir The direction from the point towards the light
   * @param [out] distance The parametric distance
   *
   * The actual distance can be computed by multiplying the output parametric distance by the direction's norm.
   * In this case, the direction is always `-this->direction` and the distance is always considered to be infinite.
   */
  __device__ inline void direction_to(const vector *point, vector *out_dir, float *distance) const {
    *out_dir = -1.0f * direction;
    *distance = INFINITY;
  }

  __host__ inline void gpu_clean() {}
};

/**
 * @brief Struct representing a point light.
 */
struct point_light {
  vector point; //!< The point where the light is shining from
  vector color; //!< The color of the light

  /**
   * @brief Gets the direction from a point towards the light, as well as the parametric distance.
   * @param [in] pt The query point
   * @param [out] direction The direction from the point towards the light
   * @param [out] distance The parametric distance
   *
   * The actual distance can be computed by multiplying the output parametric distance by the direction's norm.
   * In this case, the direction is always the difference `*point - *this->point` normalized, and the distance is the
   * non-normalized norm of the difference above.
   */
  __device__ inline void direction_to(const vector *pt, vector *direction, float *distance) const {
    *direction = (point - *pt).normalized();
    *distance = (point - *pt).norm();
  }

  __host__ inline void gpu_clean() {}
};

static_assert(is_light<sun>);
static_assert(is_light<point_light>);

#define I_A 0.1

/**
 * @brief Struct representing a material.
 */
struct phong_material {
  vector color; //!< The base color of the material
  float specular, //!< The specular factor for the material (how smooth/shiny it is)
        reflexivity, //!< The reflexivity factor for the material (how much it reflects/mirrors)
        phong_exp, //!< The Phong lighting exponent for the material
        transparency; //!< The transparency/translucency factor for the material

  __device__ inline void get_phong_params(const vector *, const uv *, vector *color_out, vector *spec, float *ref, float *tran, float *phong) const {
    *color_out = color;
    *spec = specular * color;
    *ref = reflexivity;
    *tran = transparency;
    *phong = phong_exp;
  }

  __host__ __device__ constexpr bool is_transparent() const { return transparency >= 1e-6; }
  __host__ __device__ constexpr bool is_reflecting() const { return reflexivity >= 1e-6; }

  __device__ inline void get_bounce_params(const vector *, const uv *, float *ref, float *trans) const {
    *ref = reflexivity;
    *trans = transparency;
  }

  __host__ inline void gpu_clean() {}
};

static_assert(is_material<phong_material>);

/**
 * @brief Struct representing a camera
 */
struct cam {
  vector pos = { 0.0f, 0.0f, 0.0f }; //!< Eye position of the camera
  vector up = { 0.0f, 1.0f, 0.0f }; //!< Up direction for the camera
  vector forward = { 0.0f, 0.0f, 1.0f }; //!< Forward direction for the camera (look-at)
  vector right = { 1.0f, 0.0f, 0.0f }; //!< Right direction for the camera
  float near = 0.1f, //!< Distance to the near plane (unused)
        far = 100.0f, //!< Distance to the far plane (unused)
        ambient = 0.1f;
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
  __host__ inline void look_at(const vector &v) {
    forward = (v - pos).normalized();
    right = forward.cross(up).normalized(); // perpendicular to plane formed by forward & up
    up = right.cross(forward).normalized(); // make them all perpendicular to each other
  }

  __host__ __device__ constexpr ray get_ray(size_t x, size_t y) const {
    float aspect = (float)w / (float)h;
    return {
      .start = pos,
      .dir = (forward + ((float)x / (float)w - 0.5f) * aspect * right + ((float)y / (float)h) * up).normalized()
    };
  }

  __host__ __device__ inline void get_bounds(size_t *w, size_t *h) const {
    *w = this->w;
    *h = this->h;
  }

  __host__ __device__ constexpr float get_ambient() const { return ambient; }

  __host__ inline void gpu_clean() {}
};

static_assert(is_camera<cam>);
}

namespace cutrace::cpu::schema {
namespace defaults {
struct black {
  constexpr const static vector value{0.0f, 0.0f, 0.0f};
};

struct up {
  constexpr const static vector value{0.0f, 1.0f, 0.0f};
};

struct forward {
  constexpr const static vector value{0.0f, 0.0f, 1.0f};
};

struct right {
  constexpr const static vector value{1.0f, 0.0f, 0.0f};
};

struct white {
  constexpr const static vector value{1.0f, 1.0f, 1.0f};
};

#define DEFAULT_FLOAT(name, init) struct name { constexpr const static float value = init; }
DEFAULT_FLOAT(zero, 0.0f);
DEFAULT_FLOAT(point_one, 0.1f);
DEFAULT_FLOAT(point_three, 0.3f);
DEFAULT_FLOAT(thirty_two, 32.0f);
DEFAULT_FLOAT(one_hundred, 100.0f);
#undef DEFAULT_FLOAT

struct default_width : std::integral_constant<size_t, 1920> {};
struct default_height : std::integral_constant<size_t, 1080> {};
}

#define ARGUMENT(name, type) loader_argument<name, type, mandatory>
#define OPTIONAL(name, type, def) loader_argument<name, type, def>

struct triangle {
  vector p1, p2, p3;
  size_t mat_idx;

  constexpr triangle(vector p1, vector p2, vector p3, size_t mat_idx)
          : p1{p1}, p2{p2}, p3{p3}, mat_idx{mat_idx} {}

  [[nodiscard]] inline gpu::schema::triangle to_gpu() const {
    return { p1.to_gpu(), p2.to_gpu(), p3.to_gpu(), mat_idx };
  }

  constexpr const static char name[] = "triangle";
  constexpr const static char arg_p1[] = "p1";
  constexpr const static char arg_p2[] = "p2";
  constexpr const static char arg_p3[] = "p3";
  constexpr const static char arg_mat_idx[] = "material";

  using schema = object_schema<name, triangle,
    ARGUMENT(arg_p1, vector),
    ARGUMENT(arg_p2, vector),
    ARGUMENT(arg_p3, vector),
    ARGUMENT(arg_mat_idx, size_t)
  >;
};

struct mesh {
  std::vector<triangle> tris;
  size_t mat_idx;

  inline mesh(const std::string &file, size_t mat_idx) : tris{}, mat_idx{mat_idx} {
    Assimp::Importer imp;
    const aiScene *scene = imp.ReadFile(
            file.c_str(),
            aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType
    );

    if(scene == nullptr) return;

    for(size_t mesh_id = 0; mesh_id < scene->mNumMeshes; mesh_id++) {
      const auto *mesh = scene->mMeshes[mesh_id];

      if(mesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE) continue; // skip

      tris.reserve(tris.size() + mesh->mNumFaces);
      for(size_t face_id = 0; face_id < mesh->mNumFaces; face_id++) {
        const auto &face = mesh->mFaces[face_id];
        if(face.mNumIndices != 3) continue; // skip

#define VERT(idx, comp) mesh->mVertices[face.mIndices[idx]].comp
        tris.push_back(triangle(
                { VERT(0, x), VERT(0, y), VERT(0, z) },
                { VERT(1, x), VERT(1, y), VERT(1, z) },
                { VERT(2, x), VERT(2, y), VERT(2, z) },
                mat_idx
        ));
#undef VERT
      }
    }
  }

  __host__ constexpr static float min3(float f1, float f2, float f3) {
    return std::min(std::min(f1, f2), f3);
  }

  __host__ constexpr static float max3(float f1, float f2, float f3) {
    return std::max(std::max(f1, f2), f3);
  }

  [[nodiscard]] __host__ inline bound bounding_box() const {
    bound res = bound::incorrect();
    for(const auto &tri: tris) {
#define MIN_COMP(c) min3(tri.p1.c, tri.p2.c, tri.p3.c)
#define MAX_COMP(c) max3(tri.p1.c, tri.p2.c, tri.p3.c)
      res.merge(bound{
        .min = { MIN_COMP(x), MIN_COMP(y), MIN_COMP(z) },
        .max = { MAX_COMP(x), MAX_COMP(y), MAX_COMP(z) }
      });
#undef MIN_COMP
#undef MAX_COMP
    }
    return res;
  }

  [[nodiscard]] inline gpu::schema::mesh to_gpu() const {
    return { cpu2gpu::vec_to_gpu<triangle, gpu::schema::triangle>(tris), mat_idx, bounding_box() };
  }

  constexpr const static char name[] = "mesh";
  constexpr const static char arg_file[] = "file";
  constexpr const static char arg_mat_idx[] = "material";

  using schema = object_schema<name, mesh,
    ARGUMENT(arg_file, std::string),
    ARGUMENT(arg_mat_idx, size_t)
  >;
};

struct plane {
  vector point, normal;
  size_t mat_idx;

  constexpr plane(vector point, vector normal, size_t mat_idx) : point{point}, normal{normal}, mat_idx{mat_idx} {}

  [[nodiscard]] inline gpu::schema::plane to_gpu() const {
    return { point.to_gpu(), normal.to_gpu(), mat_idx };
  }

  constexpr const static char name[] = "plane";
  constexpr const static char arg_point[] = "point";
  constexpr const static char arg_normal[] = "normal";
  constexpr const static char arg_mat_idx[] = "material";

  using schema = object_schema<name, plane,
    ARGUMENT(arg_point, vector),
    ARGUMENT(arg_normal, vector),
    ARGUMENT(arg_mat_idx, size_t)
  >;
};

struct sphere {
  vector center;
  float radius;
  size_t mat_idx;

  constexpr sphere(vector center, float radius, size_t mat_idx) : center{center}, radius{radius}, mat_idx{mat_idx} {}

  [[nodiscard]] inline gpu::schema::sphere to_gpu() const {
    return { center.to_gpu(), radius, mat_idx };
  }

  constexpr const static char name[] = "sphere";
  constexpr const static char arg_center[] = "center";
  constexpr const static char arg_radius[] = "radius";
  constexpr const static char arg_mat_idx[] = "material";

  using schema = object_schema<name, sphere,
    ARGUMENT(arg_center, vector),
    ARGUMENT(arg_radius, float),
    ARGUMENT(arg_mat_idx, size_t)
  >;
};

static_assert(cpu2gpu::cpu_gpu_object_pair<triangle, gpu::schema::triangle>);
static_assert(cpu2gpu::cpu_gpu_object_pair<mesh, gpu::schema::mesh>);
static_assert(cpu2gpu::cpu_gpu_object_pair<plane, gpu::schema::plane>);
static_assert(cpu2gpu::cpu_gpu_object_pair<sphere, gpu::schema::sphere>);

using default_objects_schema = all_objects_schema<triangle::schema, mesh::schema, plane::schema, sphere::schema>;

struct sun {
  vector direction, color;

  constexpr sun(vector direction, vector color) : direction{direction}, color{color} {}

  [[nodiscard]] inline gpu::schema::sun to_gpu() const {
    return { direction.to_gpu(), color.to_gpu() };
  }

  constexpr const static char name[] = "sun";
  constexpr const static char arg_direction[] = "direction";
  constexpr const static char arg_color[] = "color";

  using schema = light_schema<name, sun,
    ARGUMENT(arg_direction, vector),
    OPTIONAL(arg_color, vector, defaults::white)
  >;
};

struct point_light {
  vector point, color;

  constexpr point_light(vector point, vector color) : point{point}, color{color} {}

  [[nodiscard]] inline gpu::schema::point_light to_gpu() const {
    return { point.to_gpu(), color.to_gpu() };
  }

  constexpr const static char name[] = "point";
  constexpr const static char arg_point[] = "point";
  constexpr const static char arg_color[] = "color";

  using schema = light_schema<name, point_light,
    ARGUMENT(arg_point, vector),
    OPTIONAL(arg_color, vector, defaults::white)
  >;
};

static_assert(cpu2gpu::cpu_gpu_light_pair<sun, gpu::schema::sun>);
static_assert(cpu2gpu::cpu_gpu_light_pair<point_light, gpu::schema::point_light>);

using default_lights_schema = all_lights_schema<sun::schema, point_light::schema>;

struct solid_material {
  vector color;
  float specular, reflexivity, phong_exp, transparency;

  constexpr solid_material(vector color, float s, float r, float p, float t) :
    color{color}, specular{s}, reflexivity{r}, phong_exp{p}, transparency{t} {}

  [[nodiscard]] inline gpu::schema::phong_material to_gpu() const {
    return { color.to_gpu(), specular, reflexivity, phong_exp, transparency };
  }

  constexpr const static char name[] = "solid";
  constexpr const static char arg_color[] = "color";
  constexpr const static char arg_spec[] = "specular";
  constexpr const static char arg_refl[] = "reflect";
  constexpr const static char arg_phong[] = "phong";
  constexpr const static char arg_trans[] = "transparency";

  using schema = material_schema<name, solid_material,
    ARGUMENT(arg_color, vector),
    OPTIONAL(arg_spec, float, defaults::point_three),
    OPTIONAL(arg_refl, float, defaults::zero),
    OPTIONAL(arg_phong, float, defaults::thirty_two),
    OPTIONAL(arg_trans, float, defaults::zero)
  >;
};

static_assert(cpu2gpu::cpu_gpu_material_pair<solid_material, gpu::schema::phong_material >);

using default_material_schema = all_materials_schema<solid_material::schema>;

struct default_cam {
  vector pos = defaults::black::value;
  vector up = defaults::up::value;
  vector look = defaults::forward::value;
  float near = defaults::point_one::value;
  float far = defaults::one_hundred::value;
  float ambient = defaults::point_one::value;
  size_t w = defaults::default_width::value;
  size_t h = defaults::default_height::value;

  constexpr default_cam() = default;
  constexpr default_cam(vector e, vector u, vector l, float n, float f, size_t w, size_t h, float ambient)
    : pos{e}, up{u}, look{l}, near{n}, far{f}, w{w}, h{h}, ambient{ambient} {}

  [[nodiscard]] inline gpu::schema::cam to_gpu() const {
    gpu::schema::cam res{ pos.to_gpu(), up.to_gpu(), {}, {}, near, far, ambient, w, h };
    res.look_at(look.to_gpu());
    return res;
  }

  constexpr const static char arg_pos[] = "eye";
  constexpr const static char arg_up[] = "up";
  constexpr const static char arg_look[] = "look";
  constexpr const static char arg_near[] = "near_plane";
  constexpr const static char arg_far[] = "far_plane";
  constexpr const static char arg_w[] = "width";
  constexpr const static char arg_h[] = "height";
  constexpr const static char arg_ambient[] = "ambient";

  using schema = cam_schema<default_cam,
    OPTIONAL(arg_pos, vector, defaults::black),
    OPTIONAL(arg_up, vector, defaults::up),
    OPTIONAL(arg_look, vector, defaults::forward),
    OPTIONAL(arg_near, float, defaults::point_one),
    OPTIONAL(arg_far, float, defaults::one_hundred),
    OPTIONAL(arg_w, size_t, defaults::default_width),
    OPTIONAL(arg_h, size_t, defaults::default_height),
    OPTIONAL(arg_ambient, float, defaults::point_one)
  >;
};

static_assert(cpu2gpu::cpu_gpu_camera_pair<default_cam, gpu::schema::cam>);

using default_schema = full_schema<default_objects_schema, default_lights_schema, default_material_schema, default_cam::schema>;

inline default_schema::scene_t load_default(const std::string &file) {
  return default_schema::load_file(file);
}

using default_cpu_object = cpu_object_set<triangle, mesh, plane, sphere>;
using default_cpu_light = cpu_light_set<sun, point_light>;
using default_cpu_material = cpu_material_set<solid_material>;
using default_cpu_cam = default_cam;
using default_gpu_object = gpu::gpu_object_set<gpu::schema::triangle, gpu::schema::mesh, gpu::schema::plane, gpu::schema::sphere>;
using default_gpu_light = gpu::gpu_light_set<gpu::schema::sun, gpu::schema::point_light>;
using default_gpu_material = gpu::gpu_material_set<gpu::schema::phong_material>;
using deafult_gpu_cam = gpu::schema::cam;

using default_cpu_scene = cpu_scene_<default_cpu_object, default_cpu_light, default_cpu_material, default_cpu_cam>;
using default_gpu_scene = gpu::gpu_scene_<default_gpu_object, default_gpu_light, default_gpu_material, deafult_gpu_cam>;

using default_converter = cpu2gpu::cpu_to_gpu<default_cpu_scene, default_gpu_scene>;

inline default_gpu_scene default_to_gpu(const default_cpu_scene &cpu) {
  return default_converter::convert(cpu);
}
}

#endif //CUTRACE_DEFAULT_SCHEMA_HPP
