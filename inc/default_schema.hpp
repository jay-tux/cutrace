//
// Created by jay on 11/29/23.
//

#ifndef CUTRACE_DEFAULT_SCHEMA_HPP
#define CUTRACE_DEFAULT_SCHEMA_HPP

#include <vector>
#include "vector.hpp"
#include "loader.hpp"
#include "gpu_types.hpp"
#include "cpu_to_gpu.hpp"
#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"
#include "ray_cast.hpp"
#include "schema_view.hpp"

/**
 * @brief Namespace for the default schema types on GPU.
 */
namespace cutrace::gpu::schema {
/**
 * @brief A struct representing a single triangle. Triangle corners are expected to be counter-clockwise.
 */
struct triangle {
  vector p1, //!< The first point of the triangle
         p2, //!< The second point of the triangle
         p3; //!< The third point of the triangle
  size_t mat_idx; //!< The index of the material to render the triangle with

  /**
   * @brief Gets the UV-coordinates for a given point.
   * @param point The point
   * @return The UV-coordinates
   */
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
    auto a = p2 - p1, b = p2 - p3,
         c = r->dir, d = p2 - r->start;

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

  /**
   * @brief Cleans up all used resources on GPU
   */
  __host__ inline void gpu_clean() {}
};

/**
 * @brief A struct representing a set of triangles (usually loaded as a model).
 */
struct mesh {
  gpu_array<triangle> triangles; //!< The triangles
  size_t mat_idx; //!< The index of the material to render the model with
  bound bounding_box; //!< The bounding box of the model, required for optimization

  /**
   * @brief Checks if a ray intersects the mesh's bounding box.
   * @param r The ray
   * @return True if there's an intersection, false otherwise
   */
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

  /**
   * @brief Cleans up all resources used on the GPU.
   */
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

  /**
   * @brief Gets the UV-coordinates for a point on the plane.
   * @param p The point
   * @return The UV-coordinates
   */
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

  /**
   * @brief Cleans up all resources used on the GPU.
   */
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
    float sub = dec * dec - d.dot(d) * ((e - c).dot(e - c) - R * R);

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

  /**
   * @brief Cleans up all resources used on the GPU.
   */
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

//static_assert(is_light<sun>);
//static_assert(is_light<point_light>);

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

//static_assert(is_material<phong_material>);

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
    vector x_v = (((float)x / (float)w) - 0.5f) * aspect * right;
    vector y_v = (0.5f - ((float)y / (float)h)) * up;
    vector z_v = forward;

    return {
      .start = pos,
      .dir = (x_v + y_v + z_v).normalized()
    };
  }

  __host__ __device__ inline void get_bounds(size_t *w, size_t *h) const {
    *w = this->w;
    *h = this->h;
  }

  __host__ __device__ constexpr float get_ambient() const { return ambient; }

  __host__ inline void gpu_clean() {}
};

//static_assert(is_camera<cam>);
}

/**
 * @brief Namespace for the default schema types on CPU.
 */
namespace cutrace::cpu::schema {
/**
 * @brief Namespace for the default values for the schemas.
 */
namespace defaults {
/**
 * @brief Compile-time constant for the black/zero vector.
 */
struct black {
  constexpr const static vector value{0.0f, 0.0f, 0.0f}; //!< The black/zero vector.
};

/**
 * @brief Compile-time constant for the up vector (y=1).
 */
struct up {
  constexpr const static vector value{0.0f, 1.0f, 0.0f}; //!< The up vector.
};

/**
 * @brief Compile-time constant for the forward vector (z=1).
 */
struct forward {
  constexpr const static vector value{0.0f, 0.0f, 1.0f}; //!< The forward vector.
};

/**
 * @brief Compile-time constant for the right vector (x=1).
 */
struct right {
  constexpr const static vector value{1.0f, 0.0f, 0.0f}; //!< The right vector.
};

/**
 * @brief Compile-time constant for the white/all-ones vector.
 */
struct white {
  constexpr const static vector value{1.0f, 1.0f, 1.0f}; //!< The white/one vector.
};

#define DEFAULT_FLOAT(name, init) /** @brief Compile-time float value. */struct name { constexpr const static float value = init; }
DEFAULT_FLOAT(zero, 0.0f);
DEFAULT_FLOAT(point_one, 0.1f);
DEFAULT_FLOAT(point_three, 0.3f);
DEFAULT_FLOAT(thirty_two, 32.0f);
DEFAULT_FLOAT(one_hundred, 100.0f);
#undef DEFAULT_FLOAT

struct default_width : std::integral_constant<size_t, 1920> {}; //!< Compile-time constant 1920.
struct default_height : std::integral_constant<size_t, 1080> {}; //!< Compile-time constant 1080.
}

#define ARGUMENT(name, type) loader_argument<name, type, mandatory>
#define OPTIONAL(name, type, def) loader_argument<name, type, def>
#define MK_MANDATORY(x) loader_argument<x::n, x::type, mandatory>

/**
 * @brief Struct representing a triangle.
 */
struct triangle {
  vector p1, //!< The first point
         p2, //!< The second point
         p3; //!< The third point
  size_t mat_idx; //!< The material index

  /**
   * @brief Constructs a new triangle
   * @param p1 The first point
   * @param p2 The second point
   * @param p3 The third point
   * @param mat_idx The material index
   */
  constexpr triangle(vector p1, vector p2, vector p3, size_t mat_idx)
          : p1{p1}, p2{p2}, p3{p3}, mat_idx{mat_idx} {}

  /**
   * @brief Converts this triangle to GPU.
   * @return The GPU representation
   */
  [[nodiscard]] inline gpu::schema::triangle to_gpu() const {
    return { p1.to_gpu(), p2.to_gpu(), p3.to_gpu(), mat_idx };
  }

  constexpr const static char name[] = "triangle"; //!< The type to use in the JSON schema
  constexpr const static char arg_p1[] = "p1"; //!< The argument name for the first point
  constexpr const static char arg_p2[] = "p2"; //!< The argument name for the second point
  constexpr const static char arg_p3[] = "p3"; //!< The argument name for the third point
  constexpr const static char arg_mat_idx[] = "material"; //!< The argument name for the material index

  /**
   * @brief Type alias for the triangle schema.
   */
  using schema = object_schema<name, triangle,
    ARGUMENT(arg_p1, vector),
    ARGUMENT(arg_p2, vector),
    ARGUMENT(arg_p3, vector),
    ARGUMENT(arg_mat_idx, size_t)
  >;
};

/**
 * @brief Struct representing a mesh.
 */
struct mesh {
  std::vector<triangle> tris; //!< The triangles of the mesh
  size_t mat_idx; //!< The material index

  /**
   * @brief Loads a mesh from file.
   * @param file The file to load
   * @param mat_idx The material index
   */
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

  /**
   * @brief Computes the minimum of three values
   * @param f1 The first value
   * @param f2 The second value
   * @param f3 The third value
   * @return The minimum
   */
  __host__ constexpr static float min3(float f1, float f2, float f3) {
    return std::min(std::min(f1, f2), f3);
  }

  /**
   * @brief Computes the maximum of three values
   * @param f1 The first value
   * @param f2 The second value
   * @param f3 The third value
   * @return The maximum
   */
  __host__ constexpr static float max3(float f1, float f2, float f3) {
    return std::max(std::max(f1, f2), f3);
  }

  /**
   * @brief Computes the bounding box of this mesh.
   * @return The AABB
   */
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

  /**
   * @brief Converts this mesh to GPU.
   * @return The GPU representation
   */
  [[nodiscard]] inline gpu::schema::mesh to_gpu() const {
    return { cpu2gpu::vec_to_gpu<triangle, gpu::schema::triangle>(tris), mat_idx, bounding_box() };
  }

  constexpr const static char name[] = "mesh"; //!< The type to use in the JSON schema
  constexpr const static char arg_file[] = "file"; //!< The name of the file argument
  constexpr const static char arg_mat_idx[] = "material"; //!< The name of the material index argument

  /**
   * @brief Type alias for the mesh schema.
   */
  using schema = object_schema<name, mesh,
    ARGUMENT(arg_file, std::string),
    ARGUMENT(arg_mat_idx, size_t)
  >;
};

/**
 * @brief Struct representing a plane.
 */
struct plane {
  vector point, //!< Any point on this plane
         normal; //!< The normal to this plane
  size_t mat_idx; //!< The material index

  /**
   * @brief Constructs a new plane
   * @param point Any point on the plane
   * @param normal The normal to the plane
   * @param mat_idx The material index
   */
  constexpr plane(vector point, vector normal, size_t mat_idx) : point{point}, normal{normal}, mat_idx{mat_idx} {}

  /**
   * @brief Converts this plane to GPU.
   * @return The GPU representation
   */
  [[nodiscard]] inline gpu::schema::plane to_gpu() const {
    return { point.to_gpu(), normal.to_gpu(), mat_idx };
  }

  constexpr const static char name[] = "plane"; //!< The type to use in the JSON schema
  constexpr const static char arg_point[] = "point"; //!< The name of the point argument
  constexpr const static char arg_normal[] = "normal"; //!< The name of the normal argument
  constexpr const static char arg_mat_idx[] = "material"; //!< The name of the material index argument

  /**
   * @brief Type alias for the plane schema.
   */
  using schema = object_schema<name, plane,
    ARGUMENT(arg_point, vector),
    ARGUMENT(arg_normal, vector),
    ARGUMENT(arg_mat_idx, size_t)
  >;
};

/**
 * @brief Struct representing a sphere.
 */
struct sphere {
  vector center; //!< The center of this sphere
  float radius; //!< The radius of this sphere
  size_t mat_idx; //!< The material index

  /**
   * @brief Constructs a new sphere
   * @param center The center of the sphere
   * @param radius The radius of the sphere
   * @param mat_idx The material index
   */
  constexpr sphere(vector center, float radius, size_t mat_idx) : center{center}, radius{radius}, mat_idx{mat_idx} {}

  /**
   * @brief Converts this sphere to GPU.
   * @return The GPU representation
   */
  [[nodiscard]] inline gpu::schema::sphere to_gpu() const {
    return { center.to_gpu(), radius, mat_idx };
  }

  constexpr const static char name[] = "sphere"; //!< The type to use in the JSON schema
  constexpr const static char arg_center[] = "center"; //!< The name of the center argument
  constexpr const static char arg_radius[] = "radius"; //!< The name of the radius argument
  constexpr const static char arg_mat_idx[] = "material"; //!< The name of the material index argument

  /**
   * @brief Type alias for the sphere schema.
   */
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

/**
 * @brief Type alias for the default object schema (triangle, mesh, plane, and sphere).
 */
using default_objects_schema = all_objects_schema<triangle::schema, mesh::schema, plane::schema, sphere::schema>;

/**
 * @brief Struct representing a sun
 */
struct sun {
  vector direction, //!< The direction this directional light is shining in
         color; //!< The color of this light

  /**
   * @brief Constructs a new sun.
   * @param direction The direction
   * @param color The color
   */
  constexpr sun(vector direction, vector color) : direction{direction}, color{color} {}

  /**
   * @brief Converts this sun to GPU.
   * @return The GPU representation
   */
  [[nodiscard]] inline gpu::schema::sun to_gpu() const {
    return { direction.to_gpu(), color.to_gpu() };
  }

  constexpr const static char name[] = "sun"; //!< The type to use in the JSON schema
  constexpr const static char arg_direction[] = "direction"; //!< The name of the direction argument
  constexpr const static char arg_color[] = "color"; //!< The name of the color argument

  /**
   * @brief Type alias for the sun schema.
   */
  using schema = light_schema<name, sun,
    ARGUMENT(arg_direction, vector),
    OPTIONAL(arg_color, vector, defaults::white)
  >;
};

/**
 * @brief Struct representing a point light.
 */
struct point_light {
  vector point, //!< The point from which the light shines
         color; //!< The color of the light

  /**
   * @brief Constructs a new point light.
   * @param point The point to use
   * @param color The color of the light
   */
  constexpr point_light(vector point, vector color) : point{point}, color{color} {}

  /**
   * @brief Converts this point light to GPU.
   * @return The GPU representation
   */
  [[nodiscard]] inline gpu::schema::point_light to_gpu() const {
    return { point.to_gpu(), color.to_gpu() };
  }

  constexpr const static char name[] = "point"; //!< The type to use in the JSON schema
  constexpr const static char arg_point[] = "point"; //!< The name of the point argument
  constexpr const static char arg_color[] = "color"; //!< The name of the color argument

  /**
   * @brief Type alias for the point light schema.
   */
  using schema = light_schema<name, point_light,
    ARGUMENT(arg_point, vector),
    OPTIONAL(arg_color, vector, defaults::white)
  >;
};

static_assert(cpu2gpu::cpu_gpu_light_pair<sun, gpu::schema::sun>);
static_assert(cpu2gpu::cpu_gpu_light_pair<point_light, gpu::schema::point_light>);

/**
 * @brief Type alias for the default light schema (sun, and point light).
 */
using default_lights_schema = all_lights_schema<sun::schema, point_light::schema>;

/**
 * @brief Struct representing a solid-color Phong material.
 */
struct solid_material {
  vector color; //!< The base color of the material
  float specular, //!< How specular the material is
        reflexivity, //!< How mirror-like the material is
        phong_exp, //!< The Phong exponent of the material
        transparency; //!< How translucent/see-through the material is


  /**
   * @brief Constructs a new solid-color material.
   * @param color The base color
   * @param s The specular factor
   * @param r The reflection factor
   * @param p The phong exponent
   * @param t The translucency factor
   */
  constexpr solid_material(vector color, float s, float r, float p, float t) :
    color{color}, specular{s}, reflexivity{r}, phong_exp{p}, transparency{t} {}

  /**
   * @brief Converts this material to GPU.
   * @return The GPU representation
   */
  [[nodiscard]] inline gpu::schema::phong_material to_gpu() const {
    return { color.to_gpu(), specular, reflexivity, phong_exp, transparency };
  }

  constexpr const static char name[] = "solid"; //!< The type to use in the JSON schema
  constexpr const static char arg_color[] = "color"; //!< The name of the color argument
  constexpr const static char arg_spec[] = "specular"; //!< The name of the specular factor argument
  constexpr const static char arg_refl[] = "reflect"; //!< The name of the reflection factor argument
  constexpr const static char arg_phong[] = "phong"; //!< The name of the phong exponent argument
  constexpr const static char arg_trans[] = "transparency"; //!< The name of the translucency factor argument

  /**
   * @brief Type alias for the solid-color material schema.
   */
  using schema = material_schema<name, solid_material,
    ARGUMENT(arg_color, vector),
    OPTIONAL(arg_spec, float, defaults::point_three),
    OPTIONAL(arg_refl, float, defaults::zero),
    OPTIONAL(arg_phong, float, defaults::thirty_two),
    OPTIONAL(arg_trans, float, defaults::zero)
  >;
};

static_assert(cpu2gpu::cpu_gpu_material_pair<solid_material, gpu::schema::phong_material >);

/**
 * @brief Type alias for the default material schema (only solid-color).
 */
using default_material_schema = all_materials_schema<solid_material::schema>;

/**
 * @brief Struct representing the default camera.
 */
struct default_cam {
  vector pos = defaults::black::value; //!< The position of the camera
  vector up = defaults::up::value; //!< The direction the camera considers "up"
  vector look = defaults::forward::value; //!< The point the camera looks at
  float near = defaults::point_one::value; //!< The distance from the camera to the near-plane (unused) @deprecated
  float far = defaults::one_hundred::value; //!< The distance from the camera to the far-plane (unused) @deprecated
  float ambient = defaults::point_one::value; //!< The ambient lighting factor
  size_t w = defaults::default_width::value; //!< The width of the image to render
  size_t h = defaults::default_height::value; //!< The height of the image to render

  /**
   * @brief Constructs a default camera.
   *
   * The default camera is located at (0, 0, 0), with up being y=1, looking at (0, 0, 1).
   * Its near-plane is 0.1 and far-plane is 100, with an ambient factor of 0.1.
   * The default camera renders a 1920x1080 image.
   */
  constexpr default_cam() = default;
  /**
   * @brief Constructs a new camera.
   * @param e The eye position
   * @param u The up vector
   * @param l The look-at point
   * @param n The near-distance
   * @param f The far-distance
   * @param w The width
   * @param h The height
   * @param ambient The ambient factor
   */
  constexpr default_cam(vector e, vector u, vector l, float n, float f, size_t w, size_t h, float ambient)
    : pos{e}, up{u}, look{l}, near{n}, far{f}, w{w}, h{h}, ambient{ambient} {}

  /**
   * @brief Converts this camera to GPU.
   * @return The GPU representation
   */
  [[nodiscard]] inline gpu::schema::cam to_gpu() const {
    gpu::schema::cam res{ pos.to_gpu(), up.to_gpu(), {}, {}, near, far, ambient, w, h };
    res.look_at(look.to_gpu());
    return res;
  }

  constexpr const static char arg_pos[] = "eye"; //!< The name of the eye position argument
  constexpr const static char arg_up[] = "up"; //!< The name of the up vector argument
  constexpr const static char arg_look[] = "look"; //!< The name of the look-at point argument
  constexpr const static char arg_near[] = "near_plane"; //!< The name of the near-plane distance argument
  constexpr const static char arg_far[] = "far_plane"; //!< The name of the far-plane distance argument
  constexpr const static char arg_w[] = "width"; //!< The name of the image width argument
  constexpr const static char arg_h[] = "height"; //!< The name of the image height argument
  constexpr const static char arg_ambient[] = "ambient"; //!< The name of the ambient factor argument

  /**
   * @brief Type alias for the default camera schema.
   */
  using schema = cam_schema<default_cam,
    MK_MANDATORY(OPTIONAL(arg_pos, vector, defaults::black)),
    MK_MANDATORY(OPTIONAL(arg_up, vector, defaults::up)),
    MK_MANDATORY(OPTIONAL(arg_look, vector, defaults::forward)),
    MK_MANDATORY(OPTIONAL(arg_near, float, defaults::point_one)),
    MK_MANDATORY(OPTIONAL(arg_far, float, defaults::one_hundred)),
    MK_MANDATORY(OPTIONAL(arg_w, size_t, defaults::default_width)),
    MK_MANDATORY(OPTIONAL(arg_h, size_t, defaults::default_height)),
    MK_MANDATORY(OPTIONAL(arg_ambient, float, defaults::point_one))
  >;
};

static_assert(cpu2gpu::cpu_gpu_camera_pair<default_cam, gpu::schema::cam>);

/**
 * @brief Type alias for the default schema.
 */
using default_schema = full_schema<default_objects_schema, default_lights_schema, default_material_schema, default_cam::schema>;

/**
 * @brief Convenience function to use the default schema to load a file.
 * @param file The file to load
 * @return The scene in the file
 */
inline default_schema::scene_t load_default(const std::string &file) {
  return default_schema::load_file(file);
}

using default_cpu_object = cpu_object_set<triangle, mesh, plane, sphere>; //!< Type alias for the default object type on CPU.
using default_cpu_light = cpu_light_set<sun, point_light>; //!< Type alias for the default light type on CPU.
using default_cpu_material = cpu_material_set<solid_material>; //!< Type alias for the default material type on CPU.
using default_cpu_cam = default_cam; //!< Type alias for the default camera type on CPU.
using default_gpu_object = gpu::gpu_object_set<gpu::schema::triangle, gpu::schema::mesh, gpu::schema::plane, gpu::schema::sphere>; //!< Type alias for the default object type on GPU.
using default_gpu_light = gpu::gpu_light_set<gpu::schema::sun, gpu::schema::point_light>; //!< Type alias for the default light type on GPU.
using default_gpu_material = gpu::gpu_material_set<gpu::schema::phong_material>; //!< Type alias for the default material type on GPU.
using deafult_gpu_cam = gpu::schema::cam; //!< Type alias for the default camera type on GPU.

using default_cpu_scene = cpu_scene<default_cpu_object, default_cpu_light, default_cpu_material, default_cpu_cam>; //!< Type alias for the default CPU scene type.
using default_gpu_scene = gpu::gpu_scene_<default_gpu_object, default_gpu_light, default_gpu_material, deafult_gpu_cam>; //!< Type alias for the default GPU scene type.

using default_converter = cpu2gpu::cpu_to_gpu<default_cpu_scene, default_gpu_scene>; //!< Type alias for the converter between default CPU and GPU scenes.

/**
 * @brief Converts a default CPU scene to GPU.
 * @param cpu The default CPU scene
 * @return The GPU scene
 */
inline default_gpu_scene default_to_gpu(const default_cpu_scene &cpu) {
  return default_converter::convert(cpu);
}

using default_schema_viewer = viewer_for_t<default_schema>; //!< Type alias for a scene viewer for the default schema
}

#endif //CUTRACE_DEFAULT_SCHEMA_HPP
