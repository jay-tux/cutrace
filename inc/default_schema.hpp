//
// Created by jay on 11/29/23.
//

#ifndef CUTRACE_DEFAULT_SCHEMA_HPP
#define CUTRACE_DEFAULT_SCHEMA_HPP

#include <vector>
#include "vector.hpp"
#include "loader.hpp"

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

  constexpr triangle(cpu::vector p1, cpu::vector p2, cpu::vector p3, size_t mat_idx)
          : p1{p1}, p2{p2}, p3{p3}, mat_idx{mat_idx} {}

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

  constexpr mesh(std::string file, size_t mat_idx) : tris{}, mat_idx{mat_idx} {
    // TODO
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

using default_objects_schema = all_objects_schema<triangle::schema, mesh::schema, plane::schema, sphere::schema>;

struct sun {
  vector direction, color;

  constexpr sun(vector direction, vector color) : direction{direction}, color{color} {}

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

  constexpr const static char name[] = "point";
  constexpr const static char arg_point[] = "point";
  constexpr const static char arg_color[] = "color";

  using schema = light_schema<name, point_light,
    ARGUMENT(arg_point, vector),
    OPTIONAL(arg_color, vector, defaults::white)
  >;
};

using default_lights_schema = all_lights_schema<sun::schema, point_light::schema>;

struct solid_material {
  vector color;
  float specular, reflexivity, phong_exp, transparency;

  constexpr solid_material(vector color, float s, float r, float p, float t) :
    color{color}, specular{s}, reflexivity{r}, phong_exp{p}, transparency{t} {}

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

using default_material_schema = all_materials_schema<solid_material::schema>;

struct default_cam {
  vector pos = defaults::black::value;
  vector up = defaults::up::value;
  vector look = defaults::forward::value;
  float near = defaults::point_one::value;
  float far = defaults::one_hundred::value;
  size_t w = defaults::default_width::value;
  size_t h = defaults::default_height::value;

  constexpr default_cam() = default;
  constexpr default_cam(vector e, vector u, vector l, float n, float f, size_t w, size_t h)
    : pos{e}, up{u}, look{l}, near{n}, far{f}, w{w}, h{h} {}

  constexpr const static char arg_pos[] = "eye";
  constexpr const static char arg_up[] = "up";
  constexpr const static char arg_look[] = "look";
  constexpr const static char arg_near[] = "near_plane";
  constexpr const static char arg_far[] = "far_plane";
  constexpr const static char arg_w[] = "width";
  constexpr const static char arg_h[] = "height";

  using schema = cam_schema<default_cam,
    OPTIONAL(arg_pos, vector, defaults::black),
    OPTIONAL(arg_up, vector, defaults::up),
    OPTIONAL(arg_look, vector, defaults::forward),
    OPTIONAL(arg_near, float, defaults::point_one),
    OPTIONAL(arg_far, float, defaults::one_hundred),
    OPTIONAL(arg_w, size_t, defaults::default_width),
    OPTIONAL(arg_h, size_t, defaults::default_height)
  >;
};

using default_schema = full_schema<default_objects_schema, default_lights_schema, default_material_schema, default_cam::schema>;
}

#endif //CUTRACE_DEFAULT_SCHEMA_HPP
