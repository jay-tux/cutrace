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
struct white {
  constexpr const static vector value{1.0f, 1.0f, 1.0f};
};

struct zero { constexpr const static float value = 0.0f; };
struct point_three { constexpr const static float value = 0.3f; };
struct thirty_two { constexpr const static float value = 32.0f; };
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

using default_schema = full_schema<default_objects_schema, default_lights_schema, default_material_schema>;
}

#endif //CUTRACE_DEFAULT_SCHEMA_HPP
