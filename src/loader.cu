//
// Created by jay on 11/18/23.
//

#include <fstream>
#include <sstream>
#include "loader.hpp"
#include "picojson.h"
#include <sstream>

using namespace cutrace;
using namespace cutrace::cpu;

using array = picojson::value::array;
using object = picojson::value::object;

template <typename T, typename Fun>
inline void with(const picojson::value &val, Fun &&f) {
  if(val.is<T>()) {
    f(val.get<T>());
  }
  else {
    std::cerr << "Expected a " << typeid(T).name() << ", got " << val.serialize() << "\n";
  }
}

template <typename T, typename Fun> requires(std::invocable<Fun, const T &>)
inline void withKey(const picojson::value &val, const char *key, Fun &&f) {
  if(val.is<object>()) {
    auto it = val.get<object>().find(key);
    if(it != val.get<object>().end() && it->second.is<T>()) {
      f(it->second.get<T>());
    } else if(it == val.get<object>().end()) {
      std::cerr << "Key '" << key << "' does not exist.\n";
    } else {
      std::cerr << "Expected a " << typeid(T).name() << ", got " << it->second.serialize() << "\n";
    }
  } else {
    std::cerr << "Expected an object, got " << val.serialize() << "\n";
  }
}

template <typename T, typename Fun> requires(std::invocable<Fun, const T &, const object &>)
inline void withKey(const picojson::value &val, const char *key, Fun &&f) {
  if(val.is<object>()) {
    auto it = val.get<object>().find(key);
    if(it != val.get<object>().end() && it->second.is<T>()) {
      f(it->second.get<T>(), val.get<object>());
    } else if(it == val.get<object>().end()) {
      std::cerr << "Key '" << key << "' does not exist.\n";
    } else {
      std::cerr << "Expected a " << typeid(T).name() << ", got " << it->second.serialize() << "\n";
    }
  } else {
    std::cerr << "Expected an object, got " << val.serialize() << "\n";
  }
}

template <typename T, typename Fun> requires(std::invocable<Fun, const T &>)
inline void withKey(const object &o, const char *key, Fun &&f) {
  auto it = o.find(key);
  if(it != o.end() && it->second.is<T>()) {
    f(it->second.get<T>());
  } else if(it == o.end()) {
    std::cerr << "Key '" << key << "' does not exist.\n";
  } else {
    std::cerr << "Expected a " << typeid(T).name() << ", got " << it->second.serialize() << "\n";
  }
}

inline cpu::vector parse_vec(const picojson::value &o) {
  cpu::vector res{0,0,0};
  with<array>(o, [&res](const array &a) {
    if(a.size() >= 3) {
      with<double>(a[0], [&res](const double &d){ res.x = (float)d; });
      with<double>(a[1], [&res](const double &d){ res.y = (float)d; });
      with<double>(a[2], [&res](const double &d){ res.z = (float)d; });
    }
  });
  return res;
}

inline cpu::vector parse_vec(const array &a) {
  cpu::vector res{0,0,0};
  with<double>(a[0], [&res](const double &d){ res.x = (float)d; });
  with<double>(a[1], [&res](const double &d){ res.y = (float)d; });
  with<double>(a[2], [&res](const double &d){ res.z = (float)d; });
  return res;
}

inline cpu_object parse_obj(const std::string &file, size_t mat_idx) {
  std::vector<cpu::vector> vertices;
  std::vector<std::vector<size_t>> faces;

  std::string line;
  std::ifstream strm(file);
  while(std::getline(strm, line)) {
//    std::getline(strm, line);
//    if(line.empty()) break;

    std::string part;
    std::stringstream line_strm(line);
    line_strm >> part;
    if(part == "v") {
      float x, y, z;
      line_strm >> x >> y >> z;
      vertices.push_back(vector{ x, y, z });
    }
    else if(part == "f") {
      std::vector<size_t> verts;
      size_t idx;
      std::string garbage;
      while(!line_strm.eof()) {
        line_strm >> idx;
        verts.push_back(idx);
        if(line_strm.peek() == '/') line_strm >> garbage;
      }
      faces.push_back(std::move(verts));
    }
  }

  cpu::triangle_set set{
    .tris = {},
    .mat_idx = mat_idx
  };
  for(const auto &f: faces) {
    size_t v1 = f[0], v2 = f[1], v3 = f[2];
    set.tris.push_back(triangle{ vertices[v1], vertices[v2], vertices[v3] });
    for(size_t i = 3; i < f.size(); i++)
      set.tris.push_back(triangle{
        .p1 = vertices[v1], .p2 = vertices[f[i - 1]], .p3 = vertices[f[i]]
      });
  }

  return set;
}

cpu::cpu_scene loader::load(const std::string &file) {
  picojson::value res;
  std::ifstream strm(file);
  picojson::parse(res, strm);

  std::string err = picojson::get_last_error();
  if (!err.empty()) {
    std::cerr << err << std::endl;
    return {};
  }

  std::vector<cpu_object> objects;
  std::vector<cpu_light> lights;
  std::vector<cpu_mat> materials;

  withKey<array>(res, "objects", [&objects](const array &a) {
    for (const auto &obj: a) {
      withKey<std::string>(obj, "type", [&objects](const std::string &type, const object &o) {
        withKey<double>(o, "material", [&objects, &o, &type](const int &mat_idx) {
          if (type == "triangle") {
            withKey<array>(o, "points", [&objects, &mat_idx](const array &a) {
              if (a.size() == 3) {
                objects.emplace_back(cpu::triangle{
                        .p1 = parse_vec(a[0]),
                        .p2 = parse_vec(a[1]),
                        .p3 = parse_vec(a[2]),
                        .mat_idx = (size_t) mat_idx,
                });
              }
            });
          }
          else if (type == "model") {
            withKey<std::string>(o, "file", [&objects, &mat_idx](const std::string &fname) {
              objects.push_back(std::move(parse_obj(fname, (size_t) mat_idx)));
            });
          }
          else if (type == "plane") {
            withKey<array>(o, "point", [&objects, &o, &mat_idx](const array &point) {
              withKey<array>(o, "normal", [&objects, &mat_idx, &point](const array &normal) {
                objects.emplace_back(cpu::plane {
                  .point = parse_vec(point),
                  .normal = parse_vec(normal),
                  .mat_idx = (size_t) mat_idx
                });
              });
            });
          }
          else if (type == "sphere") {
            withKey<double>(o, "radius", [&objects, &mat_idx, &o](const double &radius) {
              withKey<array>(o, "center", [&objects, &mat_idx, &radius](const array &pt) {
                objects.emplace_back(cpu::sphere {
                  .center = parse_vec(pt),
                  .radius = (float)radius,
                  .mat_idx = (size_t)mat_idx
                });
              });
            });
          }
          else {
            std::cerr << "Invalid object type '" << type << "'.\n";
          }
        });
      });
    }
  });

  withKey<array>(res, "lights", [&lights](const array &a) {
    for (const auto &obj: a) {
      withKey<std::string>(obj, "type", [&lights](const std::string &type, const object &o) {
        if (type == "sun") {
          withKey<array>(o, "direction", [&lights, &o](const array &a) {
            withKey<array>(o, "color", [&lights, &a](const array &a2) {
              lights.emplace_back(sun{
                      .direction = parse_vec(a),
                      .color = parse_vec(a2)
              });
            });
          });
        }
        else if (type == "point") {
          withKey<array>(o, "position", [&lights, &o](const array &pos) {
            withKey<array>(o, "color", [&lights, &pos](const array &color) {
              lights.emplace_back(point_light{
                .point = parse_vec(pos),
                .color = parse_vec(color)
              });
            });
          });
        }
        else {
          std::cerr << "Invalid light type '" << type << "'.\n";
        }
      });
    }
  });

  withKey<array>(res, "materials", [&materials](const array &a) {
    for (const auto &mat: a) {
      withKey<array>(mat, "color", [&materials](const array &color, const object &o) {
        withKey<double>(o, "specular", [&materials, &o, &color](const double &specular) {
          withKey<double>(o, "phong", [&materials, &o, &color, &specular](const double &phong) {
            withKey<double>(o, "reflect", [&materials, &color, &specular, &phong](const double &reflect) {
              materials.push_back({
                  .color = parse_vec(color),
                  .specular = (float) specular,
                  .reflexivity = (float) reflect,
                  .phong_exp = (float) phong
              });
            });
          });
        });
      });
    }
  });

  return {.objects = objects, .lights = lights, .materials = materials};
}