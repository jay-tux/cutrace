//
// Created by jay on 11/18/23.
//

#include <fstream>
#include <sstream>
#include "loader.hpp"
#include "picojson.h"
#include "mesh_loader.hpp"
#include <sstream>

using namespace cutrace;
using namespace cutrace::cpu;

using array = picojson::value::array;
using object = picojson::value::object;

#define STRINGIZE(x) STRINGIZE2(x)
#define STRINGIZE2(x) #x
#define LINE_STR STRINGIZE(__LINE__)
#define LINE_END " (at " __FILE__ ":" LINE_STR ").\n"

template <typename T, typename Fun>
inline void with(const picojson::value &val, Fun &&f) {
  if(val.is<T>()) {
    f(val.get<T>());
  }
  else {
    std::cerr << "Expected a " << typeid(T).name() << ", got " << val.serialize() << LINE_END;
  }
}

template <typename T, typename Fun> requires(std::invocable<Fun, const T &>)
inline void withKey(const picojson::value &val, const char *key, Fun &&f, bool warn_missing = true) {
  if(val.is<object>()) {
    auto it = val.get<object>().find(key);
    if (it != val.get<object>().end() && it->second.is<T>()) {
      f(it->second.get<T>());
    }
    else if (warn_missing) {
      if (it == val.get<object>().end()) {
        std::cerr << "Key '" << key << "' does not exist" LINE_END;
      }
      else {
        std::cerr << "Expected a " << typeid(T).name() << ", got " << it->second.serialize() << LINE_END;
      }
    }
  } else {
    std::cerr << "Expected an object, got " << val.serialize() << LINE_END;
  }
}

template <typename T, typename Fun> requires(std::invocable<Fun, const T &, const object &>)
inline void withKey(const picojson::value &val, const char *key, Fun &&f, bool warn_missing = true) {
  if(val.is<object>()) {
    auto it = val.get<object>().find(key);
    if(it != val.get<object>().end() && it->second.is<T>()) {
      f(it->second.get<T>(), val.get<object>());
    }
    else if(warn_missing) {
      if (it == val.get<object>().end()) {
        std::cerr << "Key '" << key << "' does not exist" LINE_END;
      } else {
        std::cerr << "Expected a " << typeid(T).name() << ", got " << it->second.serialize() << LINE_END;
      }
    }
  } else {
    std::cerr << "Expected an object, got " << val.serialize() << LINE_END;
  }
}

template <typename T, typename Fun> requires(std::invocable<Fun, const T &>)
inline void withKey(const object &o, const char *key, Fun &&f, bool warn_missing = true) {
  auto it = o.find(key);
  if(it != o.end() && it->second.is<T>()) {
    f(it->second.get<T>());
  }
  else if(warn_missing) {
    if (it == o.end()) {
      std::cerr << "Key '" << key << "' does not exist" LINE_END;
    } else {
      std::cerr << "Expected a " << typeid(T).name() << ", got " << it->second.serialize() << LINE_END;
    }
  }
}

inline vector parse_vec(const picojson::value &o) {
  vector res{0,0,0};
  with<array>(o, [&res](const array &a) {
    if(a.size() >= 3) {
      with<double>(a[0], [&res](const double &d){ res.x = (float)d; });
      with<double>(a[1], [&res](const double &d){ res.y = (float)d; });
      with<double>(a[2], [&res](const double &d){ res.z = (float)d; });
    }
  });
  return res;
}

inline vector parse_vec(const array &a) {
  vector res{0,0,0};
  with<double>(a[0], [&res](const double &d){ res.x = (float)d; });
  with<double>(a[1], [&res](const double &d){ res.y = (float)d; });
  with<double>(a[2], [&res](const double &d){ res.z = (float)d; });
  return res;
}

cpu::cpu_scene loader::load(const std::string &file) {
  picojson::value res;
  std::ifstream strm(file);
  picojson::parse(res, strm);

  const std::string& err = picojson::get_last_error();
  if (!err.empty()) {
    std::cerr << err << std::endl;
    return {};
  }

  gpu::cam cam{};
  std::vector<cpu_object> objects;
  std::vector<cpu_light> lights;
  std::vector<cpu_mat> materials;

  withKey<object>(res, "camera", [&cam](const object &camera) {
    withKey<double>(camera, "near_plane", [&cam](const double &near) {
      cam.near = (float)near;
    }, false);
    withKey<double>(camera, "far_plane", [&cam](const double &far) {
      cam.far = (float)far;
    }, false);
    withKey<array>(camera, "eye", [&cam](const array &eye) {
      cam.pos = parse_vec(eye).to_gpu();
    }, false);
    withKey<array>(camera, "up", [&cam](const array &up) {
      cam.up = parse_vec(up).to_gpu();
    }, false);
    withKey<array>(camera, "look", [&cam](const array &look) {
      cam.look_at(parse_vec(look).to_gpu());
    }, false);
    withKey<double>(camera, "width", [&cam](const double &w) {
      cam.w = (size_t)w;
    }, false);
    withKey<double>(camera, "height", [&cam](const double &h) {
      cam.h = (size_t)h;
    }, false);
  });

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
              auto loaded = load_meshes(fname, mat_idx);
              objects.insert(objects.end(), loaded.begin(), loaded.end());
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
            auto l = sun { .direction = parse_vec(a) };
            withKey<array>(o, "color", [&l](const array &a2) {
              l.color = parse_vec(a2);
            }, false);
            lights.push_back(l);
          });
        }
        else if (type == "point") {
          withKey<array>(o, "position", [&lights, &o](const array &pos) {
            auto l = point_light { .point = parse_vec(pos) };
            withKey<array>(o, "color", [&l](const array &color) {
              l.color = parse_vec(color);
            });
            lights.push_back(l);
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
        cpu_mat m {
          .color = parse_vec(color)
        };

        withKey<double>(o, "specular", [&m](const double &specular) {
          m.specular = specular;
        }, false);
        withKey<double>(o, "phong", [&m](const double &phong) {
          m.phong_exp = phong;
        }, false);
        withKey<double>(o, "reflect", [&m](const double &reflect) {
          m.reflexivity = reflect;
        }, false);
        withKey<double>(o, "transparency", [&m](const double &trans) {
          m.transparency = trans;
        }, false);

        materials.push_back(m);
      });
    }
  });

  return {.camera = cam, .objects = objects, .lights = lights, .materials = materials};
}