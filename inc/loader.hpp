//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_LOADER_HPP
#define CUTRACE_LOADER_HPP

#include <vector>
#include <string>
#include <fstream>
#include "cpu_types.hpp"
#include "cpu_types_.hpp"
#include "picojson.h"
#include "json_helpers.hpp"
#include "either.hpp"

/**
 * @brief Namespace containing all of cutrace's code.
 */
namespace cutrace {
//region loader arguments

template <typename T>
using or_error = either<json_error, T>;

template <typename F, typename T>
concept compile_time_T = requires() {
  { F::value } -> std::same_as<const T &>;
};

struct mandatory {};

template <const char *name, typename T, typename D = mandatory> struct loader_argument;

template <const char *name, typename T>
struct loader_argument<name, T, mandatory> {
  using type = T;
  using json_t = json_type<T>::type;
  constexpr const static bool is_required = true;

  static inline or_error<T> load_from(const picojson::object &o) {
    return coerce_key<T>(o, name);
  }
};

template <const char *name, typename T, typename D> requires(compile_time_T<D, T>)
struct loader_argument<name, T, D> {
  using type = T;
  using json_t = json_type<T>::type;
  constexpr const static bool is_required = false;
  constexpr const static T default_v = D::value;

  static inline or_error<T> load_from(const picojson::object &o) {
    return coerce_key<T>(o, name, default_v);
  }
};

template <const char *name>
struct loader_argument<name, cpu::vector, mandatory> {
  using type = cpu::vector;
  using json_t = picojson::array;
  constexpr const static bool is_required = false;

  static inline or_error<cpu::vector> load_from(const picojson::object &o) {
    return coerce_key<json_t>(o, name).fmap([](const json_t &arr) -> or_error<cpu::vector> {
      if(arr.size() != 3) return {json_error{"Expected a 3-value array, got " + std::to_string(arr.size()) + " instead."} };

      return coerce<float>(arr[0]).fmap([&arr](const float &x) {
        return coerce<float>(arr[1]).fmap([&arr, x](const float &y) {
          return coerce<float>(arr[1]).fmap([x, y](const float &z) -> or_error<cpu::vector> {
            return { cpu::vector{ x, y, z } };
          });
        });
      });
    });
  }
};

template <const char *name, typename D> requires(compile_time_T<D, cpu::vector>)
struct loader_argument<name, cpu::vector, D> {
  using type = cpu::vector;
  using json_t = picojson::array;
  constexpr const static bool is_required = false;
  constexpr const static cpu::vector default_v = D::value;

  static inline or_error<cpu::vector> load_from(const picojson::object &o) {
    auto d = picojson::array{
      {picojson::value(default_v.x), picojson::value(default_v.y), picojson::value(default_v.z)}
    };

    return coerce_key(o, name, d).fmap([](const json_t &arr) -> or_error<cpu::vector> {
      if(arr.size() != 3) return {json_error{"Expected a 3-value array, got " + std::to_string(arr.size()) + " instead."} };

      return coerce<float>(arr[0]).fmap([&arr](const float &x) {
        return coerce<float>(arr[1]).fmap([&arr, x](const float &y) {
          return coerce<float>(arr[1]).fmap([x, y](const float &z) -> or_error<cpu::vector> {
            return { cpu::vector{ x, y, z } };
          });
        });
      });
    });
  }
};

template <typename T, typename ... Ts> struct constructible_from_arguments : std::bool_constant<false> {};

template <typename T, const char *... names, typename ... As, typename ... Ds>
struct constructible_from_arguments<T, loader_argument<names, As, Ds>...> : std::bool_constant<std::constructible_from<T, As...>> {};

template <typename T, typename ... Ts>
concept constructible_from_args = constructible_from_arguments<T, Ts...>::value;

//endregion

//region object schema(s)

template <const char *name, typename O, typename ... Ts> struct object_schema;

template <const char *name, typename O, const char *... names, typename ... Ts, typename ... Ds>
  requires constructible_from_args<O, loader_argument<names, Ts, Ds>...>
struct object_schema<name, O, loader_argument<names, Ts, Ds>...> {
  using object_t = O;
  constexpr const static char *type = name;

  static inline or_error<O> load_from(const picojson::object &o) {
    return fmap_all([](const auto &... args) -> O {
      return O(args...);
    }, loader_argument<names, Ts, Ds>::load_from(o)...);
  }
};

template <typename T> struct is_object_schema : std::bool_constant<false> {};
template <const char *name, typename T, typename ... Ts> struct is_object_schema<object_schema<name, T, Ts...>> : std::bool_constant<true> {};

template <typename ... Ts> struct extract_object_variant;
template <typename ... Ts> requires(is_object_schema<Ts>::value && ...)
struct extract_object_variant<Ts...> {
  using type = cpu::cpu_object_set<typename Ts::object_t...>;
};

template <typename Out, typename ... Ts> struct find_matching_object;
template <typename Out, typename T> requires(is_object_schema<T>::value)
struct find_matching_object<Out, T> {
  static inline or_error<Out> find(const std::string &key, const picojson::object &o) {
    if(T::type == key) return T::load_from(o).template re_wrap<Out>();
    return or_error<Out>::left(json_error{"Type '" + key + "' is invalid."});
  }
};

template <typename Out, typename T1, typename T2, typename ... TRest>
  requires(is_object_schema<T1>::value && is_object_schema<T2>::value && (is_object_schema<TRest>::value && ...))
struct find_matching_object<Out, T1, T2, TRest...> {
  static inline or_error<Out> find(const std::string &key, const picojson::object &o) {
    if(T1::type == key) return T1::load_from(o).template re_wrap<Out>();
    return find_matching_object<Out, T2, TRest...>::find(key, o);
  }
};

template <typename ... Ts> requires(is_object_schema<Ts>::value && ...)
struct all_objects_schema {
  using any = extract_object_variant<Ts...>::type;

  static inline or_error<any> load_from(const picojson::object &o) {
    return coerce_key<std::string>(o, "type").fmap([&o](const std::string &type) {
      return find_matching_object<any, Ts...>::find(type, o);
    });
  }
};

//endregion

//region light schema(s)

template <const char *name, typename L, typename ... Ts> struct light_schema;

template <const char *name, typename L, const char *... names, typename ... Ts, typename ... Ds>
  requires constructible_from_args<L, loader_argument<names, Ts, Ds>...>
struct light_schema<name, L, loader_argument<names, Ts, Ds>...> {
  using light_t = L;
  constexpr const static char *type = name;

  static or_error<L> load_from(const picojson::object &o) {
    return fmap_all([](const auto &... args) -> L {
      return L(args...);
    }, loader_argument<names, Ts, Ds>::load_from(o)...);
  }
};

template <typename T> struct is_light_schema : std::bool_constant<false> {};
template <const char *name, typename T, typename ... Ts> struct is_light_schema<light_schema<name, T, Ts...>> : std::bool_constant<true> {};

template <typename ... Ts> struct extract_light_variant;
template <typename ... Ts> requires(is_light_schema<Ts>::value && ...)
struct extract_light_variant<Ts...> {
  using type = cpu::cpu_light_set<typename Ts::light_t...>;
};

template <typename Out, typename ... Ts> struct find_matching_light;
template <typename Out, typename T> requires(is_light_schema<T>::value)
struct find_matching_light<Out, T> {
  static inline or_error<Out> find(const std::string &key, const picojson::object &o) {
    if(T::type == key) T::load_from(o).template re_wrap<Out>();
    return or_error<Out>::left(json_error{ "Type '" + key + "' is invalid." });
  }
};

template <typename Out, typename T1, typename T2, typename ... TRest>
requires(is_light_schema<T1>::value && is_light_schema<T2>::value && (is_light_schema<TRest>::value && ...))
struct find_matching_light<Out, T1, T2, TRest...> {
  static inline or_error<Out> find(const std::string &key, const picojson::object &o) {
    if(T1::type == key) return T1::load_from(o).template re_wrap<Out>();
    return find_matching_light<Out, T2, TRest...>::find(key, o);
  }
};

template <typename ... Ts> requires(is_light_schema<Ts>::value && ...)
struct all_lights_schema {
  using any = extract_light_variant<Ts...>::type;

  static inline or_error<any> load_from(const picojson::object &o) {
    return coerce_key<std::string>(o, "type").fmap([&o](const std::string &type) {
      return find_matching_light<any, Ts...>::find(type, o);
    });
  }
};

//endregion

//region material schema(s)

template <const char *name, typename M, typename ... Ts> struct material_schema;

template <const char *name, typename M, const char *... names, typename ... Ts, typename ... Ds>
struct material_schema<name, M, loader_argument<names, Ts, Ds>...> {
  using material_t = M;
  constexpr const static char *type = name;

  static or_error<M> load_from(const picojson::object &o) {
    return fmap_all([](const auto &... args) -> M {
      return M(args...);
    }, loader_argument<names, Ts, Ds>::load_from(o)...);
  }
};

template <typename T> struct is_material_schema : std::bool_constant<false> {};
template <const char *name, typename M, typename ... Ts> struct is_material_schema<material_schema<name, M, Ts...>> : std::bool_constant<true> {};

template <typename ... Ts> struct extract_material_variant;
template <typename ... Ts> requires(is_material_schema<Ts>::value && ...)
struct extract_material_variant<Ts...> {
  using type = cpu::cpu_material_set<typename Ts::material_t...>;
};

template <typename Out, typename ... Ts> struct find_matching_mat;
template <typename Out, typename T> requires(is_material_schema<T>::value)
struct find_matching_mat<Out, T> {
  static inline or_error<Out> find(const std::string &key, const picojson::object &o) {
    if(T::type == key) return T::load_from(o).template re_wrap<Out>();
    return or_error<Out>::left(json_error{ "Type '" + key + "' is invalid." });
  }
};

template <typename Out, typename T1, typename T2, typename ... TRest>
requires(is_light_schema<T1>::value && is_light_schema<T2>::value && (is_material_schema<TRest>::value && ...))
struct find_matching_mat<Out, T1, T2, TRest...> {
  static inline or_error<Out> find(const std::string &key, const picojson::object &o) {
    if(T1::type == key) return T1::load_from(o).template re_wrap<Out>();
    return find_matching_mat<Out, T2, TRest...>::find(key, o);
  }
};

template <typename ... Ts> requires(is_material_schema<Ts>::value && ...)
struct all_materials_schema {
  using any = extract_material_variant<Ts...>::type;

  static inline or_error<any> load_from(const picojson::object &o) {
    return coerce_key<std::string>(o, "type").fmap([&o](const std::string &type) {
      return find_matching_mat<any, Ts...>::find(type, o);
    });
  }
};

//endregion

//region camera schema
template <typename C, typename ... Ts> struct cam_schema;

template <typename C, const char *... names, typename ... Ts, typename ... Ds>
requires constructible_from_args<C, loader_argument<names, Ts, Ds>...>
struct cam_schema<C, loader_argument<names, Ts, Ds>...> {
  using cam_t = C;

  static inline or_error<C> load_from(const picojson::object &o) {
    return fmap_all([](const auto &... args) -> C {
      return C(args...);
    }, loader_argument<names, Ts, Ds>::load_from(o)...);
  }
};
//endregion

template <typename O, typename L, typename M, typename C> struct full_schema;

template <typename ... Os, typename ... Ls, typename ... Ms, typename C, const char *...c_names, typename ... CTypes, typename ... CDs>
struct full_schema<all_objects_schema<Os...>, all_lights_schema<Ls...>, all_materials_schema<Ms...>, cam_schema<C, loader_argument<c_names, CTypes, CDs>...>> {
  using object_schema = all_objects_schema<Os...>;
  using light_schema = all_lights_schema<Ls...>;
  using material_schema = all_materials_schema<Ms...>;
  using camera_schema = cam_schema<C, loader_argument<c_names, CTypes, CDs>...>;
  using object_t = object_schema::any;
  using light_t = light_schema::any;
  using material_t = material_schema::any;
  using cam_t = C;
  using scene_t = cpu::cpu_scene_<object_t, light_t, material_t, cam_t>;

  static inline scene_t load_from(const picojson::object &o) {
    std::vector<object_t> objects{};
    std::vector<light_t> lights{};
    std::vector<material_t> materials{};
    cam_t camera{};

    coerce_key<picojson::array>(o, "objects").map([&objects](const picojson::array &objs) {
      objects.reserve(objs.size());
      for(size_t i = 0; i < objs.size(); i++) {
        force_object(objs[i]).map([i, &objects](const auto *v) {
          object_schema::load_from(*v).map([&objects](const auto &obj) {
            objects.push_back(obj);
          });
        }).map_left([i](const json_error &err) {
          std::cerr << "Error while loading object #" << i << ": " << err.message << "\n";
        });
      }
    }).map_left([](const json_error &err) {
      std::cerr << "Could not find 'objects' array: " << err.message << ".\n";
    });

    coerce_key<picojson::array>(o, "lights").map([&lights](const picojson::array &objs) {
      lights.reserve(objs.size());
      for(size_t i = 0; i < objs.size(); i++) {
        force_object(objs[i]).map([i, &lights](const auto *v) {
          light_schema::load_from(*v).map([&lights](const auto &light) {
            lights.push_back(light);
          });
        }).map_left([i](const json_error &err) {
          std::cerr << "Error while loading light #" << i << ": " << err.message << "\n";
        });
      }
    }).map_left([](const json_error &err) {
      std::cerr << "Could not find 'lights' array: " << err.message << ".\n";
    });

    coerce_key<picojson::array>(o, "materials").map([&materials](const picojson::array &objs) {
      materials.reserve(objs.size());
      for(size_t i = 0; i < objs.size(); i++) {
        force_object(objs[i]).map([i, &materials](const auto *v) {
          material_schema::load_from(*v).map([&materials](const auto &material) {
            materials.push_back(material);
          });
        }).map_left([i](const json_error &err) {
          std::cerr << "Error while loading material #" << i << ": " << err.message << "\n";
        });
      }
    }).map_left([](const json_error &err) {
      std::cerr << "Could not find 'materials' array: " << err.message << ".\n";
    });

    coerce_key<picojson::object>(o, "camera").map([&camera](const picojson::object *obj) {
      camera_schema::load_from(*obj).map([&camera](const auto &c) {
        camera = c;
      });
    }).map_left([](const json_error &err) {
      std::cerr << "Could not find 'camera' object: " << err.message << ".\n";
    });

    return scene_t {
      .objects = objects,
      .lights = lights,
      .materials = materials
    };
  }

  static inline scene_t load_file(const std::string &file) {
    picojson::value res;
    std::ifstream strm(file);
    picojson::parse(res, strm);

    if(!picojson::get_last_error().empty()) {
      std::cerr << "Error while loading file '" << file << "': " << picojson::get_last_error() << "\n";
      return {};
    }

    return force_object(res).fold(
            [&file](const auto &err) -> scene_t {
              std::cerr << "Error while loading file '" << file << "': " << err.message << "\n";
              return scene_t{};
            },
            [](const auto *data) -> scene_t { return load_from(*data); }
    );
  }
};

/**
 * @brief Struct containing the static load method to load a scene from a file.
 */
struct loader {
  /**
   * @brief Loads a scene from a file.
   * @param [in] file The file to load from
   * @return The parsed CPU scene
   *
   * The file should be a valid JSON file, conforming to the schema. Only minimal checking and error recovery is
   * implemented.
   */
  static cpu::cpu_scene load(const std::string &file);
};
}

#endif //CUTRACE_LOADER_HPP
