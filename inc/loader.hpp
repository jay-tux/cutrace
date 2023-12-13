//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_LOADER_HPP
#define CUTRACE_LOADER_HPP

#include <vector>
#include <string>
#include <fstream>
#include "cpu_types.hpp"
#include "picojson.h"
#include "json_helpers.hpp"
#include "either.hpp"

/**
 * @brief Namespace containing all of cutrace's code.
 */
namespace cutrace {
/**
 * Helper operator to print a 3D vector.
 * @param strm The stream to print to
 * @param v The vector to print
 * @return A reference to the stream
 */
__host__ inline std::ostream &operator<<(std::ostream &strm, const vector &v) {
  return strm << "vector{ .x = " << v.x << ", .y = " << v.y << ", .z = " << v.z << " }";
}

//region loader arguments

/**
 * @brief Type alias for `cutrace::either<cutrace::json_error, T>`.
 */
template <typename T>
using or_error = either<json_error, T>;

/**
 * @brief Concept relating what it means to be a type that holds a compile-time value.
 * @tparam F The type to check
 * @tparam T The type of the compile-time value
 *
 * To satisfy this constraint, `F::value` must be a `const T &`.
 */
template <typename F, typename T>
concept compile_time_T = requires() {
  { F::value } -> std::same_as<const T &>;
};

/**
 * @brief Type tag to mark a loader argument as mandatory.
 */
struct mandatory {};

/**
 * @brief Struct to represent a loader argument in the schema.
 * @tparam name The name of the argument
 * @tparam T The type of the argument
 * @tparam D Either `cutrace::mandatory` or a type holding a compile-time value of type `T`
 * @see cutrace::mandatory
 */
template <const char *name, typename T, typename D = mandatory> struct loader_argument;

/**
 * @brief Specialization of `cutrace::loader_argument` representing a mandatory argument in the schema.
 * @tparam name The name of the argument
 * @tparam T The type of the argument
 */
template <const char *name, typename T>
struct loader_argument<name, T, mandatory> {
  constexpr static const char *n = name; //!< The name of the argument
  using type = T; //!< The type of the argument
  using json_t = json_type<T>::type; //!< The JSON type corresponding to the type of the argument

  /**
   * @brief Attempts to load this argument from the given JSON object.
   * @param o The JSON object
   * @return Either a JSON error, or the argument value
   */
  static inline or_error<T> load_from(const picojson::object &o) {
    return coerce_key<T>(o, name);
  }
};

/**
 * @brief Specialization of `cutrace::loader_argument` representing an optional argument in the schema.
 * @tparam name The name of the argument
 * @tparam T The type of the argument
 * @tparam D A type representing a compile-time value of the argument type
 */
template <const char *name, typename T, typename D> requires(compile_time_T<D, T>)
struct loader_argument<name, T, D> {
  constexpr static const char *n = name; //!< The name of the argument
  using type = T; //!< The type of the argument
  using json_t = json_type<T>::type; //!< The JSON type corresponding to the type of the argument
  constexpr const static T default_v = D::value; //!< The default value

  /**
   * @brief Attempts to load this argument from the given JSON object, falling back to the default value if needed.
   * @param o The JSON object
   * @return Either a JSON error, or the default value, or the argument value
   */
  static inline or_error<T> load_from(const picojson::object &o) {
    return coerce_key<T>(o, name, default_v);
  }
};

/**
 * @brief Specialization of `cutrace::loader_argument` representing a mandatory 3D-vector in the schema.
 * @tparam name The name of the argument
 */
template <const char *name>
struct loader_argument<name, vector, mandatory> {
  constexpr static const char *n = name; //!< The name of the argument
  using type = vector; //!< The type of the argument
  using json_t = json_type<vector>::type; //!< The JSON type corresponding to the type of the argument

  /**
   * @brief Attempts to load this argument from the given JSON object.
   * @param o The JSON object
   * @return Either a JSON error, or the argument value
   */
  static inline or_error<vector> load_from(const picojson::object &o) {
    return coerce_key<json_t>(o, name).fmap([](const json_t &arr) -> or_error<vector> {
      if(arr.size() != 3) return {json_error{"Expected a 3-value array, got " + std::to_string(arr.size()) + " instead."} };

      return coerce<float>(arr[0]).fmap([&arr](const float &x) {
        return coerce<float>(arr[1]).fmap([&arr, x](const float &y) {
          return coerce<float>(arr[2]).fmap([x, y](const float &z) -> or_error<vector> {
            return { vector{ x, y, z } };
          });
        });
      });
    });
  }
};

/**
 * @brief Specialization of `cutrace::loader_argument` representing an optional 3D-vector in the schema.
 * @tparam name The name of the argument
 * @tparam D A type representing a compile-time 3D-vector
 */
template <const char *name, typename D> requires(compile_time_T<D, vector>)
struct loader_argument<name, vector, D> {
  constexpr static const char *n = name; //!< The name of the argument
  using type = vector; //!< The type of the argument
  using json_t = json_type<vector>::type; //!< The JSON type corresponding to the type of the argument
  constexpr const static vector default_v = D::value; //!< The default value

  /**
   * @brief Attempts to load this argument from the given JSON object, falling back to the default value if needed.
   * @param o The JSON object
   * @return Either a JSON error, or the default value, or the argument value
   */
  static inline or_error<vector> load_from(const picojson::object &o) {
    auto d = picojson::array{
      {picojson::value(default_v.x), picojson::value(default_v.y), picojson::value(default_v.z)}
    };

    return coerce_key(o, name, d).fmap([](const json_t &arr) -> or_error<vector> {
      if(arr.size() != 3) return {json_error{"Expected a 3-value array, got " + std::to_string(arr.size()) + " instead."} };

      return coerce<float>(arr[0]).fmap([&arr](const float &x) {
        return coerce<float>(arr[1]).fmap([&arr, x](const float &y) {
          return coerce<float>(arr[2]).fmap([x, y](const float &z) -> or_error<vector> {
            return { vector{ x, y, z } };
          });
        });
      });
    });
  }
};

/**
 * @brief Type trait to verify if a type `T` is constructible from given loader arguments.
 * @tparam T The type to verify
 * @tparam Ts The loader argument types
 *
 * If `Ts...` are not all specializations of `cutrace::loader_argument`, then this type will always hold a false value.
 */
template <typename T, typename ... Ts> struct constructible_from_arguments : std::bool_constant<false> {};

/**
 * @brief Specialization of `cutrace::constructible_from_arguments` where all of `Ts...` are `cutrace::loader_argument`.
 * @tparam T The type to verify
 * @tparam names The names of the arguments, in order
 * @tparam As The types of the arguments, in order
 * @tparam Ds The default values (or `cutrace::mandatory` tags), in order
 */
template <typename T, const char *... names, typename ... As, typename ... Ds>
struct constructible_from_arguments<T, loader_argument<names, As, Ds>...> : std::bool_constant<std::constructible_from<T, As...>> {};

/**
 * @brief Ease-of-use concept wrapping `cutrace::constructible_from_arguments`.
 * @tparam T The type to check
 * @tparam Ts The types of the arguments (should be specializations of `cutrace::loader_argument`)
 */
template <typename T, typename ... Ts>
concept constructible_from_args = constructible_from_arguments<T, Ts...>::value;

//endregion

//region object schema(s)

/**
 * @brief Structure representing an object schema.
 * @tparam name The name of the object (its type in JSON)
 * @tparam O The type of the object
 * @tparam Ts The parameters for this object
 */
template <const char *name, typename O, typename ... Ts> struct object_schema;

/**
 * @brief Structure representing an object schema.
 * @tparam name The name of the object (its type in JSON)
 * @tparam O The type of the object
 * @tparam names The names of its arguments, in order
 * @tparam Ts The types of its arguments, in order
 * @tparam Ds The default values for its arguments (or mandatory tags), in order
 */
template <const char *name, typename O, const char *... names, typename ... Ts, typename ... Ds>
  requires constructible_from_args<O, loader_argument<names, Ts, Ds>...>
struct object_schema<name, O, loader_argument<names, Ts, Ds>...> {
  using object_t = O; //!< The object type
  constexpr const static char *type = name; //!< The type's name

  /**
   * @brief Attempts to load the object from a JSON value.
   * @param o The JSON object
   * @return Either a JSON error, or the parsed object
   */
  static inline or_error<O> load_from(const picojson::object &o) {
    return fmap_all([](const auto &... args) -> O {
      return O(args...);
    }, loader_argument<names, Ts, Ds>::load_from(o)...);
  }
};

/**
 * Type trait to detect if something is an object schema.
 * @tparam T The type to check
 */
template <typename T> struct is_object_schema : std::bool_constant<false> {};
/**
 * Specialization of `cutrace::is_object_schema` for `cutrace::object_schema`.
 * @tparam name The object name
 * @tparam T The object type
 * @tparam Ts The object's arguments
 */
template <const char *name, typename T, typename ... Ts> struct is_object_schema<object_schema<name, T, Ts...>> : std::bool_constant<true> {};

/**
 * @brief Type trait to extract the `std::variant` for a set of object schemas.
 * @tparam Ts The loader arguments for the schema
 */
template <typename ... Ts> struct extract_object_variant;
/**
 * @brief Type trait to extract the `std::variant` for a set of object schemas.
 * @tparam Ts The loader arguments for the schema
 */
template <typename ... Ts> requires(is_object_schema<Ts>::value && ...)
struct extract_object_variant<Ts...> {
  using type = cpu::cpu_object_set<typename Ts::object_t...>;
};

/**
 * @brief Helper struct to recursively load a single object from a set of schemas.
 * @tparam Out The `std::variant` matching the object schemas
 * @tparam Ts The schemas
 */
template <typename Out, typename ... Ts> struct find_matching_object;
/**
 * @brief Helper struct to recursively load a single object from a set of schemas (base case).
 * @tparam Out The `std::variant` matching the original set of object schemas
 * @tparam T The final schema
 */
template <typename Out, typename T> requires(is_object_schema<T>::value)
struct find_matching_object<Out, T> {
  /**
   * @brief If the given key matches the type given in the schema, uses the schema to load the object.
   * @param key The key or type of the object
   * @param o The JSON object to load
   * @return Either a JSON error, or the loaded object
   */
  static inline or_error<Out> find(const std::string &key, const picojson::object &o) {
    if(T::type == key) return T::load_from(o).template re_wrap<Out>();
    return or_error<Out>::left(json_error{"Type '" + key + "' is invalid."});
  }
};

/**
 * @brief Helper struct to recursively load a single object from a set of schemas (recursive case).
 * @tparam Out The `std::variant` matching the (original) set of schemas
 * @tparam T1 The first schema
 * @tparam T2 The second schema
 * @tparam TRest The rest of the schemas
 */
template <typename Out, typename T1, typename T2, typename ... TRest>
  requires(is_object_schema<T1>::value && is_object_schema<T2>::value && (is_object_schema<TRest>::value && ...))
struct find_matching_object<Out, T1, T2, TRest...> {
  /**
   * @brief If the given key matches the type given in the first schema, uses that schema to load the object. Otherwise,
   * recurses with the next schema type.
   * @param key The key or type of the object
   * @param o The JSON object to load
   * @return Either a JSON error, or the loaded object
   */
  static inline or_error<Out> find(const std::string &key, const picojson::object &o) {
    if(T1::type == key) return T1::load_from(o).template re_wrap<Out>();
    return find_matching_object<Out, T2, TRest...>::find(key, o);
  }
};

/**
 * @brief Struct representing a set of object schemas.
 * @tparam Ts The schemas
 */
template <typename ... Ts> requires(is_object_schema<Ts>::value && ...)
struct all_objects_schema {
  using any = extract_object_variant<Ts...>::type; //!< The variant type for the schemas.

  /**
   * @brief Loads a single object.
   * @param o The JSON object to load from
   * @return Either a JSON error, or the loaded object
   */
  static inline or_error<any> load_from(const picojson::object &o) {
    return coerce_key<std::string>(o, "type").fmap([&o](const std::string &type) {
      return find_matching_object<any, Ts...>::find(type, o);
    });
  }
};

//endregion

//region light schema(s)

/**
 * @brief Structure representing a light schema.
 * @tparam name The name of the light (its type in JSON)
 * @tparam O The type of the light
 * @tparam Ts The parameters for this light
 */
template <const char *name, typename L, typename ... Ts> struct light_schema;

/**
 * @brief Structure representing a light schema.
 * @tparam name The name of the light (its type in JSON)
 * @tparam L The type of the light
 * @tparam names The names of its arguments, in order
 * @tparam Ts The types of its arguments, in order
 * @tparam Ds The default values for its arguments (or mandatory tags), in order
 */
template <const char *name, typename L, const char *... names, typename ... Ts, typename ... Ds>
  requires constructible_from_args<L, loader_argument<names, Ts, Ds>...>
struct light_schema<name, L, loader_argument<names, Ts, Ds>...> {
  using light_t = L; //!< The light type
  constexpr const static char *type = name; //!< The type's name

  /**
   * @brief Attempts to load the light from a JSON value.
   * @param o The JSON object
   * @return Either a JSON error, or the parsed light
   */
  static or_error<L> load_from(const picojson::object &o) {
    return fmap_all([](const auto &... args) -> L {
      return L(args...);
    }, loader_argument<names, Ts, Ds>::load_from(o)...);
  }
};

/**
 * Type trait to detect if something is a light schema.
 * @tparam T The type to check
 */
template <typename T> struct is_light_schema : std::bool_constant<false> {};
/**
 * Specialization of `cutrace::is_light_schema` for `cutrace::light_schema`.
 * @tparam name The light name
 * @tparam T The light type
 * @tparam Ts The light's arguments
 */
template <const char *name, typename T, typename ... Ts> struct is_light_schema<light_schema<name, T, Ts...>> : std::bool_constant<true> {};

/**
 * @brief Type trait to extract the `std::variant` for a set of light schemas.
 * @tparam Ts The loader arguments for the schema
 */
template <typename ... Ts> struct extract_light_variant;
/**
 * @brief Type trait to extract the `std::variant` for a set of light schemas.
 * @tparam Ts The loader arguments for the schema
 */
template <typename ... Ts> requires(is_light_schema<Ts>::value && ...)
struct extract_light_variant<Ts...> {
  using type = cpu::cpu_light_set<typename Ts::light_t...>;
};

/**
 * @brief Helper struct to recursively load a single light from a set of schemas.
 * @tparam Out The `std::variant` matching the light schemas
 * @tparam Ts The schemas
 */
template <typename Out, typename ... Ts> struct find_matching_light;
/**
 * @brief Helper struct to recursively load a single light from a set of schemas (base case).
 * @tparam Out The `std::variant` matching the original set of light schemas
 * @tparam T The final schema
 */
template <typename Out, typename T> requires(is_light_schema<T>::value)
struct find_matching_light<Out, T> {
  /**
   * @brief If the given key matches the type given in the schema, uses the schema to load the light.
   * @param key The key or type of the light
   * @param o The JSON object to load
   * @return Either a JSON error, or the loaded light
   */
  static inline or_error<Out> find(const std::string &key, const picojson::object &o) {
    if(T::type == key) return T::load_from(o).template re_wrap<Out>();
    return or_error<Out>::left(json_error{ "Type '" + key + "' is invalid." });
  }
};

/**
 * @brief Helper struct to recursively load a single light from a set of schemas (recursive case).
 * @tparam Out The `std::variant` matching the (original) set of schemas
 * @tparam T1 The first schema
 * @tparam T2 The second schema
 * @tparam TRest The rest of the schemas
 */
template <typename Out, typename T1, typename T2, typename ... TRest>
requires(is_light_schema<T1>::value && is_light_schema<T2>::value && (is_light_schema<TRest>::value && ...))
struct find_matching_light<Out, T1, T2, TRest...> {
  /**
   * @brief If the given key matches the type given in the first schema, uses that schema to load the light. Otherwise,
   * recurses with the next schema type.
   * @param key The key or type of the light
   * @param o The JSON object to load
   * @return Either a JSON error, or the loaded light
   */
  static inline or_error<Out> find(const std::string &key, const picojson::object &o) {
    if(T1::type == key) return T1::load_from(o).template re_wrap<Out>();
    return find_matching_light<Out, T2, TRest...>::find(key, o);
  }
};

/**
 * @brief Struct representing a set of light schemas.
 * @tparam Ts The schemas
 */
template <typename ... Ts> requires(is_light_schema<Ts>::value && ...)
struct all_lights_schema {
  using any = extract_light_variant<Ts...>::type; //!< The variant type for the schemas.

  /**
   * @brief Loads a single light.
   * @param o The JSON object to load from
   * @return Either a JSON error, or the loaded light
   */
  static inline or_error<any> load_from(const picojson::object &o) {
    return coerce_key<std::string>(o, "type").fmap([&o](const std::string &type) {
      return find_matching_light<any, Ts...>::find(type, o);
    });
  }
};

//endregion

//region material schema(s)

/**
 * @brief Structure representing a material schema.
 * @tparam name The name of the material (its type in JSON)
 * @tparam O The type of the material
 * @tparam Ts The parameters for this material
 */
template <const char *name, typename M, typename ... Ts> struct material_schema;

/**
 * @brief Structure representing a material schema.
 * @tparam name The name of the material (its type in JSON)
 * @tparam M The type of the material
 * @tparam names The names of its arguments, in order
 * @tparam Ts The types of its arguments, in order
 * @tparam Ds The default values for its arguments (or mandatory tags), in order
 */
template <const char *name, typename M, const char *... names, typename ... Ts, typename ... Ds>
struct material_schema<name, M, loader_argument<names, Ts, Ds>...> {
  using material_t = M; //!< The material type
  constexpr const static char *type = name; //!< The type's name

  /**
   * @brief Attempts to load the material from a JSON value.
   * @param o The JSON object
   * @return Either a JSON error, or the parsed material
   */
  static or_error<M> load_from(const picojson::object &o) {
    return fmap_all([](const auto &... args) -> M {
      return M(args...);
    }, loader_argument<names, Ts, Ds>::load_from(o)...);
  }
};

/**
 * Type trait to detect if something is a light schema.
 * @tparam T The type to check
 */
template <typename T> struct is_material_schema : std::bool_constant<false> {};
/**
 * Specialization of `cutrace::is_material_schema` for `cutrace::material_schema`.
 * @tparam name The material name
 * @tparam T The material type
 * @tparam Ts The material's arguments
 */
template <const char *name, typename M, typename ... Ts> struct is_material_schema<material_schema<name, M, Ts...>> : std::bool_constant<true> {};

/**
 * @brief Type trait to extract the `std::variant` for a set of material schemas.
 * @tparam Ts The loader arguments for the schema
 */
template <typename ... Ts> struct extract_material_variant;
/**
 * @brief Type trait to extract the `std::variant` for a set of material schemas.
 * @tparam Ts The loader arguments for the schema
 */
template <typename ... Ts> requires(is_material_schema<Ts>::value && ...)
struct extract_material_variant<Ts...> {
  using type = cpu::cpu_material_set<typename Ts::material_t...>;
};

/**
 * @brief Helper struct to recursively load a single material from a set of schemas.
 * @tparam Out The `std::variant` matching the material schemas
 * @tparam Ts The schemas
 */
template <typename Out, typename ... Ts> struct find_matching_mat;
/**
 * @brief Helper struct to recursively load a single material from a set of schemas (base case).
 * @tparam Out The `std::variant` matching the original set of material schemas
 * @tparam T The final schema
 */
template <typename Out, typename T> requires(is_material_schema<T>::value)
struct find_matching_mat<Out, T> {
  /**
   * @brief If the given key matches the type given in the schema, uses the schema to load the material.
   * @param key The key or type of the material
   * @param o The JSON object to load
   * @return Either a JSON error, or the loaded material
   */
  static inline or_error<Out> find(const std::string &key, const picojson::object &o) {
    if(T::type == key) return T::load_from(o).template re_wrap<Out>();
    return or_error<Out>::left(json_error{ "Type '" + key + "' is invalid." });
  }
};

/**
 * @brief Helper struct to recursively load a single material from a set of schemas (recursive case).
 * @tparam Out The `std::variant` matching the (original) set of schemas
 * @tparam T1 The first schema
 * @tparam T2 The second schema
 * @tparam TRest The rest of the schemas
 */
template <typename Out, typename T1, typename T2, typename ... TRest>
requires(is_light_schema<T1>::value && is_light_schema<T2>::value && (is_material_schema<TRest>::value && ...))
struct find_matching_mat<Out, T1, T2, TRest...> {
  /**
   * @brief If the given key matches the type given in the first schema, uses that schema to load the material. Otherwise,
   * recurses with the next schema type.
   * @param key The key or type of the material
   * @param o The JSON object to load
   * @return Either a JSON error, or the loaded material
   */
  static inline or_error<Out> find(const std::string &key, const picojson::object &o) {
    if(T1::type == key) return T1::load_from(o).template re_wrap<Out>();
    return find_matching_mat<Out, T2, TRest...>::find(key, o);
  }
};

/**
 * @brief Struct representing a set of material schemas.
 * @tparam Ts The schemas
 */
template <typename ... Ts> requires(is_material_schema<Ts>::value && ...)
struct all_materials_schema {
  using any = extract_material_variant<Ts...>::type; //!< The variant type for the schemas.

  /**
   * @brief Loads a single material.
   * @param o The JSON object to load from
   * @return Either a JSON error, or the loaded material
   */
  static inline or_error<any> load_from(const picojson::object &o) {
    return coerce_key<std::string>(o, "type").fmap([&o](const std::string &type) {
      return find_matching_mat<any, Ts...>::find(type, o);
    });
  }
};

//endregion

//region camera schema

/**
 * @brief Structure representing camera schema.
 * @tparam name The name of the camera (its type in JSON)
 * @tparam O The type of the camera
 * @tparam Ts The parameters for this camera
 */
template <typename C, typename ... Ts> struct cam_schema;

/**
 * @brief Structure representing a camera schema.
 * @tparam name The name of the camera (its type in JSON)
 * @tparam O The type of the camera
 * @tparam names The names of its arguments, in order
 * @tparam Ts The types of its arguments, in order
 * @tparam Ds The default values for its arguments (or mandatory tags), in order
 */
template <typename C, const char *... names, typename ... Ts, typename ... Ds>
requires constructible_from_args<C, loader_argument<names, Ts, Ds>...>
struct cam_schema<C, loader_argument<names, Ts, Ds>...> {
  using cam_t = C; //!< The camera type

  /**
   * @brief Attempts to load the camera from a JSON value.
   * @param o The JSON object
   * @return Either a JSON error, or the parsed camera
   */
  static inline or_error<C> load_from(const picojson::object &o) {
    auto res = fmap_all([](const auto &... args) -> C {
      return C(args...);
    }, loader_argument<names, Ts, Ds>::load_from(o)...);
    return res;
  }
};
//endregion

/**
 * @brief Struct representing the full schema for a scene.
 * @tparam O The full object schema
 * @tparam L The full light schema
 * @tparam M The full material schema
 * @tparam C The camera schema
 */
template <typename O, typename L, typename M, typename C> struct full_schema;

/**
 * @brief Struct representing the full schema for a scene.
 * @tparam Os The types of object schemas
 * @tparam Ls The types of light schemas
 * @tparam Ms The types of material schemas
 * @tparam C The camera schema
 * @tparam c_names The names of the arguments for the camera schema
 * @tparam CTypes The types of the arguments for the camera schema
 * @tparam CDs The default values for the camera schema (or mandatory tags)
 */
template <typename ... Os, typename ... Ls, typename ... Ms, typename C, const char *...c_names, typename ... CTypes, typename ... CDs>
struct full_schema<all_objects_schema<Os...>, all_lights_schema<Ls...>, all_materials_schema<Ms...>, cam_schema<C, loader_argument<c_names, CTypes, CDs>...>> {
  using object_schema = all_objects_schema<Os...>; //!< The type of the object schemas.
  using light_schema = all_lights_schema<Ls...>; //!< The type of the light schemas.
  using material_schema = all_materials_schema<Ms...>; //!< The type of the material schemas.
  using camera_schema = cam_schema<C, loader_argument<c_names, CTypes, CDs>...>; //!< The type of the camera schema.
  using object_t = object_schema::any; //!< The `std::variant` that the object schema loads.
  using light_t = light_schema::any; //!< The `std::variant` that the light schema loads.
  using material_t = material_schema::any; //!< The `std::variant` that the material schema loads.
  using cam_t = C; //!< The type that the camera schema loads.
  using scene_t = cpu::cpu_scene<object_t, light_t, material_t, cam_t>; //!< The type of the CPU scene that this loader loads.

  static inline bool last_was_success = true; //!< Keeps track of whether the last load was successful.

  /**
   * @brief Loads a scene from a JSON object.
   * @param o The object to load from
   * @return The loaded scene, or an empty one on failure
   *
   * If the loading fails, errors will be logged to `stderr`, and `full_schema::last_was_success` will be set to `false`.
   * @see load_file
   */
  static inline scene_t load_from(const picojson::object &o) {
    last_was_success = true;

    std::vector<object_t> objects{};
    std::vector<light_t> lights{};
    std::vector<material_t> materials{};
    cam_t camera{};

    coerce_key<picojson::array>(o, "objects").map([&objects](const picojson::array &objs) {
      objects.reserve(objs.size());
      for(size_t i = 0; i < objs.size(); i++) {
        force_object(objs[i]).fmap([&objects](const auto *v) {
          return object_schema::load_from(*v).map([&objects](const auto &obj) {
            objects.push_back(obj);
          });
        }).map_left([i](const json_error &err) {
          std::cerr << "Error while loading object #" << i << ": " << err.message << "\n";
          last_was_success = false;
        });
      }
    }).map_left([](const json_error &err) {
      std::cerr << "Could not find 'objects' array: " << err.message << ".\n";
      last_was_success = false;
    });

    coerce_key<picojson::array>(o, "lights").map([&lights](const picojson::array &objs) {
      lights.reserve(objs.size());
      for(size_t i = 0; i < objs.size(); i++) {
        force_object(objs[i]).fmap([&lights](const auto *v) {
          return light_schema::load_from(*v).map([&lights](const auto &light) {
            lights.push_back(light);
          });
        }).map_left([i](const json_error &err) {
          std::cerr << "Error while loading light #" << i << ": " << err.message << "\n";
          last_was_success = false;
        });
      }
    }).map_left([](const json_error &err) {
      std::cerr << "Could not find 'lights' array: " << err.message << ".\n";
      last_was_success = false;
    });

    coerce_key<picojson::array>(o, "materials").map([&materials](const picojson::array &objs) {
      materials.reserve(objs.size());
      for(size_t i = 0; i < objs.size(); i++) {
        force_object(objs[i]).fmap([&materials](const auto *v) {
          return material_schema::load_from(*v).map([&materials](const auto &material) {
            materials.push_back(material);
          });
        }).map_left([i](const json_error &err) {
          std::cerr << "Error while loading material #" << i << ": " << err.message << "\n";
          last_was_success = false;
        });
      }
    }).map_left([](const json_error &err) {
      std::cerr << "Could not find 'materials' array: " << err.message << ".\n";
      last_was_success = false;
    });

    coerce_key<picojson::object>(o, "camera").fmap([&camera](const picojson::object &obj) {
      return camera_schema::load_from(obj).map([&camera](const auto &c) {
        camera = c;
      });
    }).map_left([](const json_error &err) {
      std::cerr << "Could not find 'camera' object or it's invalid: " << err.message << ".\n";
      last_was_success = false;
    });

    return scene_t {
      .objects = objects,
      .lights = lights,
      .materials = materials,
      .cam = camera
    };
  }

  /**
   * @brief Loads a scene from a JSON file.
   * @param file The file to load
   * @return The loaded scene, or an empty one on failure
   *
   * If the loading fails, errors will be logged to `stderr`, and `full_schema::last_was_success` will be set to `false`.
   * @see load_from
   */
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
}

#endif //CUTRACE_LOADER_HPP
