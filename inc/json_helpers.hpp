//
// Created by jay on 11/28/23.
//

#ifndef CUTRACE_JSON_HELPERS_HPP
#define CUTRACE_JSON_HELPERS_HPP

#include <string>
#include "picojson.h"
#include "either.hpp"
#include "vector.hpp"

/**
 * @brief Namespace containing all of cutrace’s code.
 */
namespace cutrace {
/**
 * @brief Structure representing an error while loading a JSON file.
 */
struct json_error {
  std::string message; //!< The error message
};

/**
 * @brief Type trait to keep track of a type and its name.
 * @tparam name The name of the type
 * @tparam T The type
 */
template <const char name[], typename T>
struct name_type {
  using type = T; //!< The type
  constexpr const static char *type_name = name; //!< A pointer to the name of the type
};

/**
 * @brief Struct to keep track of the names of all types used by the JSON loader.
 */
struct json_type_names {
  constexpr const static char s_bool[] = "bool"; //!< Type name for a `bool`
  constexpr const static char s_short[] = "short"; //!< Type name for a `short`
  constexpr const static char s_int[] = "int"; //!< Type name for an `int`
  constexpr const static char s_uint[] = "uint"; //!< Type name for an `unsigned int`
  constexpr const static char s_long[] = "long"; //!< Type name for a `long int`
  constexpr const static char s_size_t[] = "size_t"; //!< Type name for a `size_t`
  constexpr const static char s_float[] = "float"; //!< Type name for a `float`
  constexpr const static char s_double[] = "double"; //!< Type name for a `double`
  constexpr const static char s_array[] = "array"; //!< Type name for a JSON array
  constexpr const static char s_string[] = "string"; //!< Type name for an `std::string`
  constexpr const static char s_object[] = "object"; //!< Type name for a JSON object
};

/**
 * @brief Type trait to keep track of JSON types.
 * @tparam T The type
 *
 * Each of these have a member type `type` that holds the corresponding type used by the PicoJSON library, as well as
 * its type name.
 */
template <typename T> struct json_type;
template <> struct json_type<bool> : name_type<json_type_names::s_bool, bool> {}; //!< Type trait that maps `bool` to the corresponding PicoJSON type.
template <> struct json_type<short> : name_type<json_type_names::s_short, double> {}; //!< Type trait that maps `short` to the corresponding PicoJSON type.
template <> struct json_type<int> : name_type<json_type_names::s_int, double> {}; //!< Type trait that maps `int` to the corresponding PicoJSON type.
template <> struct json_type<unsigned int> : name_type<json_type_names::s_uint, double> {}; //!< Type trait that maps `unsigned int` to the corresponding PicoJSON type.
template <> struct json_type<long> : name_type<json_type_names::s_long, double> {}; //!< Type trait that maps `long` to the corresponding PicoJSON type.
template <> struct json_type<size_t> : name_type<json_type_names::s_size_t, double> {}; //!< Type trait that maps `size_t` to the corresponding PicoJSON type.
template <> struct json_type<float> : name_type<json_type_names::s_float, double> {}; //!< Type trait that maps `float` to the corresponding PicoJSON type.
template <> struct json_type<double> : name_type<json_type_names::s_double, double> {}; //!< Type trait that maps `double` to the corresponding PicoJSON type.
template <> struct json_type<picojson::array> : name_type<json_type_names::s_array, picojson::array> {}; //!< Type trait that maps `picojson::array` to the corresponding PicoJSON type.
template <> struct json_type<vector> : name_type<json_type_names::s_array, picojson::array> {}; //!< Type trait that maps `cutrace::vector` to the corresponding PicoJSON type.
template <> struct json_type<std::string> : name_type<json_type_names::s_string, std::string> {}; //!< Type trait that maps `std::string` to the corresponding PicoJSON type.
template <> struct json_type<picojson::object> : name_type<json_type_names::s_object, picojson::object> {}; //!< Type trait that maps `picojson::object` to the corresponding PicoJSON type.

/**
 * @brief Type alias for `cutrace::either<cutrace::json_error, T>`.
 * @tparam T The type in the either.
 * @see cutrace::either
 */
template <typename T> using json_either = either<json_error, T>;

/**
 * Attempts to convert a generic JSON value to the given type.
 * @tparam T The type to convert to
 * @param v The JSON value to convert
 * @return Either a JSON error, or the converted value
 *
 * This function requires `T` to be one of the specializations of `cutrace::json_type`.
 */
template <typename T>
constexpr auto coerce(const picojson::value &v) -> json_either<T> {
  using t = json_type<T>::type;
  if(v.is<t>()) return json_either<T>::right((T)v.get<t>());
  else return json_either<T>::left(json_error{std::string("Expected a value of type ") + (const char *)json_type<T>::type_name + std::string(".")});
}

/**
 * Attempts to get a value from the given object by key, then converts it to the required type.
 * @tparam T The type to convert to
 * @param o The JSON object to search in
 * @param name The key to search for
 * @return Either a JSON error, or the value matching the key, converted to `T`
 *
 * This function requires `T` to be one of the specializations of `cutrace::json_type`.
 */
template <typename T>
inline auto coerce_key(const picojson::object &o, const std::string &name) -> json_either<T> {
  auto it = o.find(name);
  if(it != o.end()) return coerce<T>(it->second);
  else return { json_error{std::string("Cannot find key '") + name + std::string("' in object.")} };
}

/**
 * Attempts to get a value from the given object by key, then converts it to the required type.
 * @tparam T The type to convert to
 * @param o The JSON object to search in
 * @param name The key to search for
 * @param def The default to use
 * @return Either a JSON error, or a value of type `T`
 *
 * This function requires `T` to be one of the specializations of `cutrace::json_type`.
 *
 * In case the key does not exist, the default value `def` is returned. In case the key does exist, but is of the wrong
 * type, a JSON error is returned.
 */
template <typename T>
inline auto coerce_key(const picojson::object &o, const std::string &name, const T &def) -> json_either<T> {
  auto it = o.find(name);
  if(it != o.end()) return coerce<T>(it->second);
  else { return {(T) def}; }
}

/**
 * Attempts to convert the given generic JSON value to a JSON object.
 * @param v The value to check
 * @return Either a JSON error, or `v` as a JSON object
 */
inline either<json_error, const picojson::object *> force_object(const picojson::value &v) {
  if(v.is<picojson::object>()) return { &v.get<picojson::object>() };
  else return { json_error{"Value is not a JSON object."} };
}
}

#endif //CUTRACE_JSON_HELPERS_HPP
