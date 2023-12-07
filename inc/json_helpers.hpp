//
// Created by jay on 11/28/23.
//

#ifndef CUTRACE_JSON_HELPERS_HPP
#define CUTRACE_JSON_HELPERS_HPP

#include <string>
#include "picojson.h"
#include "either.hpp"
#include "vector.hpp"

namespace cutrace {
struct json_error {
  std::string message;
};

template <const char name[], typename T>
struct name_type {
  using type = T;
  constexpr const static char *type_name = name;
};

struct json_type_names {
  constexpr const static char s_bool[] = "bool";
  constexpr const static char s_short[] = "short";
  constexpr const static char s_int[] = "int";
  constexpr const static char s_uint[] = "uint";
  constexpr const static char s_long[] = "long";
  constexpr const static char s_size_t[] = "size_t";
  constexpr const static char s_float[] = "float";
  constexpr const static char s_double[] = "double";
  constexpr const static char s_array[] = "array";
  constexpr const static char s_string[] = "string";
  constexpr const static char s_object[] = "object";
};

template <typename T> struct json_type;
template <> struct json_type<bool> : name_type<json_type_names::s_bool, bool> {};
template <> struct json_type<short> : name_type<json_type_names::s_short, double> {};
template <> struct json_type<int> : name_type<json_type_names::s_int, double> {};
template <> struct json_type<unsigned int> : name_type<json_type_names::s_uint, double> {};
template <> struct json_type<long> : name_type<json_type_names::s_long, double> {};
template <> struct json_type<size_t> : name_type<json_type_names::s_size_t, double> {};
template <> struct json_type<float> : name_type<json_type_names::s_float, double> {};
template <> struct json_type<double> : name_type<json_type_names::s_double, double> {};
template <> struct json_type<picojson::array> : name_type<json_type_names::s_array, picojson::array> {};
template <> struct json_type<vector> : name_type<json_type_names::s_array, picojson::array> {};
template <> struct json_type<std::string> : name_type<json_type_names::s_string, std::string> {};
template <> struct json_type<picojson::object> : name_type<json_type_names::s_object, picojson::object> {};

template <typename T> using json_either = either<json_error, T>;

template <typename T>
constexpr auto coerce(const picojson::value &v) -> json_either<T> {
  using t = json_type<T>::type;
  if(v.is<t>()) return json_either<T>::right((T)v.get<t>());
  else return json_either<T>::left(json_error{std::string("Expected a value of type ") + (const char *)json_type<T>::type_name + std::string(".")});
}

template <typename T>
inline auto coerce_key(const picojson::object &o, const std::string &name) -> json_either<T> {
  auto it = o.find(name);
  if(it != o.end()) return coerce<T>(it->second);
  else return { json_error{std::string("Cannot find key '") + name + std::string("' in object.")} };
}

template <typename T>
inline auto coerce_key(const picojson::object &o, const std::string &name, const T &def) -> json_either<T> {
  auto it = o.find(name);
  if(it != o.end()) return coerce<T>(it->second);
  else { return {(T) def}; }
}

inline either<json_error, const picojson::object *> force_object(const picojson::value &v) {
  if(v.is<picojson::object>()) return { &v.get<picojson::object>() };
  else return { json_error{"Value is not a JSON object."} };
}
}

#endif //CUTRACE_JSON_HELPERS_HPP
