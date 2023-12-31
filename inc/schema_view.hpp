//
// Created by jay on 12/3/23.
//

#ifndef CUTRACE_SCHEMA_VIEW_HPP
#define CUTRACE_SCHEMA_VIEW_HPP

#include <iostream>
#include <typeinfo>
#include <string>
#include <cxxabi.h>
#include "loader.hpp"

/**
 * @brief Namespace for the CPU schema
 */
namespace cutrace::cpu::schema {
/**
 * @brief Helper struct to print the type name of a type at runtime (RTTI).
 * @tparam T The type to print
 *
 * This struct relies on `typeid(T).name` and the ABI function `abi::__cxa_demangle`.
 */
template <typename T> struct type {
  /**
   * @brief Gets the name of the type.
   * @return The type name
   */
  static inline std::string dump() {
    return abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
  }
};

/**
 * @brief Concept relating what it means to be printable.
 * @tparam T The type to check
 *
 * A type is printable if the operation `std::cout << t` is valid and returns `decltype(std::cout) &` for any `const T &t`.
 */
template <typename T>
concept printable = requires(const T &t) {
  { std::cout << t } -> std::same_as<decltype(std::cout) &>;
};

static_assert(printable<vector>);

namespace impl {
template<typename T>
struct loader_argument_viewer;

template<const char *name, typename T>
struct loader_argument_viewer<loader_argument<name, T, mandatory>> {
  static void dump_schema() {
    std::cout << " |   -> Mandatory argument '" << name << "' of (CUDA/C++) type " << type<T>::dump() << "\n";
  }
};

template<const char *name, typename T, typename D>
struct loader_argument_viewer<loader_argument<name, T, D>> {
  static void dump_schema() {
    std::cout << " |   -> Optional argument '" << name << "' of (CUDA/C++) type " << type<T>::dump() << "\n"
              << " |          Default value: ";

    if constexpr (printable<T>) std::cout << D::value;
    else std::cout << "(unprintable)";

    std::cout << "\n";
  }
};

template<typename ... Ts>
struct loader_argument_set_viewer;

template<>
struct loader_argument_set_viewer<> {
  static inline void dump_schema() {
    std::cout << " |   -> (no arguments)\n";
  }
};

template<const char *n1, const char *... ns, typename T1, typename ... Ts, typename D1, typename ... Ds>
struct loader_argument_set_viewer<loader_argument<n1, T1, D1>, loader_argument<ns, Ts, Ds>...> {
  static inline void dump_schema() {
    loader_argument_viewer<loader_argument<n1, T1, D1>>::dump_schema();
    if constexpr (sizeof...(ns) > 0)
      loader_argument_set_viewer<loader_argument<ns, Ts, Ds>...>::dump_schema();
  }
};

template<typename T>
struct object_schema_viewer;

template<const char *name, typename O, typename ... Ts>
struct object_schema_viewer<object_schema<name, O, Ts...>> {
  static inline void dump_schema() {
    std::cout << " |-> Object  type '" << name << "': \n";
    loader_argument_set_viewer<Ts...>::dump_schema();
  }
};

template<typename T>
struct light_schema_viewer;

template<const char *name, typename L, typename ... Ts>
struct light_schema_viewer<light_schema<name, L, Ts...>> {
  static inline void dump_schema() {
    std::cout << " |-> Light   type '" << name << "': \n";
    loader_argument_set_viewer<Ts...>::dump_schema();
  }
};

template<typename T>
struct material_schema_viewer;

template<const char *name, typename M, typename ... Ts>
struct material_schema_viewer<material_schema<name, M, Ts...>> {
  static inline void dump_schema() {
    std::cout << " |-> Material type '" << name << "': \n";
    loader_argument_set_viewer<Ts...>::dump_schema();
  }
};

template<typename T>
struct camera_schema_viewer;

template<typename T, typename ... Ts>
struct camera_schema_viewer<cam_schema<T, Ts...>> {
  static inline void dump_schema() {
    std::cout << " |-> Camera: \n";
    loader_argument_set_viewer<Ts...>::dump_schema();
  }
};

template<typename T1, typename ... Ts> requires(is_object_schema<T1>::value &&(is_object_schema<Ts>::value &&...))
struct any_object_schema_viewer {
  static inline void dump_schema() {
    object_schema_viewer<T1>::dump_schema();
    if constexpr (sizeof...(Ts) > 0) any_object_schema_viewer<Ts...>::dump_schema();
  }
};

template<typename T1, typename ... Ts> requires(is_light_schema<T1>::value &&(is_light_schema<Ts>::value &&...))
struct any_light_schema_viewer {
  static inline void dump_schema() {
    light_schema_viewer<T1>::dump_schema();
    if constexpr (sizeof...(Ts) > 0) any_light_schema_viewer<Ts...>::dump_schema();
  }
};

template<typename T1, typename ... Ts> requires(is_material_schema<T1>::value &&(is_material_schema<Ts>::value &&...))
struct any_material_schema_viewer {
  static inline void dump_schema() {
    material_schema_viewer<T1>::dump_schema();
    if constexpr (sizeof...(Ts) > 0) any_material_schema_viewer<Ts...>::dump_schema();
  }
};
}

/**
 * @brief Helper struct to print a full schema.
 * @tparam O The object schemas
 * @tparam L The light schemas
 * @tparam M The material schemas
 * @tparam C The camera schema
 */
template<typename O, typename L, typename M, typename C>
struct schema_viewer;

/**
 * @brief Helper struct to print a full schema.
 * @tparam Os The set of object schemas
 * @tparam Ls The set of light schemas
 * @tparam Ms The set of material schemas
 * @tparam C The camera type
 * @tparam c_names The camera schema argument names
 * @tparam CTypes The camera schema argument types
 * @tparam CDs The camera schema argument default values (or their mandatory tags)
 */
template<typename ... Os, typename ... Ls, typename ... Ms, typename C, const char *... c_names, typename ... CTypes, typename ... CDs>
struct schema_viewer<all_objects_schema<Os...>, all_lights_schema<Ls...>, all_materials_schema<Ms...>, cam_schema<C, loader_argument<c_names, CTypes, CDs>...>> {
  using schema_t = full_schema<all_objects_schema<Os...>, all_lights_schema<Ls...>, all_materials_schema<Ms...>, cam_schema<C, loader_argument<c_names, CTypes, CDs>...>>; //!< The actual schema
  using object_schema_t = schema_t::object_schema; //!< The object schema set
  using light_schema_t = schema_t::light_schema; //!< The light schema set
  using material_schema_t = schema_t::material_schema; //!< The material schema set
  using camera_schema_t = cam_schema<C, loader_argument<c_names, CTypes, CDs>...>; //!<  The camera schema

  /**
   * @brief Prints the schema details to `stdout`.
   */
  static inline void dump_schema() {
    std::cout << " +----- Object   types in schema: ----- \n";
    impl::any_object_schema_viewer<Os...>::dump_schema();
    std::cout << " |\n"
              << " +----- Light    types in schema: ----- \n";
    impl::any_light_schema_viewer<Ls...>::dump_schema();
    std::cout << " |\n"
              << " +----- Material types in schema: ----- \n";
    impl::any_material_schema_viewer<Ms...>::dump_schema();
    std::cout << " |\n"
              << " +----- Camera   type  in schema: ----- \n";
    impl::camera_schema_viewer<camera_schema_t>::dump_schema();
  }
};


/**
 * @brief Helper struct to get a viewer for a full schema.
 * @tparam T The type of the full schema
 */
template <typename T> struct viewer_for;

/**
 * Helper struct to get a viewer for a full schema.
 * @tparam O The set of object schema types
 * @tparam L The set of light schema types
 * @tparam M The set of material schema types
 * @tparam C The camera schema type
 */
template <typename O, typename L, typename M, typename C>
struct viewer_for<full_schema<O, L, M, C>> {
  using type = schema_viewer<O, L, M, C>; //!< The viewer type
};

/**
 * Type alias for `viewer_for<T>::type`.
 */
template <typename T>
using viewer_for_t = typename viewer_for<T>::type;
}

#endif //CUTRACE_SCHEMA_VIEW_HPP
