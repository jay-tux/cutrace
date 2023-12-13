//
// Created by jay on 11/28/23.
//

#ifndef CUTRACE_CPU_TYPES_HPP
#define CUTRACE_CPU_TYPES_HPP

#include <variant>
#include <vector>

/**
 * @brief Main namespace for GPU-related code.
 */
namespace cutrace::cpu {
/**
 * @brief Type alias for `std::variant<Ts...>`.
 */
template <typename ... Ts>
using cpu_object_set = std::variant<Ts...>;

/**
 * @brief Type alias for `std::variant<Ts...>`.
 */
template <typename ... Ts>
using cpu_light_set = std::variant<Ts...>;

/**
 * @brief Type alias for `std::variant<Ts...>`.
 */
template <typename ... Ts>
using cpu_material_set = std::variant<Ts...>;

/**
 * @brief Struct representing a scene on CPU-side.
 * @tparam O The type of the objects
 * @tparam L The type of the lights
 * @tparam M The type of the materials
 * @tparam C The type of the camera
 */
template <typename O, typename L, typename M, typename C> struct cpu_scene;

/**
 * @brief Struct representing a scene on CPU-side.
 * @tparam Os The types of the objects
 * @tparam Ls The types of the lights
 * @tparam Ms The types of the materials
 * @tparam C The type of the camera
 */
template <typename ... Os, typename ... Ls, typename ... Ms, typename C> requires(std::default_initializable<C>)
struct cpu_scene<cpu_object_set<Os...>, cpu_light_set<Ls...>, cpu_material_set<Ms...>, C> {
  using object = cpu_object_set<Os...>; //!< The type of the objects
  using light = cpu_light_set<Ls...>; //!< The type of the lights
  using material = cpu_material_set<Ms...>; //!< The type of the materials
  using camera = C; //!< The type of the camera

  std::vector<object> objects; //!< The objects in the scene
  std::vector<light> lights; //!< The lights in the scene
  std::vector<material> materials; //!< The materials in the scene
  camera cam; //!< The camera rendering the scene
};

}

#endif //CUTRACE_CPU_TYPES_HPP
