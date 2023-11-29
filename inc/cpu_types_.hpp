//
// Created by jay on 11/28/23.
//

#ifndef CUTRACE_CPU_TYPES__HPP
#define CUTRACE_CPU_TYPES__HPP

#include <variant>
#include <vector>

namespace cutrace::cpu {
template <typename ... Ts>
using cpu_object_set = std::variant<Ts...>;

template <typename ... Ts>
using cpu_light_set = std::variant<Ts...>;

template <typename ... Ts>
using cpu_material_set = std::variant<Ts...>;

template <typename O, typename L, typename M> struct cpu_scene_;

template <typename ... Os, typename ... Ls, typename ... Ms>
struct cpu_scene_<cpu_object_set<Os...>, cpu_light_set<Ls...>, cpu_material_set<Ms...>> {
  using object = cpu_object_set<Os...>;
  using light = cpu_light_set<Ls...>;
  using material = cpu_material_set<Ms...>;

  std::vector<object> objects;
  std::vector<light> lights;
  std::vector<material> materials;
};

}

#endif //CUTRACE_CPU_TYPES__HPP
