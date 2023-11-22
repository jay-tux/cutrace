//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_LOADER_HPP
#define CUTRACE_LOADER_HPP

#include <vector>
#include <string>
#include "cpu_types.hpp"

namespace cutrace {
struct loader {
  static cpu::cpu_scene load(const std::string &file);
};
}

#endif //CUTRACE_LOADER_HPP
