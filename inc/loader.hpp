//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_LOADER_HPP
#define CUTRACE_LOADER_HPP

#include <vector>
#include <string>
#include "cpu_types.hpp"

/**
 * @brief Namespace containing all of cutrace's code.
 */
namespace cutrace {
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
