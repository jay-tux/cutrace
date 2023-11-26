//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_CPU_TYPES_HPP
#define CUTRACE_CPU_TYPES_HPP

#include <vector>
#include <variant>
#include "gpu_types.hpp"

/**
 * @brief Main namespace for CPU-related code.
 */
namespace cutrace::cpu {
/**
 * @brief Structure representing a triangle on CPU.
 *
 * This structure expects the three corners to be in counter-clockwise order.
 */
struct triangle {
  vector p1, //!< The first point of the triangle
         p2, //!< The second point of the triangle
         p3; //!< The third point of the triangle
  size_t mat_idx; //!< The index of the material

  /**
   * @brief Converts this triangle to GPU.
   * @return The GPU-representation of this triangle
   */
  __host__ gpu::triangle to_gpu() const;
};

/**
 * @brief Structure representing a model on CPU.
 */
struct triangle_set {
  std::vector<triangle> tris; //!< The triangles of the mesh
  size_t mat_idx; //!< The index of the material

  /**
   * @brief Converts this model to GPU.
   * @return The GPU-representation of this mesh
   */
  __host__ gpu::triangle_set to_gpu() const;
};

/**
 * @brief Structure representing a plane on CPU.
 */
struct plane {
  vector point; //!< A point in the plane
  vector normal; //!< A normal vector to the plane
  size_t mat_idx; //!< The index of the material

  /**
   * @brief Converts this plane to GPU.
   * @return The GPU-representation of this plane
   */
  __host__ gpu::plane to_gpu() const;
};

/**
 * @brief Structure representing a sphere on CPU.
 */
struct sphere {
  vector center; //!< The center of the sphere
  float radius; //!< The radius of the sphere
  size_t mat_idx; //!< The index of the material

  /**
   * @brief Converts this sphere to GPU.
   * @return The GPU-representation of this sphere
   */
  __host__ gpu::sphere to_gpu() const;
};

/**
 * @brief Type alias for a renderable object on CPU (either \ref cutrace::cpu::triangle, \ref cutrace::cpu::triangle_set,
 * \ref cutrace::cpu::plane, or \ref cutrace::cpu::sphere).
 */
using cpu_object = std::variant<triangle, triangle_set, plane, sphere>;

/**
 * @brief Converts a CPU object to GPU.
 * @param [in] cpu The CPU object to convert
 * @return A copy of `cpu`, but in GPU memory
 */
__host__ gpu::gpu_object to_gpu(const cpu_object &cpu);
/**
 * @brief Converts a set of CPU objects to GPU.
 * @param [in] cpus The CPU objects to convert
 * @return The same CPU objects, but on GPU
 */
__host__ gpu::gpu_array<gpu::gpu_object> to_gpu(const std::vector<cpu_object> &cpus);

/**
 * @brief A structure representing a sun (directional light) on CPU.
 */
struct sun {
  vector direction; //!< The direction of the light source
  vector color; //!< The color of the light

  /**
   * @brief Converts this sun (directional light) to GPU.
   * @return The GPU-representation of this sun
   */
  __host__ gpu::sun to_gpu() const;
};

/**
 * @brief A structure representing a point light on CPU.
 */
struct point_light {
  vector point; //!< The point from where the light shines
  vector color; //!< The color of the light

  /**
   * @brief Converts this point light to GPU.
   * @return The GPU-representation of this point light
   */
  __host__ gpu::point_light to_gpu() const;
};

/**
 * @brief Type alias for a light on CPU (either \ref cutrace::cpu::sun or \ref cutrace::cpu::point_light)
 */
using cpu_light = std::variant<sun, point_light>;

/**
 * @brief Converts a CPU light to GPU.
 * @param [in] cpu The CPU light to convert
 * @return A copy of `cpu`, but in GPU memory
 */
__host__ gpu::gpu_light to_gpu(const cpu_light &cpu);
/**
 * @brief Converts a set of CPU lights to GPU.
 * @param [in] cpus The CPU lights to convert
 * @return The same CPU lights, but on GPU
 */
__host__ gpu::gpu_array<gpu::gpu_light> to_gpu(const std::vector<cpu_light>& cpus);

/**
 * @brief Struct representing a material on CPU.
 */
struct cpu_mat {
  vector color; //!< The base color of the material
  float specular, //!< The specular factor for the material (how smooth/shiny it is)
        reflexivity, //!< The reflexivity factor for the material (how much it reflects/mirrors)
        phong_exp, //!< The Phong lighting exponent for the material
        transparency; //!< The transparency/translucency factor for the material

  /**
   * @brief Converts this material to GPU.
   * @return The same material, but in GPU memory
   */
  [[nodiscard]] __host__ gpu::gpu_mat to_gpu() const;
};

/**
 * @brief Converts a set of CPU materials to GPU.
 * @param [in] cpus The CPU materials to convert
 * @return The same CPU materials, but on GPU
 */
__host__ gpu::gpu_array<gpu::gpu_mat> to_gpu(const std::vector<cpu_mat> &cpus);

/**
 * @brief Struct containing all information about a scene, on CPU.
 */
struct cpu_scene {
  gpu::cam camera; //!< The camera information
  std::vector<cpu_object> objects; //!< The objects to be rendered
  std::vector<cpu_light> lights; //!< The lights to render with
  std::vector<cpu_mat> materials; //!< All used materials in the scene

  /**
   * @brief Converts this scene to GPU.
   * @return The GPU representation of the scene
   */
  [[nodiscard]] __host__ gpu::gpu_scene to_gpu() const;
};

/**
 * @brief Finds the indexes of all objects that are models (meshes).
 * @param [in] scene The scene to search
 * @return A set of indexes in GPU memory
 */
__host__ gpu::gpu_array<size_t> find_model_indexes(const cpu_scene &scene);
}

#endif //CUTRACE_CPU_TYPES_HPP
