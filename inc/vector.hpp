//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_VECTOR_HPP
#define CUTRACE_VECTOR_HPP

#include <cmath>

/**
 * @brief Namespace containing all of cutrace’s code.
 */
namespace cutrace {
/**
 * @brief Struct representing a 3D-vector on GPU (point, direction, or color).
 */
struct vector {
  float x; //!< The X-component of the vector
  float y; //!< The Y-component of the vector
  float z; //!< The Z-component of the vector

  /**
   * @brief Gets the `i`-th component of the vector.
   * @param i The index of the component
   * @return The component's value
   *
   * X corresponds to index 0, Y to index 1, Z to index 2. Any other index is undefined behavior (but will result in Z).
   */
  constexpr __host__ __device__ float operator[](int i) const {
    return i == 0 ? x : i == 1 ? y : z;
  }

  [[nodiscard]] constexpr inline vector to_gpu() const {
    return *this;
  }

  /**
   * @brief Computes the cross product with another vector.
   * @param [in] other The other vector for the cross product
   * @return \f$(*this) \times other\f$
   */
  [[nodiscard]] constexpr __host__ __device__ vector cross(const vector &other) const {
    return {
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
    };
  }

  /**
   * @brief Normalizes a vector.
   * @return \f$\frac{(\star this)}{||\star this||}\f$
   */
  [[nodiscard]] constexpr __host__ __device__ vector normalized() const {
    return (*this) * (1.0f / norm());
  }

  /**
   * @brief Computes the Euclidean norm of this vector.
   * @return \f$||\star this||\f$
   */
  [[nodiscard]] constexpr __host__ __device__ float norm() const {
#ifdef __CUDA_ARCH__
    return sqrt(x * x + y * y + z * z);
#else
//    return std::sqrtf(x * x + y * y + z * z);
    return sqrtf(x * x + y * y + z * z);
#endif
  }

  /**
   * @brief Computes the (component-wise) sum of this vector with v2.
   * @param [in] v2 The other vector
   * @return \f$(\star this) + v2\f$
   */
  __host__ __device__ constexpr vector operator+(const vector &v2) const {
    return {x + v2.x, y + v2.y, z + v2.z};
  }

  /**
   * @brief Computes the (component-wise) difference of this vector with v2.
   * @param [in] v2 The other vector
   * @return \f$(\star this) - v2\f$
   */
  __host__ __device__ constexpr vector operator-(const vector &v2) const {
    return {x - v2.x, y - v2.y, z - v2.z};
  }

  /**
   * @brief Computes the scalar product of this vector with f.
   * @param [in] f The scaling factor
   * @return \f$f * (\star this)\f$
   */
  __host__ __device__ constexpr vector operator*(float f) const {
    return {f * x, f * y, f * z};
  }

  /**
   * @brief Computes the dot product between this vector and other.
   * @param [in] other The other vector
   * @return \f$(\star this) \cdot other\f$
   */
  __host__ __device__ constexpr float dot(const vector &other) const {
    return x * other.x + y * other.y + z * other.z;
  }

  /**
   * @brief Computes the component-wise product between this vector and other.
   * @param [in] other The other vector
   * @return \f$\begin{pmatrix}this\to x * other.x \\ this\to y * other.y \\ this\to z * other.z \end{pmatrix}\f$
   */
  __host__ __device__ constexpr vector operator*(const vector &other) const {
    return {x * other.x, y * other.y, z * other.z};
  }

  /**
   * @brief Add other to this vector
   * @param [in] other The vector to add
   * @return A reference to `*this`
   */
  __host__ __device__ inline vector &operator+=(const vector &other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }
};

/**
 * @brief Struct representing an AABB (axis-aligned bounding box).
 */
struct bound {
  vector min, //!< The minimal point of the AABB
         max; //!< The maximal point of the AABB

  /**
   * Merges this AABB with another
   * @param [in] other The other AABB
   * @return A reference to `*this`
   */
  __host__ __device__ inline bound &merge(const bound &other) {
    min.x = fminf(min.x, other.min.x);
    min.y = fminf(min.y, other.min.y);
    min.z = fminf(min.z, other.min.z);

    max.x = fmaxf(max.x, other.max.x);
    max.y = fmaxf(max.y, other.max.y);
    max.z = fmaxf(max.z, other.max.z);

    return *this;
  }

  __host__ __device__ constexpr static bound incorrect() {
    return {
            { INFINITY, INFINITY, INFINITY },
            { -INFINITY, -INFINITY, -INFINITY }
    };
  }
};

/**
 * @brief Scales `v` by a factor `f`.
 * @param [in] f The scaling factor
 * @param [in] v The vector
 * @return The scaled vector
 */
__host__ __device__ constexpr vector operator*(float f, const vector &v) {
  return v * f;
}

/**
 * @brief Computes the result of reflecting the incoming vector from a surface defined by its normal.
 * @param [in] incoming The incoming vector
 * @param [in] normal The normal of the surface
 * @return \f$incoming - 2 * (normal \cdot incoming) * normal\f$
 */
__host__ __device__ constexpr vector reflect(const vector &incoming, const vector &normal) {
  return incoming - 2.0f * (normal.dot(incoming)) * normal;
}

struct matrix {
  vector columns[3];

  __device__ constexpr float determinant() {
    float a = columns[0].x, b = columns[1].x, c = columns[2].x,
            d = columns[0].y, e = columns[1].y, f = columns[2].y,
            g = columns[0].z, h = columns[1].z, i = columns[2].z;

    return a*e*i + b*f*g + c*d*h - c*e*g - a*f*h - b*d*i;
  }
};
}

#endif //CUTRACE_VECTOR_HPP

