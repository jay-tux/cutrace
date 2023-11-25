//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_VECTOR_HPP
#define CUTRACE_VECTOR_HPP

namespace cutrace {
namespace gpu {
struct vector {
  float x;
  float y;
  float z;

  constexpr __host__ __device__ float operator[](int i) const {
    return i == 0 ? x : i == 1 ? y : z;
  }

  [[nodiscard]] constexpr __host__ __device__ vector cross(const vector &other) const {
    return {
      y * other.z - z * other.y,
      z * other.x - x * other.z,
      x * other.y - y * other.x
    };
  }

  [[nodiscard]] constexpr __host__ __device__ vector normalized() const {
    return (*this) * (1.0f / norm());
  }

  [[nodiscard]] constexpr __host__ __device__ float norm() const {
    return sqrt(x * x + y * y + z * z);
  }

  __host__ __device__ constexpr vector operator+(const vector &v2) const {
    return {x+v2.x, y+v2.y, z+v2.z};
  }

  __host__ __device__ constexpr vector operator-(const vector &v2) const {
    return {x-v2.x, y-v2.y, z-v2.z};
  }

  __host__ __device__ constexpr vector operator*(float f) const {
    return {f*x, f*y, f*z};
  }

  __host__ __device__ constexpr float dot(const vector &other) const {
    return x * other.x + y * other.y + z * other.z;
  }

  __host__ __device__ constexpr vector operator*(const vector &other) const {
    return { x * other.x, y * other.y, z * other.z };
  }

  __host__ __device__ inline vector &operator+=(const vector &other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
  }
};

struct bound {
  vector min, max;

  __host__ __device__ inline bound &merge(const bound &other) {
    min.x = fminf(min.x, other.min.x);
    min.y = fminf(min.y, other.min.y);
    min.z = fminf(min.z, other.min.z);

    max.x = fmaxf(max.x, other.max.x);
    max.y = fmaxf(max.y, other.max.y);
    max.z = fmaxf(max.z, other.max.z);

    return *this;
  }
};

__host__ __device__ constexpr vector operator*(float f, const vector &v) {
  return v * f;
}

__host__ __device__ constexpr vector reflect(const vector &incoming, const vector &normal) {
  return incoming - 2.0f * (normal.dot(incoming)) * normal;
}
}

namespace cpu {
struct vector {
  float x;
  float y;
  float z;

  [[nodiscard]] __host__ constexpr gpu::vector to_gpu() const noexcept {
    return { .x = x, .y = y, .z = z };
  }
};
}
}

#endif //CUTRACE_VECTOR_HPP

