//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_GPU_ARRAY_HPP
#define CUTRACE_GPU_ARRAY_HPP

namespace cutrace::gpu {
template <typename T>
struct gpu_arr_it {
  T *current;

  __host__ __device__ inline gpu_arr_it operator++() {
    return { ++current };
  }

  __host__ __device__ constexpr bool operator==(const gpu_arr_it<T> &other) {
    return current == other.current;
  }

  __host__ __device__ constexpr bool operator!=(const gpu_arr_it<T> &other) {
    return current != other.current;
  }

  __host__ __device__ T &operator*() {
    return *current;
  }
};

template <typename T>
struct gpu_array {
  T *buffer;
  size_t size;

  __host__ __device__ constexpr T &operator[](size_t s) {
    return buffer[s];
  }

  __host__ __device__ constexpr const T &operator[](size_t s) const {
    return buffer[s];
  }

  __host__ __device__ constexpr gpu_arr_it<T> begin() { return { buffer }; }
  __host__ __device__ constexpr gpu_arr_it<T> end() { return { buffer + size }; }

  __host__ __device__ constexpr gpu_arr_it<const T> begin() const { return { buffer }; }
  __host__ __device__ constexpr gpu_arr_it<const T> end() const { return { buffer + size }; }
};
}

#endif //CUTRACE_GPU_ARRAY_HPP
