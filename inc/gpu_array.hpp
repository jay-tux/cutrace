//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_GPU_ARRAY_HPP
#define CUTRACE_GPU_ARRAY_HPP

/**
 * @brief Main namespace for GPU-related code.
 */
namespace cutrace::gpu {
/**
 * @brief Structure representing an iterator into a \ref cutrace::gpu::gpu_array.
 * @tparam T The type of objects in the array
 */
template <typename T>
struct gpu_arr_it {
  T *current; //!< A pointer to the current element

  /**
   * @brief Moves to the next element.
   * @return An iterator to the next element
   *
   * This modifies the iterator itself too.
   */
  __host__ __device__ inline gpu_arr_it operator++() {
    return { ++current };
  }

  /**
   * @brief Compares this iterator with another for equality.
   * @param [in] other The iterator to compare to
   * @return True if both iterators point to the same element, otherwise false
   */
  __host__ __device__ constexpr bool operator==(const gpu_arr_it<T> &other) {
    return current == other.current;
  }

  /**
   * @brief Compares this iterator with another for inequality.
   * @param [in] other The iterator to compare to
   * @return True if both iterators point to another element, otherwise false
   */
  __host__ __device__ constexpr bool operator!=(const gpu_arr_it<T> &other) {
    return current != other.current;
  }

  /**
   * @brief Dereferences this iterator.
   * @return A reference to the element this iterator points to
   */
  __host__ __device__ T &operator*() {
    return *current;
  }
};

/**
 * @brief Structure representing a sized array in GPU memory.
 * @tparam T The type of objects contained in the array
 */
template <typename T>
struct gpu_array {
  T *buffer; //!< The buffer with objects
  size_t size; //!< The amount of objects in the buffer

  /**
   * @brief Gets the `s`-th object in the array.
   * @param [in] s The index of the object
   * @return A constant reference to the object at index `s`
   *
   * As there is no bounds-checking in this function, reading out-of-bounds is undefined behavior.
   */
  __host__ __device__ constexpr T &operator[](size_t s) {
    return buffer[s];
  }

  /**
   * @brief Gets the `s`-th object in the array.
   * @param [in] s The index of the object
   * @return A reference to the object at index `s`
   *
   * As there is no bounds-checking in this function, reading or writing out-of-bounds is undefined behavior.
   */
  __host__ __device__ constexpr const T &operator[](size_t s) const {
    return buffer[s];
  }

  /**
   * @brief An iterator to the beginning of the buffer.
   * @return The iterator
   */
  __host__ __device__ constexpr gpu_arr_it<T> begin() { return { buffer }; }
  /**
   * @brief An iterator past the end of the buffer.
   * @return The iterator
   */
  __host__ __device__ constexpr gpu_arr_it<T> end() { return { buffer + size }; }

  /**
   * @brief A constant iterator to the beginning of the buffer.
   * @return The iterator
   */
  __host__ __device__ constexpr gpu_arr_it<const T> begin() const { return { buffer }; }
  /**
   * @brief A constant iterator past the end of the buffer.
   * @return The iterator
   */
  __host__ __device__ constexpr gpu_arr_it<const T> end() const { return { buffer + size }; }
};
}

#endif //CUTRACE_GPU_ARRAY_HPP
