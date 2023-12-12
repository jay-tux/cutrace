//
// Created by jay on 12/2/23.
//

#ifndef CUTRACE_GRID_HPP
#define CUTRACE_GRID_HPP

#include <cstring>

/**
 * @brief Namespace containing all of cutrace’s code.
 */
namespace cutrace {
/**
 * @brief Class representing a 2D grid (a rectangular array).
 * @tparam T The type of values contained within the grid
 *
 * This class uses a single buffer, so data is guaranteed to be laid out contiguously in row-major order.
 */
template <typename T>
class grid {
public:
  /**
   * Class representing an iterator over a row in the grid.
   * @tparam T1 The type of values used in the iterator
   *
   * Note: the type used by the iterator can be a `const` version of the type used by the grid itself.
   *
   * @see cutrace::grid
   */
  template <typename T1>
  class row_it {
  public:
    /**
     * Constructs a new iterator from a pointer.
     * @param elem The current element
     */
    constexpr row_it(T1 *elem) : elem{elem} {}

    /**
     * Deferences the iterator, getting the current element.
     * @return A (possibly const) reference to the current element
     */
    constexpr T1 &operator*() noexcept { return *elem; }
    /**
     * Increments the iterator, moving to the next element.
     * @return The incremented version of this iterator
     */
    constexpr row_it<T1> operator++() noexcept { return row_it<T1>(++elem); }
    /**
     * Compares two iterators for inequality.
     * @param other The other iterator
     * @return True if both elements point to a different element, otherwise false
     */
    constexpr bool operator!=(const row_it<T1> &other) { return elem != other.elem; }

    T1 *elem; //!< A pointer to the current element.
  };

  /**
   * Constructs a row-view over a grid.
   * @tparam T1 The type of values used in the row
   *
   * Note: the type used by the iterator can be a `const` version of the type used by the grid itself.
   *
   * @see cutrace::grid
   */
  template <typename T1 = T>
  class row {
  public:
    /**
     * Gets an element from the row by index.
     * @param i The index of the element
     * @return A reference to the element
     */
    constexpr T1 &operator[](size_t i) noexcept requires(!std::is_const_v<T1>) { return ptr[i]; }
    /**
     * Gets an element from the row by index.
     * @param i The index of the element
     * @return A const reference to the element
     */
    constexpr const T1 &operator[](size_t i) const noexcept { return ptr[i]; }

    /**
     * Gets an iterator to the first element of the row.
     * @return The iterator
     */
    constexpr row_it<T1> begin() { return row_it<T1>(ptr); }
    /**
     * Gets an iterator past the last element of the row.
     * @return The iterator
     */
    constexpr row_it<T1> end() { return row_it<T1>(ptr + w); }
    /**
     * Gets a const iterator to the first element of the row.
     * @return The iterator
     */
    constexpr row_it<const T1> begin() const { return row_it<const T1>(ptr); }
    /**
     * Gets a const iterator past the last element of the row.
     * @return The iterator
     */
    constexpr row_it<const T1> end() const { return row_it<const T1>(ptr + w); }

  private:
    constexpr row(T1 *ptr, size_t w) : ptr{ptr}, w{w} {}

    T1 *ptr;
    size_t w;

    friend grid;
    friend class grid_it;
  };

  /**
   * Class representing a row-based iterator over a grid.
   * @tparam T1 The type of values used in the iterator
   *
   * Note: the type used by the iterator can be a `const` version of the type used by the grid itself.
   *
   * @see cutrace::grid
   */
  template <typename T1>
  class grid_it {
  public:
    /**
     * Constructs a new row-based iterator using a pointer and size.
     * @param ptr The pointer
     * @param w The size of the row
     */
    constexpr grid_it(T1 *ptr, size_t w) : ptr{ptr}, w{w} {}
    /**
     * Gets the row this iterator points to.
     * @return The current row
     */
    constexpr row<T1> operator*() noexcept { return row<T1>(ptr, w); }
    /**
     * Increments this iterator, making it point to the next row.
     * @return The incremented iterator
     */
    constexpr grid_it<T1> operator++() noexcept {
      ptr += w;
      return grid_it<T1>(ptr, w);
    }
    /**
     * Compares two iterators for inequality.
     * @param other The other iterator
     * @return True if both iterators point to a different row, false otherwise.
     */
    constexpr bool operator!=(const grid_it<T1> &other) const noexcept { return ptr != other.ptr; }

    T1 *ptr; //!< Pointer to the first element of the current row.
    size_t w; //!< Size of this row.
  };

  /**
   * Constructs a new 0x0 grid.
   */
  constexpr grid() = default;

  /**
   * Constructs a new grid of the specified size, filled with default values for `T`.
   * @param w The width of the grid (i.e. the amount of elements per row)
   * @param h The height of the grid (i.e. the amount of rows)
   */
  inline grid(size_t w, size_t h) requires(std::default_initializable<T>) : w{w}, h{h} {
    buffer = new T[w * h];
    for(size_t i = 0; i < w * h; i++) buffer[i] = T{};
  }

  /**
   * Constructs a new grid of the specified size, filled with copies of the specified value.
   * @param w The width of the grid (i.e. the amount of elements per row)
   * @param h The height of the grid (i.e. the amount of rows)
   * @param init The element to copy
   */
  inline grid(size_t w, size_t h, const T &init) : w{w}, h{h} {
    buffer = new T[w * h];
    for(size_t i = 0; i < w * h; i++) buffer[i] = init;
  }

  /**
   * Constructs a new grid, copying all data from the other grid.
   * @param other The grid to copy
   */
  inline grid(const grid<T> &other) { *this = other; }
  /**
   * Constructs a new grid, moving all data from the other grid.
   * @param other The grid to move from
   */
  inline grid(grid<T> &&other) noexcept { *this = std::move(other); }

  /**
   * Removes all data in this grid, replacing it with the data in the other grid (by copy)
   * @param other The grid to copy
   * @return A reference to this grid
   */
  inline grid<T> &operator=(const grid<T> &other) {
    if(this == &other) return *this;

    delete [] buffer;
    w = other.w;
    h = other.h;

    buffer = new T[w * h];
    for(size_t i = 0; i < w * h; i++) buffer[i] = other.buffer[i];

    return *this;
  }

  /**
   * Swaps the data between this grid and the other.
   * @param other The grid to swap with
   * @return A reference to this grid
   */
  inline grid<T> &operator=(grid<T> &&other) noexcept {
    std::swap(w, other.w);
    std::swap(h, other.h);
    std::swap(buffer, other.buffer);
    return *this;
  }

  /**
   * Gets a row from the grid by index.
   * @param i The index of the row
   * @return A row-view for the given index
   */
  constexpr row<> operator[](size_t i) noexcept {
    return row(buffer + i * w, w);
  }

  /**
   * Gets an iterator to the first row in the grid.
   * @return The iterator
   */
  constexpr grid_it<T> begin() { return grid_it<T>(buffer, w); }
  /**
   * Gets an iterator past the last row in the grid.
   * @return The iterator
   */
  constexpr grid_it<T> end() { return grid_it<T>(buffer + w * h, w); }
  /**
   * Gets a const iterator to the first row in the grid.
   * @return The iterator
   */
  constexpr grid_it<const T> begin() const { return grid_it<const T>(buffer, w); }
  /**
   * Gets a const iterator past the last row in the grid.
   * @return The iterator
   */
  constexpr grid_it<const T> end() const { return grid_it<const T>(buffer + w * h, w); }

  /**
   * Gets a raw pointer to the data.
   * @return The pointer
   *
   * Data is laid out in memory in a row-major order using a contiguous buffer.
   */
  constexpr T *data() { return buffer; }
  /**
   * Gets a raw pointer to the data for the given row.
   * @param i The index of the row
   * @return The pointer
   *
   * Data is laid out in memory in a row-major order using a contiguous buffer.
   */
  constexpr T *data(size_t i) { return buffer + w * i; }

  /**
   * Resizes this grid.
   * @param new_w The new width (amount of elements per row)
   * @param new_h The new height (amount of rows)
   *
   * All data in this grid will be overwritten with the default value for `T`.
   */
  inline void resize(size_t new_w, size_t new_h) requires(std::default_initializable<T>) {
    *this = std::move(grid<T>(new_w, new_h));
  }

  /**
   * Gets the amount of columns in the grid.
   * @return The amount of columns
   *
   * This is equal to the width of the grid.
   */
  [[nodiscard]] constexpr size_t cols() const { return w; }
  /**
   * Gets the amount of rows in the grid.
   * @return The amount of rows
   *
   * This is equal to the height of the grid.
   */
  [[nodiscard]] constexpr size_t rows() const { return h; }
  /**
   * Gets the amount of elements in the grid.
   * @return The amount of elements
   *
   * This is equal to the width of the grid multiplied by its height.
   */
  [[nodiscard]] constexpr size_t elems() const { return w * h; }

  /**
   * Gets an element by raw index.
   * @param i The raw index
   * @return A reference to the element
   *
   * Since the grid is laid out in memory as a contiguous array, you can use `y * width + x` to get the `(x, y)`-th element.
   */
  constexpr T &raw(size_t i) { return buffer[i]; }
  /**
   * Gets an element by raw index.
   * @param i The raw index
   * @return A const reference to the element
   *
   * Since the grid is laid out in memory as a contiguous array, you can use `y * width + x` to get the `(x, y)`-th element.
   */
  constexpr const T&raw(size_t i) const { return buffer[i]; }

  /**
   * Deletes the grid, cleaning up all resources.
   */
  inline ~grid() noexcept {
    delete [] buffer;
    w = 0;
    h = 0;
    buffer = nullptr;
  }

private:
  T *buffer = nullptr;
  size_t w = 0, h = 0;
};
}

#endif //CUTRACE_GRID_HPP
