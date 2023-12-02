//
// Created by jay on 12/2/23.
//

#ifndef CUTRACE_GRID_HPP
#define CUTRACE_GRID_HPP

#include <cstring>

namespace cutrace {
template <typename T>
class grid {
public:
  template <typename T1>
  class row_it {
  public:
    constexpr row_it(T1 *elem) : elem{elem} {}

    constexpr T1 &operator*() noexcept { return *elem; }
    constexpr row_it<T1> operator++() noexcept { return row_it<T1>(elem++); }

    constexpr bool operator!=(const row_it<T1> &other) { return elem != other.elem; }

    T1 *elem;
  };

  template <typename T1 = T>
  class row {
  public:
    constexpr T1 &operator[](size_t i) noexcept requires(!std::is_const_v<T1>) { return ptr[i]; }
    constexpr const T1 &operator[](size_t i) const noexcept { return ptr[i]; }

    constexpr row_it<T1> begin() { return row_it<T1>(ptr); }
    constexpr row_it<T1> end() { return row_it<T1>(ptr + w); }
    constexpr row_it<const T1> begin() const { return row_it<const T1>(ptr); }
    constexpr row_it<const T1> end() const { return row_it<const T1>(ptr + w); }

  private:
    constexpr row(T1 *ptr, size_t w) : ptr{ptr}, w{w} {}

    T1 *ptr;
    size_t w;

    friend grid;
    friend class grid_it;
  };

  template <typename T1>
  class grid_it {
  public:
    constexpr grid_it(T1 *ptr, size_t w) : ptr{ptr}, w{w} {}
    constexpr row<T1> operator*() noexcept { return row<T1>(ptr, w); }

    constexpr grid_it<T1> operator++() noexcept {
      return grid_it<T1>(ptr + w, w);
    }
    constexpr bool operator!=(const grid_it<T1> &other) const noexcept { return ptr != other.ptr; }

    T1 *ptr;
    size_t w;
  };

  constexpr grid() = default;

  inline grid(size_t w, size_t h) requires(std::default_initializable<T>) : w{w}, h{h} {
    buffer = new T[w * h];
    for(size_t i = 0; i < w * h; i++) buffer[i] = T{};
  }

  inline grid(size_t w, size_t h, const T &init) : w{w}, h{h} {
    buffer = new T[w * h];
    for(size_t i = 0; i < w * h; i++) buffer[i] = init;
  }

  inline grid(const grid<T> &other) { *this = other; }
  inline grid(grid<T> &&other) noexcept { *this = std::move(other); }

  inline grid<T> &operator=(const grid<T> &other) {
    if(this == &other) return *this;

    delete [] buffer;
    w = other.w;
    h = other.h;

    buffer = new T[w * h];
    for(size_t i = 0; i < w * h; i++) buffer[i] = other.buffer[i];

    return *this;
  }

  inline grid<T> &operator=(grid<T> &&other) noexcept {
    std::swap(w, other.w);
    std::swap(h, other.h);
    std::swap(buffer, other.buffer);
    return *this;
  }

  constexpr row<> operator[](size_t i) noexcept {
    return row(buffer + i * w, w);
  }

  constexpr grid_it<T> begin() { return grid_it<T>(buffer, w); }
  constexpr grid_it<T> end() { return grid_it<T>(buffer + w * h, w); }
  constexpr grid_it<const T> begin() const { return grid_it<const T>(buffer, w); }
  constexpr grid_it<const T> end() const { return grid_it<const T>(buffer + w * h, w); }

  constexpr T *data() { return buffer; }
  constexpr T *data(size_t i) { return buffer + w * i; }

  inline grid resize(size_t new_w, size_t new_h) requires(std::default_initializable<T>) {
    *this = std::move(grid<T>(new_w, new_h));
  }

  [[nodiscard]] constexpr size_t cols() const { return w; }
  [[nodiscard]] constexpr size_t rows() const { return h; }
  [[nodiscard]] constexpr size_t elems() const { return w * h; }

  constexpr T &raw(size_t i) { return buffer[i]; }
  constexpr const T&raw(size_t i) const { return buffer[i]; }

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
