//
// Created by jay on 11/18/23.
//

#ifndef CUTRACE_GPU_VARIANT_HPP
#define CUTRACE_GPU_VARIANT_HPP

#include <concepts>
#include <type_traits>

/**
 * @brief Main namespace for GPU-related code.
 */
namespace cutrace::gpu {
namespace impl {
constexpr const static size_t invalid = (size_t)-1;

template<size_t ... szs>
struct max_of;
template<size_t sz>
struct max_of<sz> {
  constexpr const static size_t value = sz;
};
template<size_t sz1, size_t sz2, size_t ... szs> requires(sz1 < sz2)
struct max_of<sz1, sz2, szs...> {
  constexpr const static size_t value = max_of<sz2, szs...>::value;
};
template<size_t sz1, size_t sz2, size_t ... szs> requires(sz1 > sz2)
struct max_of<sz1, sz2, szs...> {
  constexpr const static size_t value = max_of<sz1, szs...>::value;
};
template <size_t sz1, size_t ... szs>
struct max_of<sz1, sz1, szs...> {
  constexpr const static size_t value = max_of<sz1, szs...>::value;
};

static_assert(max_of<0>::value == 0);
static_assert(max_of<1, 2, 3, 4>::value == 4);
static_assert(max_of<4, 3, 2, 1>::value == 4);
static_assert(max_of<1, 3, 4, 1, 3>::value == 4);
static_assert(max_of<4, 4>::value == 4);

template<typename ... Ts>
struct max_size {
  constexpr const static size_t value = max_of<(sizeof(Ts))...>::value;
};

template<typename T1, typename T2>
concept cvr_equal = std::same_as<
        typename std::remove_cvref<T1>::type,
        typename std::remove_cvref<T2>::type
>;

static_assert(cvr_equal<int, int>);
static_assert(cvr_equal<const int, int>);
static_assert(cvr_equal<int &, int>);
static_assert(cvr_equal<const int &, int>);
static_assert(cvr_equal<int &&, int>);
static_assert(cvr_equal<const int &&, int>);

template<typename T1, typename ... Ts>
struct one_of;
template<typename T1, typename T2>
struct one_of<T1, T2> : std::bool_constant<cvr_equal<T1, T2>> {
};
template<typename T1, typename T2, typename ... Ts>
struct one_of<T1, T2, Ts...>
        : std::bool_constant<cvr_equal<T1, T2> || one_of<T1, Ts...>::value> {
};

static_assert(one_of<int, int, float, bool>::value);
static_assert(one_of<int, float, char, int, bool>::value);
static_assert(!one_of<int, float, char, bool>::value);

template <size_t s> struct invalid_or_incr {
  constexpr const static size_t value = s + 1;
};
template <> struct invalid_or_incr<invalid> {
  constexpr const static size_t value = invalid;
};

static_assert(invalid_or_incr<0>::value == 1);
static_assert(invalid_or_incr<invalid>::value == invalid);
static_assert(invalid_or_incr<23421>::value == 23422);

template<typename T1, typename ... Ts>
struct fst_index_of;
template<typename T1, typename T2> requires(cvr_equal<T1, T2>)
struct fst_index_of<T1, T2> {
  constexpr const static size_t value = 0;
};
template<typename T1, typename T2> requires(!cvr_equal<T1, T2>)
struct fst_index_of<T1, T2> {
  constexpr const static size_t value = invalid;
};

template <typename T1, typename T2, typename ... Ts> requires(cvr_equal<T1, T2>)
struct fst_index_of<T1, T2, Ts...> {
  constexpr const static size_t value = 0;
};

template <typename T1, typename T2, typename ... Ts> requires(!cvr_equal<T1, T2>)
struct fst_index_of<T1, T2, Ts...> {
  constexpr const static size_t value = invalid_or_incr<fst_index_of<T1, Ts...>::value>::value;
};

static_assert(fst_index_of<int, int, char, float>::value == 0);
static_assert(fst_index_of<int, char, int, float>::value == 1);
static_assert(fst_index_of<int, char, float, int>::value == 2);
static_assert(fst_index_of<int, char, float, double>::value == invalid);

template <typename T1, typename ... Ts> struct first {
  using type = T1;
};

template <typename F, typename ... Ts> struct variant_invoke {
  using type = std::invoke_result<F, typename first<Ts...>::type>;
};

template<typename F, typename ... Ts>
concept invocable_with = requires(F &&f, Ts... ts) {
  { f(ts...) };
};
}

/**
 * @brief A simple implementation of `std::variant`, but for GPU.
 * @tparam Ts The types to select from
 */
template <typename ... Ts> requires(sizeof(char) == 1)
class gpu_variant {
private:
  char data[impl::max_size<Ts...>::value] = {}; //!< The internal value
  size_t idx = invalid; //!< Which type is contained
  constexpr static size_t invalid = impl::invalid; //!< What index marks an invalid type

public:
  /**
   * @brief Constructs a default, invalid variant.
   */
  constexpr gpu_variant() = default;

  /**
   * @brief Constructs a variant containing a value of the given type.
   * @tparam T The type of the value
   * @param t The value
   */
  template <typename T> requires(impl::one_of<T, Ts...>::value)
  __host__ __device__ inline gpu_variant(T t) {
    set(t);
  }

  /**
   * @brief Checks if this variant holds a value of type `T`.
   * @tparam T The type to check
   * @return True if this variant holds a value of type `T`, otherwise false
   */
  template <typename T> requires(impl::one_of<T, Ts...>::value)
  __host__ __device__ constexpr bool holds() const {
    return idx == impl::fst_index_of<T, Ts...>::value;
  }

  /**
   * @brief Checks if this variant contains an invalid value.
   * @return True if this variant holds an invalid value, otherwise false
   */
  __host__ __device__ constexpr bool is_invalid() const {
    return idx == impl::invalid;
  }

  /**
   * @brief Stores the given value, overwriting any current value.
   * @tparam T The type of the value to set
   * @param t The value
   */
  template <typename T> requires(impl::one_of<T, Ts...>::value)
  __host__ __device__ inline void set(const T &t) {
    *reinterpret_cast<T *>(data) = t;
    idx = impl::fst_index_of<T, Ts...>::value;
  }

  /**
   * @brief Gets a pointer to the contained value, cast to `T`.
   * @tparam T The type to cast to
   * @return A pointer to the value
   *
   * If this variant does not hold a value of type `T`, the result is undefined.
   */
  template <typename T> requires(impl::one_of<T, Ts...>::value)
  __host__ __device__ inline T *get() {
    return reinterpret_cast<T *>(data);
  }

  /**
   * @brief Gets a constant pointer to the contained value, cast to `T`.
   * @tparam T The type to cast to
   * @return A constant pointer to the value
   *
   * If this variant does not hold a value of type `T`, the result is undefined.
   */
  template <typename T> requires(impl::one_of<T, Ts...>::value)
  __host__ __device__ inline const T *get() const {
    return reinterpret_cast<const T *>(data);
  }
};

namespace impl {
template <typename Fun, typename T1, typename ... Ts>
struct attempt_invoke {
  template <typename ... Us>
  __device__ inline static auto invoke_const(Fun *f, const gpu_variant<Us...> *v) {
    return v->template holds<T1>() ? (*f)(v->template get<T1>()) : attempt_invoke<Fun, Ts...>::template invoke_const<Us...>(f, v);
  }

  template <typename ... Us>
  __device__ inline static auto invoke(Fun *f, gpu_variant<Us...> *v) {
    return v->template holds<T1>() ? (*f)(v->template get<T1>()) : attempt_invoke<Fun, Ts...>::template invoke<Us...>(f, v);
  }
};

template <typename Fun, typename T1>
struct attempt_invoke<Fun, T1> {
  template <typename ... Us>
  __device__ inline static auto invoke_const(Fun *f, const gpu_variant<Us...> *v) {
    return (*f)(v->template get<T1>());
  }

  template <typename ... Us>
  __device__ inline static auto invoke(Fun *f, gpu_variant<Us...> *v) {
    return (*f)(v->template get<T1>());
  }
};
}

/**
 * @brief Invokes the given function on the contained value.
 * @tparam Fun The type of the functions, should be able to accept any of the variant's types
 * @tparam Ts The possible types of the variant
 * @param f The function to call
 * @param v The variant
 * @return The result of the function call
 */
template <typename Fun, typename ... Ts>
__device__ inline auto visit(Fun *f, gpu_variant<Ts...> *v) {
  return impl::attempt_invoke<Fun, Ts...>::invoke(f, v);
}

/**
 * @brief Invokes the given function on the contained value.
 * @tparam Fun The type of the functions, should be able to accept any of the variant's types
 * @tparam Ts The possible types of the variant
 * @param f The function to call
 * @param v The variant
 * @return The result of the function call
 */
template <typename Fun, typename ... Ts>
__device__ inline auto visit(Fun *f, const gpu_variant<Ts...> *v) {
  return impl::attempt_invoke<Fun, Ts...>::invoke_const(f, v);
}
}

#endif //CUTRACE_GPU_VARIANT_HPP
