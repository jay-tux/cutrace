//
// Created by jay on 11/28/23.
//

#ifndef CUTRACE_EITHER_HPP
#define CUTRACE_EITHER_HPP

#include <variant>

namespace cutrace {
template <typename L, typename R> struct either;

namespace impl {
template<typename T>
struct is_either_t : std::false_type {
};
template<typename L, typename R>
struct is_either_t<either<L, R>> : std::true_type {
};

template<typename T>
concept is_either = is_either_t<T>::value;

template<typename T>
struct strip_either_right {
  using type = T;
};
template<typename L, typename R>
struct strip_either_right<either<L, R>> {
  using type = R;
};

template<typename T>
struct strip_either_left {
  using type = T;
};
template<typename L, typename R>
struct strip_either_left<either<L, R>> {
  using type = L;
};

template<typename Fun, typename T>
struct fmap_results {
  using raw = std::invoke_result_t<Fun, T>;
  using stripped_right = strip_either_right<raw>::type;
  using stripped_left = strip_either_left<raw>::type;
};

template<typename Fun, typename T>
using fmap_raw = fmap_results<Fun, T>::raw;

template<typename Fun, typename T>
using fmap_r_t = fmap_results<Fun, T>::stripped_right;

template<typename Fun, typename T>
using fmap_l_t = fmap_results<Fun, T>::stripped_left;

template<typename R, typename E>
concept has_right = is_either<E> && std::same_as<R, typename E::right_t>;

template<typename L, typename E>
concept has_left = is_either<E> && std::same_as<L, typename E::left_t>;

template<typename R, typename T>
concept keeps_right = !is_either<T> || has_right<R, T>;

template<typename L, typename T>
concept keeps_left = !is_either<T> || has_left<L, T>;

template<typename Fun, typename L, typename R>
concept can_fmap_right = requires(Fun &&f, const R &r) {
  { f(r) } -> keeps_left<L>;
};

template<typename Fun, typename L, typename R>
concept can_fmap_left = requires(Fun &&f, const L &r) {
  { f(r) } -> keeps_right<R>;
};
}

/**
 * @brief Tag to indicate no value (void) as left for either.
 */
struct tag_left {
};
/**
 * @brief Tag to indicate no value (void) as right for either.
 */
struct tag_right {
};

namespace impl {
template<typename T>
struct a_void_l {
  using type = T;
};
template<>
struct a_void_l<void> {
  using type = tag_left;
};
template<typename T>
struct a_void_r {
  using type = T;
};
template<>
struct a_void_r<void> {
  using type = tag_right;
};

template<typename T>
concept is_void = std::is_void_v<T>;

template<typename F, typename T>
concept returns_void = requires(F &&f, const T &t) {
  { f(t) } -> is_void<>;
};

template<typename F, typename T>
using safe_invoke_l = typename a_void_l<std::invoke_result_t<F, T>>::type;
template<typename F, typename T>
using safe_invoke_r = typename a_void_r<std::invoke_result_t<F, T>>::type;
}

/**
 * @brief Struct representing an error or success value.
 * @tparam L The left, or error, type
 * @tparam R The right, or success, type
 */
template <typename L, typename R>
struct either {
  using left_t = L; //!< The left, or error, type
  using right_t = R; //!< The right, or success, type

  std::variant<L, R> value; //!< The actual, contained value

  /**
   * @brief Checks if this either is an error.
   * @return True if this either is an error (left), false otherwise
   */
  [[nodiscard]] constexpr bool is_left() const {
    return value.index() == 0;
  }

  /**
   * @brief Checks if this either is a success value.
   * @return True if this either is a success (right), false otherwise
   */
  [[nodiscard]] constexpr bool is_right() const {
    return value.index() == 1;
  }

  /**
   * @brief Gets the left value.
   * @return A reference to the left value
   *
   * Calling this function is undefined if `this->is_left()` returns `false`.
   */
  [[nodiscard]] constexpr L &left() { return std::get<L>(value); }
  /**
   * @brief Gets the right value.
   * @return A reference to the right value
   *
   * Calling this function is undefined if `this->is_right()` returns `false`.
   */
  [[nodiscard]] constexpr R &right() { return std::get<R>(value); }

  /**
   * @brief Gets the left value.
   * @return A const reference to the left value
   *
   * Calling this function is undefined if `this->is_left()` returns `false`.
   */
  [[nodiscard]] constexpr const L &left() const { return std::get<L>(value); }
  /**
   * @brief Gets the right value.
   * @return A const reference to the right value
   *
   * Calling this function is undefined if `this->is_right()` returns `false`.
   */
  [[nodiscard]] constexpr const R &right() const { return std::get<R>(value); }

  /**
   * @brief If this either is a success value, calls the given function, wrapping the result in an either again.
   * @tparam Fun The type of the function to call
   * @param f The function to call
   * @return The wrapped result of the function
   *
   * If `this->is_left()` is true, then the function is not invoked, and the error value in this either is wrapped and
   * returned.
   */
  template <typename Fun> requires(std::invocable<Fun, const R &>)
  constexpr auto map(Fun &&f) -> either<L, impl::safe_invoke_r<Fun, const R &>> {
    using ret_t = either<L, impl::safe_invoke_r<Fun, const R &>>;
    if(is_left()) return ret_t::left(left());
    else {
      if constexpr(impl::returns_void<Fun, R>) {
        f(right());
        return ret_t::right(tag_right{});
      }
      else {
        return ret_t::right(f(right()));
      }
    }
  }

  /**
   * @brief If this either is an error value, calls the given function, wrapping the result in an either again.
   * @tparam Fun The type of the function to call
   * @param f The function to call
   * @return The wrapped result of the function
   *
   * If `this->is_right()` is true, then the function is not invoked, and the success value in this either is wrapped
   * and returned.
   */
  template <typename Fun> requires(std::invocable<Fun, const L &>)
  constexpr auto map_left(Fun &&f) -> either<impl::safe_invoke_l<Fun, const L &>, R> {
    using ret_t = either<impl::safe_invoke_l<Fun, const L &>, R>;
    if(is_right()) return ret_t::right(right());
    else {
      if constexpr(impl::returns_void<Fun, L>) {
        f(left());
        return ret_t::left(tag_left{});
      }
      else {
        return ret_t::left(f(left()));
      }
    }
  }

  /**
   * @brief If this either holds a success value, calls the given function on it and unwraps the result.
   * @tparam Fun The type of the function to call
   * @param f The function to call
   * @return The result of the function
   *
   * If `this->is_left()` is true, the function isn't called, and the `this->left()` is re-wrapped and returned.
   *
   * If `f` returns an `either<L, R2>`, then the result will be `either<L, R2>` (the function's result is unwrapped).
   * If `f` returns an `R2`, then the result will be `either<L, R2>`.
   */
  template <typename Fun> requires(impl::can_fmap_right<Fun, L, R> && !std::is_void_v<std::invoke_result_t<Fun, const R &>>)
  constexpr auto fmap(Fun &&f) const -> either<L, impl::fmap_r_t<Fun, const R &>> {
    if(std::holds_alternative<L>(value)) return { .value = {std::get<L>(value)} };
    else {
      if constexpr(impl::is_either<impl::fmap_raw<Fun, const R &>>) return f(std::get<R>(value));
      else return { .value = f(std::get<R>(value)) };
    }
  }

  /**
   * @brief If this either holds an error value, calls the given function on it and unwraps the result.
   * @tparam Fun The type of the function to call
   * @param f The function to call
   * @return The result of the function
   *
   * If `this->is_right()` is true, the function isn't called, and the `this->right()` is re-wrapped and returned.
   *
   * If `f` returns an `either<L2, R>`, then the result will be `either<L2, R>` (the function's result is unwrapped).
   * If `f` returns an `L2`, then the result will be `either<L2, R>`.
   */
  template <typename Fun> requires(impl::can_fmap_left<Fun, L, R> && !std::is_void_v<std::invoke_result_t<Fun, const L &>>)
  constexpr auto fmap_left(Fun &&f) const -> either<impl::fmap_l_t<Fun, const L &>, R> {
    if(std::holds_alternative<R>(value)) return { .value = {std::get<R>(value)} };
    else {
      if constexpr(impl::is_either<impl::fmap_raw<Fun, const L &>>) return f(std::get<L>(value));
      else return { .value = f(std::get<L>(value)) };
    }
  }

  /**
   * @brief Folds this value.
   * @tparam Fun1 The type of the function to invoke if `this->is_left()` is true
   * @tparam Fun2 The type of the function to invoke if `this->is_right()` is true
   * @param f_left The function to invoke if `this->is_left()` is true
   * @param f_right The function to invoke if `this->is_right()` is true
   * @return The result of invoking the required function
   *
   * If this either holds an error value, then the first function (`f_left`) is invoked on it, and the result is
   * returned.
   * If this either holds a success value, however, then the other function (`f_right`) is invoked on `this->right()`
   * instead.
   *
   * Both functions should return a value of the same type.
   */
  template <typename Fun1, typename Fun2>
  requires(std::same_as<std::invoke_result_t<Fun1, const L &>, std::invoke_result_t<Fun2, const R &>>
           && !std::is_void_v<std::invoke_result_t<Fun1, const L &>> && !std::is_void_v<std::invoke_result_t<Fun2, const R &>>)
  constexpr auto fold(Fun1 &&f_left, Fun2 &&f_right) const -> std::invoke_result_t<Fun1, const L &> {
    return is_left() ? f_left(left()) : f_right(right());
  }

  /**
   * @brief Constructs a new either from a left (error) value.
   * @param l The left (error) value
   * @return The new either
   */
  static constexpr either<L, R> left(const L &l) { return either<L, R>{ l }; }
  /**
   * "@brief Constructs a new either from a right (success) value.
   * @param l The right (success) value
   * @return The new either
   */
  static constexpr either<L, R> right(const R &r) { return either<L, R>{ r }; }

  /**
   * @brief Constructs a value of type `T` from `this->right()` if this either holds a success value.
   * @tparam T The type to construct
   * @return The re-wrapped either
   */
  template <typename T> requires(std::constructible_from<T, const R &>)
  constexpr either<L, T> re_wrap() const {
    if(is_left()) return either<L, T>::left(left());
    return either<L, T>::right(T{right()});
  }
};

namespace impl {
template<typename L, typename ... Ts>
struct fmap_all_helper;

template<typename L, typename R>
struct fmap_all_helper<either<L, R>> {
  constexpr static bool any_left(const either<L, R> &e) { return e.is_left(); }

  constexpr static const L &first_left(const either<L, R> &e) { return e.left(); }
};

template<typename L, typename R1, typename R2, typename ... Rs>
struct fmap_all_helper<either<L, R1>, either<L, R2>, either<L, Rs>...> {
  constexpr static bool any_left(const either<L, R1> &e1, const either<L, R2> &e2, const either<L, Rs> &... rest) {
    return e1.is_left() || fmap_all_helper<either<L, R2>, either<L, Rs>...>::any_left(e2, rest...);
  }

  constexpr static const L &
  first_left(const either<L, R1> &e1, const either<L, R2> &e2, const either<L, Rs> &... rest) {
    return (e1.is_left()) ? e1.left() : fmap_all_helper<either<L, R2>, either<L, Rs>...>::first_left(e2, rest...);
  }
};


template<typename L, typename T>
struct maybe_wrap {
  using type = either<L, T>;
};
template<typename L, typename L2, typename R>
struct maybe_wrap<L, either<L2, R>> {
  using type = either<L2, R>;
};
template<typename L, typename T> using maybe_wrap_t = typename maybe_wrap<L, T>::type;
}

/**
 * If all given either values are right (success) values, calls the given function on all right (success) values.
 * @tparam Fun The type of the function
 * @tparam L The left (error) type
 * @tparam Ts The types of all either values
 * @param f The function to call
 * @param es The either values
 * @return The resulting value
 *
 * If any of the given either values are a left (error) value, the first of the left (error) values is returned.
 *
 * If all of the given either values are right (success) values, then the result of `f(es.right()...)` (`f` with as
 * arguments all right/success values) is returned.
 */
template <typename Fun, typename L, typename ... Ts>
constexpr auto fmap_all(Fun &&f, const either<L, Ts> &... es) {
  using f_ret = std::invoke_result_t<Fun, const Ts &...>;
  using ret_t = impl::maybe_wrap_t<L, f_ret>;
  using helper_t = impl::fmap_all_helper<either<L, Ts>...>;
  if(helper_t::any_left(es...))
    return ret_t{ helper_t::first_left(es...) };

  if constexpr(std::same_as<f_ret, ret_t>)
    return f(es.right()...);
  else
    return ret_t{ f(es.right()...) };
}
}

#endif //CUTRACE_EITHER_HPP
