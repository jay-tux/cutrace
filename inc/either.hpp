//
// Created by jay on 11/28/23.
//

#ifndef CUTRACE_EITHER_HPP
#define CUTRACE_EITHER_HPP

#include <variant>

namespace cutrace {
template <typename L, typename R> struct either;

template <typename T> struct is_either_t : std::false_type {};
template <typename L, typename R> struct is_either_t<either<L, R>> : std::true_type {};

template <typename T>
concept is_either = is_either_t<T>::value;

template <typename T> struct strip_either_right { using type = T; };
template <typename L, typename R> struct strip_either_right<either<L, R>> { using type = R; };

template <typename T> struct strip_either_left { using type = T; };
template <typename L, typename R> struct strip_either_left<either<L, R>> { using type = L; };

template <typename Fun, typename T>
struct fmap_results {
  using raw = std::invoke_result_t<Fun, T>;
  using stripped_right = strip_either_right<raw>::type;
  using stripped_left = strip_either_left<raw>::type;
};

template <typename Fun, typename T>
using fmap_raw = fmap_results<Fun, T>::raw;

template <typename Fun, typename T>
using fmap_r_t = fmap_results<Fun, T>::stripped_right;

template <typename Fun, typename T>
using fmap_l_t = fmap_results<Fun, T>::stripped_left;

template <typename R, typename E>
concept has_right = is_either<E> && std::same_as<R, typename E::right_t>;

template <typename L, typename E>
concept has_left = is_either<E> && std::same_as<L, typename E::left_t>;

template <typename R, typename T>
concept keeps_right = !is_either<T> || has_right<R, T>;

template <typename L, typename T>
concept keeps_left = !is_either<T> || has_left<L, T>;

template <typename Fun, typename L, typename R>
concept can_fmap_right = requires(Fun &&f, const R &r) {
  { f(r) } -> keeps_left<L>;
};

template <typename Fun, typename L, typename R>
concept can_fmap_left = requires(Fun &&f, const L &r) {
  { f(r) } -> keeps_right<R>;
};

struct tag_left{};
struct tag_right{};

template <typename T> struct a_void_l { using type = T; };
template <> struct a_void_l<void> { using type = tag_left; };
template <typename T> struct a_void_r { using type = T; };
template <> struct a_void_r<void> { using type = tag_right; };

template<typename T>
concept is_void = std::is_void_v<T>;

template <typename F, typename T>
concept returns_void = requires(F &&f, const T &t) {
  { f(t) } -> is_void<>;
};

template <typename F, typename T>
using safe_invoke_l = typename a_void_l<std::invoke_result_t<F, T>>::type;
template <typename F, typename T>
using safe_invoke_r = typename a_void_r<std::invoke_result_t<F, T>>::type;

template <typename L, typename R>
struct either {
  using left_t = L;
  using right_t = R;

  std::variant<L, R> value;

  [[nodiscard]] constexpr bool is_left() const {
    return value.index() == 0;
  }

  [[nodiscard]] constexpr bool is_right() const {
    return value.index() == 1;
  }

  [[nodiscard]] constexpr L &left() { return std::get<L>(value); }
  [[nodiscard]] constexpr R &right() { return std::get<R>(value); }

  [[nodiscard]] constexpr const L &left() const { return std::get<L>(value); }
  [[nodiscard]] constexpr const R &right() const { return std::get<R>(value); }

  template <typename Fun> requires(std::invocable<Fun, const R &>)
  constexpr auto map(Fun &&f) -> either<L, safe_invoke_r<Fun, const R &>> {
    using ret_t = either<L, safe_invoke_r<Fun, const R &>>;
    if(is_left()) return ret_t::left(left());
    else {
      if constexpr(returns_void<Fun, R>) {
        f(right());
        return ret_t::right(tag_right{});
      }
      else {
        return ret_t::right(f(right()));
      }
    }
  }

  template <typename Fun> requires(std::invocable<Fun, const L &>)
  constexpr auto map_left(Fun &&f) -> either<safe_invoke_l<Fun, const L &>, R> {
    using ret_t = either<safe_invoke_l<Fun, const L &>, R>;
    if(is_right()) return ret_t::right(right());
    else {
      if constexpr(returns_void<Fun, L>) {
        f(left());
        return ret_t::left(tag_left{});
      }
      else {
        return ret_t::left(f(left()));
      }
    }
  }

  template <typename Fun> requires(can_fmap_right<Fun, L, R> && !std::is_void_v<std::invoke_result_t<Fun, const R &>>)
  constexpr auto fmap(Fun &&f) const -> either<L, fmap_r_t<Fun, const R &>> {
    if(std::holds_alternative<L>(value)) return { .value = {std::get<L>(value)} };
    else {
      if constexpr(is_either<fmap_raw<Fun, const R &>>) return f(std::get<R>(value));
      else return { .value = f(std::get<R>(value)) };
    }
  }

  template <typename Fun> requires(can_fmap_left<Fun, L, R> && !std::is_void_v<std::invoke_result_t<Fun, const L &>>)
  constexpr auto fmap_left(Fun &&f) const -> either<fmap_l_t<Fun, const L &>, R> {
    if(std::holds_alternative<R>(value)) return { .value = {std::get<R>(value)} };
    else {
      if constexpr(is_either<fmap_raw<Fun, const L &>>) return f(std::get<L>(value));
      else return { .value = f(std::get<L>(value)) };
    }
  }

  template <typename Fun1, typename Fun2>
  requires(std::same_as<std::invoke_result_t<Fun1, const L &>, std::invoke_result_t<Fun2, const R &>>
           && !std::is_void_v<std::invoke_result_t<Fun1, const L &>> && !std::is_void_v<std::invoke_result_t<Fun2, const R &>>)
  constexpr auto fold(Fun1 &&f_left, Fun2 &&f_right) const -> std::invoke_result_t<Fun1, const L &> {
    return is_left() ? f_left(left()) : f_right(right());
  }

  static constexpr either<L, R> left(const L &l) { return either<L, R>{ l }; }
  static constexpr either<L, R> right(const R &r) { return either<L, R>{ r }; }

  template <typename T> requires(std::constructible_from<T, const R &>)
  constexpr either<L, T> re_wrap() const {
    if(is_left()) return either<L, T>::left(left());
    return either<L, T>::right(T{right()});
  }
};

template <typename L, typename ... Ts> struct fmap_all_helper;

template <typename L, typename R>
struct fmap_all_helper<either<L, R>> {
  constexpr static bool any_left(const either<L, R> &e) { return e.is_left(); }
  constexpr static const L &first_left(const either<L, R> &e) { return e.left(); }
};

template <typename L, typename R1, typename R2, typename ... Rs>
struct fmap_all_helper<either<L, R1>, either<L, R2>, either<L, Rs>...> {
  constexpr static bool any_left(const either<L, R1> &e1, const either<L, R2> &e2, const either<L, Rs> &... rest) {
    return e1.is_left() || fmap_all_helper<either<L, R2>, either<L, Rs>...>::any_left(e2, rest...);
  }

  constexpr static const L &first_left(const either<L, R1> &e1, const either<L, R2> &e2, const either<L, Rs> &... rest) {
    return (e1.is_left()) ? e1.left() : fmap_all_helper<either<L, R2>, either<L, Rs>...>::first_left(e2, rest...);
  }
};

template <typename L, typename T> struct maybe_wrap { using type = either<L, T>; };
template <typename L, typename L2, typename R> struct maybe_wrap<L, either<L2, R>> { using type = either<L2, R>; };
template <typename L, typename T> using maybe_wrap_t = typename maybe_wrap<L, T>::type;

template <typename Fun, typename L, typename ... Ts>
constexpr auto fmap_all(Fun &&f, const either<L, Ts> &... es) {
  using f_ret = std::invoke_result_t<Fun, const Ts &...>;
  using ret_t = maybe_wrap_t<L, f_ret>;
  using helper_t = fmap_all_helper<either<L, Ts>...>;
  if(helper_t::any_left(es...))
    return ret_t{ helper_t::first_left(es...) };

  if constexpr(std::same_as<f_ret, ret_t>)
    return f(es.right()...);
  else
    return ret_t{ f(es.right()...) };
}
}

#endif //CUTRACE_EITHER_HPP
