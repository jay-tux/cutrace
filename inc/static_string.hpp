//
// Created by jay on 11/28/23.
//

#ifndef CUTRACE_STATIC_STRING_HPP
#define CUTRACE_STATIC_STRING_HPP

#include <algorithm>

namespace cutrace {
template <size_t N>
struct static_string {
  char buf[N + 1]{};

  constexpr static_string(const char *s) noexcept {
    for(size_t i = 0; i < N; i++) buf[i] = s[i];
  }

  constexpr static_string(const char (&str)[N]) {
    std::copy_n(str, N, buf);
  }

  constexpr const char *c_str() const { return buf; }

  constexpr operator const char *() const { return buf; }
  constexpr operator std::string() const { return std::string((const char *)(*this)); }
};

template <size_t N> static_string(const char (&)[N]) -> static_string<N - 1>;
}

#endif //CUTRACE_STATIC_STRING_HPP
