#include "catch2/catch_all.hpp"
#include "either.hpp"

using namespace cutrace;

TEST_CASE("Basic either test") {
  either<int, float> v1;
  REQUIRE(v1.is_left());
  REQUIRE(!v1.is_right());

  either<int, float> v2 = {3};
  REQUIRE(v2.is_left());
  REQUIRE(!v2.is_right());

  either<int, float> v3 = {5.75f};
  REQUIRE(!v3.is_left());
  REQUIRE(v3.is_right());
}

TEST_CASE("either::map test") {
  either<int, float> v1 = { 5.5f };
  either<int, float> v2 = { 5 };

  auto res = v1.map([](const float &f) { return std::to_string(f); });
  REQUIRE(res.is_right());

  auto res2 = v2.map([](const float &f) { return std::to_string(f); });
  REQUIRE(res2.is_left());
  REQUIRE((const int &)res2 == 5);

  auto divide_100_safe = [](const int &i) {
    if(i == 0) return either<std::string, int>{ .value = "Cannot divide by 0" };
    else return either<std::string, int>{ .value = 100 / i };
  };

  either<double, int> v3 = {0};
  either<double, int> v4 = {10};

  auto r3 = v3.map(divide_100_safe);
  auto r4 = v4.map(divide_100_safe);

  REQUIRE((r3.is_right() && r3.right().is_left() && r3.right().left() == "Cannot divide by 0"));
  REQUIRE((r4.is_right() && r4.right().is_right() && r4.right().right() == 10));
}

TEST_CASE("either::fmap test") {
  auto to_either = [](const char &c) -> either<float, int> {
    if(c >= 32 && c <= 126) return { (int)c };
    else return {};
  };

  either<float, char> left{ 0.6f };
  either<float, char> becomes_left { (char)3 };
  either<float, char> stays_right { '6' };

  auto r_left = left.fmap(to_either);
  auto r_b_left = becomes_left.fmap(to_either);
  auto r_s_right = stays_right.fmap(to_either);

  REQUIRE(r_left.is_left());
  REQUIRE(r_left.left() == 0.6f);
  REQUIRE(r_b_left.is_left());
  REQUIRE(r_b_left.left() == 0.0f);
  REQUIRE(r_s_right.is_right());
  REQUIRE(r_s_right.right() == (int)'6');
}

TEST_CASE("either::fmap_left turn into right") {
  either<float, char> left{ 0.6f };

  auto mapped = left.fmap_left([](const float &f){ return either<std::string, char>{ 'e' }; });
  REQUIRE(mapped.is_right());
  REQUIRE(mapped.right() == 'e');
}