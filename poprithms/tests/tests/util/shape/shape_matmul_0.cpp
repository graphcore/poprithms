// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>

#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/shape.hpp>

namespace {
using namespace poprithms::ndarray;

void confirmMatmulShape(const Shape &a, const Shape &b, const Shape &c) {

  if (a.matmul(b) != c) {
    std::ostringstream oss;
    oss << "Failure in confirmMatmulShape, expected " << a << ".matmul(" << b
        << ") to be " << c << ", not " << a.matmul(b);
    throw poprithms::test::error(oss.str());
  }

  //
}
} // namespace

int main() {

  if (Shape::numpyVariadic({{{3, 1, 1}}, {{1, 4, 1}}, {{1, 1, 5}}}) !=
      Shape{3, 4, 5}) {
    throw poprithms::test::error("Failure in numpyVariadic test");
  }

  std::vector<std::array<Shape, 3>> shapes{
      // "If the first argument is 1-D, it is promoted by PREpending 1, and
      // then removing the dummy-1 at the end of the calculation."
      {Shape{1}, {{1, 1}}, {1}},
      {Shape{2}, {{2, 3}}, {3}},
      {Shape{3}, {{3, 1, 4, 3, 5}}, {3, 1, 4, 5}},

      // "If the second argument is 1-D, it is promoted by APPending 1, and
      // then removing the dummy-1 at the end of the calculation."
      {Shape{{1, 1}}, {1}, {1}},
      {Shape{{2, 3}}, {3}, {2}},
      {Shape{{3, 1, 4, 1, 5}}, {5}, {3, 1, 4, 1}},

      // The case where both are 1-D:
      {Shape{1}, {1}, {}},
      {Shape{10}, {10}, {}},

      // The numpy broadcasting cases:
      {Shape{5, 6}, {6, 7}, {5, 7}},
      {Shape{5, 6}, {2, 6, 7}, {2, 5, 7}},
      {Shape{1, 4, 1, 5, 100, 200},
       Shape{1, 2, 1, 7, 1, 200, 300},
       Shape{1, 2, 4, 7, 5, 100, 300}},
      {Shape{1, 1, 1, 5, 6}, {6, 7}, {1, 1, 1, 5, 7}},
      {Shape{1, 1, 1, 5, 6}, {10, 6, 7}, {1, 1, 10, 5, 7}},

  };

  for (const auto &x : shapes) {
    confirmMatmulShape(std::get<0>(x), std::get<1>(x), std::get<2>(x));
  }
  return 0;
}
