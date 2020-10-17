// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>
#include <numeric>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {

using namespace poprithms::compute::host;

void assertContainsAliases(const Tensor &t, bool expected) {
  if (t.containsAliases() != expected) {
    std::ostringstream oss;
    const auto expStr = expected ? "" : " NOT";
    oss << "Error in assertContainsAliases in test. "
        << "Tensor " << t << " was" << expStr
        << " expected to contain aliases. ";
    throw error(oss.str());
  }
}

void basicExpand() {
  // Expand: only inplace with num-elements increasing creates an alias
  assertContainsAliases(Tensor::int8(3).expand_({1, 1, 1}), false);
  assertContainsAliases(Tensor::int8(3).expand({1, 1, 1}), false);
  assertContainsAliases(Tensor::int8(3).expand({1, 2, 1}), false);
  assertContainsAliases(Tensor::int8(3).expand_({1, 2, 1}), true);
}

void basicConcat() {

  const auto a = Tensor::float32(3.f).reshape_({1});
  const auto b = Tensor::float32(3.f).reshape_({1});
  assertContainsAliases(concat_({a, b}, 0), false);
  assertContainsAliases(concat_({a, b - a}, 0), false);
  assertContainsAliases(concat_({a, b.add_(a)}, 0), false);
  assertContainsAliases(concat_({a, b, a, b, a, b, a}, 0).slice_({1}, {3}),
                        false);

  assertContainsAliases(concat_({a, b, a}, 0), true);
  assertContainsAliases(concat_({a, b, a, b, a, b, a}, 0).slice_({1}, {4}),
                        true);
}

void sliceReshapeSlice0() {

  // 1111....
  // 1111....
  // 11xxxx..
  // 11xxxx..
  // ..xxxx11
  // ..xxxx11
  // ....1111
  // ....1111
  const auto a = Tensor::arangeInt32(0, 144, 1).reshape_({12, 12});
  const auto b = a.slice_({0, 0}, {6, 6});
  const auto c = a.slice_({3, 3}, {9, 9});
  const auto d = a.slice_({6, 6}, {12, 12});
  assertContainsAliases(concat_({b.flatten_(), c.flatten_()}, 0), true);
  assertContainsAliases(concat_({b.flatten_(), d.flatten_()}, 0), false);
  assertContainsAliases(concat_({c.flatten_(), d.flatten_()}, 0), true);
}

} // namespace

int main() {

  basicExpand();
  basicConcat();
  sliceReshapeSlice0();

  return 0;
}
