// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>

namespace {

using namespace poprithms::compute::host;

void basicExpand() {
  // Expand: only inplace with num-elements increasing creates an alias
  Tensor::int8(3).expand_({1, 1, 1}).assertContainsNoAliases();
  Tensor::int8(3).expand({1, 1, 1}).assertContainsNoAliases();
  Tensor::int8(3).expand({1, 2, 1}).assertContainsNoAliases();
  Tensor::int8(3).expand_({1, 2, 1}).assertContainsAliases();
}

void basicConcat() {

  const auto a = Tensor::float32(3.f).reshape_({1});
  const auto b = Tensor::float32(3.f).reshape_({1});
  concat_({a, b}, 0).assertContainsNoAliases();
  concat_({a, b - a}, 0).assertContainsNoAliases();
  concat_({a, b.add_(a)}, 0).assertContainsNoAliases();
  concat_({a, b, a, b, a, b, a}, 0)
      .slice_({1}, {3})
      .assertContainsNoAliases();

  concat_({a, b, a}, 0).assertContainsAliases();
  concat_({a, b, a, b, a, b, a}, 0).slice_({1}, {4}).assertContainsAliases();
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
  concat_({b.flatten_(), c.flatten_()}, 0).assertContainsAliases();
  concat_({b.flatten_(), d.flatten_()}, 0).assertContainsNoAliases();
  concat_({c.flatten_(), d.flatten_()}, 0).assertContainsAliases();
}

} // namespace

int main() {

  basicExpand();
  basicConcat();
  sliceReshapeSlice0();

  return 0;
}
