// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace {

using namespace poprithms::memory::nest;

void testDimshuffle() {

  const Permutation p({3, 1, 0, 2});
  const Shape s{2, 5, 7, 11};

  {
    auto r  = Region::createEmpty(s);
    auto r0 = r.dimShuffle(p);
    if (!r0.equivalent(Region::createEmpty(s.dimShuffle(p)))) {
      throw poprithms::test::error("failed in rank-4 empty dimShuffle test");
    }
  }

  {
    auto r  = Region::createFull({2, 5, 7, 11});
    auto r0 = r.dimShuffle(p);
    if (!r0.equivalent(Region::createFull({11, 5, 2, 7}))) {
      throw poprithms::test::error("failed in rank-4 empty dimShuffle test");
    }
  }

  {
    auto r  = Region::createEmpty(s);
    auto r0 = r.dimShuffle(Permutation::identity(4));
    if (!r0.equivalent(r)) {
      throw poprithms::test::error(
          "Failed in identity dimshuffle test : Region unchanged");
    }

    auto r1 = r.dimShuffle(p);
    if (r1.equivalent(r0)) {
      throw poprithms::test::error("Region after dimShuffle is different");
    }
  }
}

void testReverse() {
  Shape s{2, 5, 6};
  {
    Region r0 = Region::createEmpty(s);
    if (!r0.equivalent(r0.reverse({0, 1}))) {
      throw poprithms::test::error(
          "Reversing an empty region should not change the region");
    }
  }
  {

    Region r0 = Region::createFull(s);
    if (!r0.equivalent(r0.reverse({0, 1}))) {
      throw poprithms::test::error(
          "Reversing a full region should not change the region");
    }
  }
}

void testExpand() {

  Shape s{2, 5, 6};
  {
    Region r0 = Region::createEmpty({});
    if (!r0.expand(s).empty()) {
      throw poprithms::test::error(
          "Expanding an empty region should result in an empty region");
    }
  }
  {

    Region r0 = Region::createFull({1, 1});
    if (!r0.expand(s).full()) {
      throw poprithms::test::error(
          "Expanding a full region should result in a full region");
    }
  }
}

void testIntersectEtc() {

  Region r1 = Region::createFull({44, 9});
  auto r2   = Region::createFull({44 * 9}).reshape({44, 9});
  auto c    = r2.intersect({r1});
  if (!c.full()) {
    throw poprithms::test::error(
        "Intersection of full regions should be full region");
  }

  auto out = r2.subtract(r1);

  if (!out.empty()) {
    throw poprithms::test::error("Full minus full should be empty");
  }

  if (!out.subtract(r1).empty()) {
    throw poprithms::test::error("Empty minus full should be full");
  }

  if (!out.contains(out)) {
    throw poprithms::test::error("Empty contains empty");
  }

  if (out.contains(r1)) {
    throw poprithms::test::error("Empty does not contain full");
  }

  if (!r1.settSample(r1).full()) {
    throw poprithms::test::error(
        "slice (settSample) of full over full is full");
  }

  if (!out.settSample(r1).empty()) {
    throw poprithms::test::error(
        "slice (settSample) of empty over full is empty");
  }
}

void testEmptyRegions() {

  for (Shape s : {Shape({}), Shape({1, 2, 3})}) {
    Region r1 = Region::createEmpty(s);
    DisjointRegions rs(s, {r1, r1, r1, r1, r1});
    if (!rs.get().empty()) {
      throw poprithms::test::error("Expected empty regions to be removed in "
                                   "DisjointRegions constructor");
    }
  }
}
} // namespace

int main() {

  testDimshuffle();
  testReverse();
  testExpand();
  testIntersectEtc();
  testEmptyRegions();

  return 0;
}
