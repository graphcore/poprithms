// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace {

using namespace poprithms::memory::nest;

void testConstructor0() {

  bool caught = false;
  try {
    Region r = Region({}, {});
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error("Failed to catch error using default "
                                 "Region constructor for rank-0 Region");
  }
}

void testConstructor1() {

  auto r0 = Region::createEmpty({});
  auto r1 = Region::createFull({});

  if (r0.totalElms() != 0) {
    throw poprithms::test::error(
        "Empty scalar region should have 0 elements");
  }

  if (r1.totalElms() != 1) {
    throw poprithms::test::error("Full scalar region should have 1 element");
  }

  if (!r0.empty() || r0.full()) {
    throw poprithms::test::error("createEmpty should create an empty region");
  }

  if (!r1.full() || r1.empty()) {
    throw poprithms::test::error("createFull should create a full region");
  }

  if (r0.rank_u64() != 0) {
    throw poprithms::test::error("Scalar should have rank 0");
  }
}

void testScalar0() {

  auto r0 = Region::createEmpty({});
  auto r1 = Region::createFull({});

  if (!Region::createEmpty({}).empty()) {
    throw poprithms::test::error("Empty should be ... empty!");
  }

  if (!Region::createFull({}).getComplement().empty()) {
    throw poprithms::test::error("full.complement should be empty");
  }

  if (!r1.getComplement().empty()) {
    throw poprithms::test::error("The complement of a full scalar is empty");
  }

  if (r1.intersect(r1).totalElms() != 1) {
    throw poprithms::test::error(
        "Intersection of 2 full scalars is a full scalar");
  }

  if (r1.intersect(r0).totalElms() != 0) {
    throw poprithms::test::error(
        "Intersection of a full and an empty scalar is an empty scalar");
  }

  if (r0.intersect(r0).totalElms() != 0) {
    throw poprithms::test::error(
        "Intersection of 2 empty scalars is an empty scalar");
  }

  if (!r0.getComplement().full()) {
    throw poprithms::test::error(
        "The complement of an empty scalar is a full scalar");
  }

  if (!r0.subtract(r0).empty()) {
    throw poprithms::test::error(
        "empty scalar 'minus' empty scalar is empty. ");
  }

  if (!r0.subtract(r1).empty()) {
    throw poprithms::test::error(
        "empty scalar 'minus' full scalar is empty. ");
  }

  if (!r1.subtract(r1).empty()) {
    throw poprithms::test::error(
        "full scalar 'minus' full scalar is empty. ");
  }

  if (!r1.subtract(r0).full()) {
    throw poprithms::test::error(
        "full scalar 'minus' empty scalar is full. ");
  }
}

void testScalar1() {

  auto r0 = Region::createEmpty({});
  auto r1 = Region::createFull({});

  if (!r0.settSample(r1).empty()) {
    throw poprithms::test::error(
        "sampling with where=full returns the sampled region");
  }

  if (!r1.settSample(r1).full()) {
    throw poprithms::test::error(
        "sampling with where=full returns the sampled region");
  }

  if (!r0.settFillInto(r1).empty()) {
    throw poprithms::test::error(
        "filling with scaffold=full returns the ink");
  }

  if (!r1.settFillInto(r1).full()) {
    throw poprithms::test::error(
        "filling with scaffold=full returns the ink");
  }

  if (r0.reduce({}).full()) {
    throw poprithms::test::error("reducing empty scalar is empty scalar");
  }

  if (r1.reduce({}).empty()) {
    throw poprithms::test::error("reducing empty scalar is empty scalar");
  }

  if (r1.reshape({1, 1}).empty() || r0.reshape({1, 1}).full()) {
    throw poprithms::test::error("Reshaping conserves number of elements");
  }

  if (r1.flatten().empty() || r0.flatten().full()) {
    throw poprithms::test::error(
        "Flattening scalars leaves their number of on elements unchanged");
  }

  if (r1.reverse({}).empty() || r0.reverse({}).full()) {
    throw poprithms::test::error(
        "Reversing scalars leaves their number of on elements unchanged");
  }

  if (r1.expand({2, 3}).empty() || r0.expand({2, 3}).full()) {
    throw poprithms::test::error(
        "Expanding scalars leaves their full/empty status unchanged");
  }

  if (r1.dimShuffle(Permutation({})).empty() ||
      r0.dimShuffle(Permutation({})).full()) {
    throw poprithms::test::error(
        "DimShuffling scalars leaves their number of on elements unchanged");
  }

  if (r0.contains(r1)) {
    throw poprithms::test::error("r0 does not contain r1");
  }

  if (!r1.contains(r1)) {
    throw poprithms::test::error("r1 does contain r1");
  }

  if (!r1.contains(r0)) {
    throw poprithms::test::error("r1 does contain r0");
  }

  if (!r0.contains(r0)) {
    throw poprithms::test::error("r0 does contain r0");
  }
}

} // namespace

int main() {
  testConstructor0();
  testConstructor1();
  testScalar0();
  testScalar1();
  return 0;
}
