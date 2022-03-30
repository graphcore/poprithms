// Copyright 2021 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <iostream>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/program/callstack/copyin.hpp>
#include <poprithms/program/callstack/copymap.hpp>
#include <poprithms/program/callstack/copyout.hpp>

namespace {

using namespace poprithms::program::callstack;

void testIn0() {
  CopyIn c0{{0, 0}, {1, 0}, 0};
  CopyIn c1{{2, 0}, {1, 0}, 0};

  {
    bool caught{false};
    try {
      // One destination with multiple sources:
      CopyIns ins({c0, c1});
    } catch (const poprithms::error::error &) {
      caught = true;
    }
    if (!caught) {
      throw poprithms::test::error(
          "Failed to catch error of multiple sources");
    }
  }

  if (CopyIns({c0}).srcIds() != TensorIds{{0, 0}}) {
    throw poprithms::test::error("Source is (OpId=0,OutIndex=0)");
  }

  if (CopyIns({c0}).dstIds() != TensorIds{{1, 0}}) {
    throw poprithms::test::error("Destination is (OpId=1,OutIndex=0)");
  }

  bool caught{false};

  // Different number of srcs and dsts:
  TensorIds srcs{{0, 0}, {1, 0}, {2, 0}};
  TensorIds dsts{{3, 0}, {4, 0}};
  try {
    CopyIns::zip(srcs, dsts, CalleeIndex(1));
  } catch (const poprithms::error::error &) {
    caught = true;
  }
  if (!caught) {
    throw poprithms::test::error(
        "Failed to catch error of differen zip sizes");
  }

  dsts.push_back({5, 0});

  if (CopyIns(CopyIns::zip(srcs, dsts, CalleeIndex(2))).srcIds() != srcs) {
    throw poprithms::test::error("Sources changed in zipping");
  }
}

void testOut0() {

  CopyOuts copo(
      {TensorIds{{1, 0}, {2, 0}}, {{3, 0}, {4, 0}}, {{5, 0}, {6, 0}}});
  if (copo.nOutTensors() != 3) {
    throw poprithms::test::error(
        "The CopyOuts object has 3 outputs [OutIndex][CalleeIndex]");
  }
  if (copo.nCallees() != 2) {
    throw poprithms::test::error(
        "The CopyOuts object has 2 callees [OutIndex][CalleeIndex]");
  }

  if (copo.outSource(OutIndex(1), CalleeIndex(0)) != TensorId{3, 0}) {
    throw poprithms::test::error("The element [1][0] is TensorId(3,0)");
  }

  bool caught{false};
  try {
    CopyOuts copo2(std::map<CalleeIndex, TensorIds>{
        {{CalleeIndex{0}, TensorIds{{0, 0}, {1, 0}}},
         {CalleeIndex{1}, {{5, 0}}}}});
  }

  catch (const poprithms::error::error &) {
    caught = true;
  }

  if (!caught) {
    throw poprithms::test::error("Failed to chatch error where different "
                                 "callees have different number of outputs");
  }
}

class CopyOutTestGraph {

public:
  OpIds opIds() const { return {0, 1, 2, 3}; }
  std::vector<SubGraphId> callees(OpId opId) const {
    if (opId == 0) {
      return {SubGraphId::createSubGraphId(1),
              SubGraphId::createSubGraphId(2)};
    }
    return {};
  }
  CopyOuts outCopies(OpId opId) const {
    if (opId == 0) {
      OptionalTensorIds opts0{OptionalTensorId{}, TensorId(1, 0)};
      OptionalTensorIds opts1{TensorId(2, 0), TensorId(1, 0)};
      return CopyOuts::fromOptionals({opts0, opts1});
    }
    throw poprithms::test::error("Should not be reached");
  }
};

void testCopyOutMap0() {
  CopyOutTestGraph g;
  CopyOutMap cmp(g);
  if (cmp.n({1, 0}) != 2) {
    throw poprithms::test::error("output of callee 1 at 2 output indices");
  }

  if (cmp.n({2, 0}) != 1) {
    throw poprithms::test::error("output of callee 0 at 2 output index 0");
  }
  if (cmp.n({3, 0}) != 0) {
    throw poprithms::test::error("not an output");
  }
}

} // namespace

int main() {
  testIn0();
  testOut0();
  testCopyOutMap0();
  return 0;
}
