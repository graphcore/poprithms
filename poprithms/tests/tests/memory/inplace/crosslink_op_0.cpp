// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <memory>

#include <memory/inplace/ops.hpp>
#include <poprithms/memory/inplace/error.hpp>

int main() {

  using namespace poprithms::memory::inplace;

  OpId id{100};
  OpIds inIds{50, 56};
  OpIds outIds{110, 112};
  TensorIds inTensors{{50, 0}, {56, 0}};
  Shapes inShapes({{1}, {2}});

  Multi m0(
      Op::State(
          id, inIds, outIds, inTensors, {}, inShapes, {{3}, {4}}, "myMulti0"),
      {});
  std::cout << "m0 : " << m0 << std::endl;

  Multi m1(
      Op::State(
          id, inIds, outIds, inTensors, {}, inShapes, {{1}, {2}}, "myMulti1"),
      {CrossLink::pureAliases(0, 0), CrossLink::modifies(1, 1)});
  std::cout << "m1 : " << m1 << std::endl;

  Multi m2(
      Op::State(
          id, inIds, outIds, inTensors, {}, inShapes, {{1}, {4}}, "myMulti2"),
      {CrossLink::pureAliases(0, 0),
       CrossLink::uses(1, 1, std::make_unique<IdentityRegsMap>())});
  std::cout << "m2 : " << m2 << std::endl;

  bool caught{false};
  try {
    Multi(Op::State(
              id, inIds, outIds, inTensors, {}, inShapes, {{1}, {1}}, "bad0"),
          {CrossLink::pureAliases(0, 0), CrossLink::pureAliases(1, 0)});
  } catch (const poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    throw error("Cannot have output aliasing 2 inputs");
  }

  caught = false;
  try {
    Multi(Op::State(
              id, inIds, outIds, inTensors, {}, inShapes, {{1}, {2}}, "bad1"),
          {CrossLink::pureAliases(0, 1), CrossLink::pureAliases(1, 0)});
  } catch (const poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    throw error("Cannot have outputs aliasing inputs of different sizes");
  }

  caught = false;
  try {
    Multi(Op::State(
              id, inIds, outIds, inTensors, {}, inShapes, {{1}, {2}}, "bad2"),
          {CrossLink::uses(0, 0, std::make_unique<IdentityRegsMap>()),
           CrossLink::modifies(0, 0)});
  } catch (const poprithms::error::error &e) {
    caught = true;
  }
  if (!caught) {
    throw error("Cannot have multi appearances of same in/out pair");
  }

  return 0;
}
