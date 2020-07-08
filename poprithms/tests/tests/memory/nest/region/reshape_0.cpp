// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/printiter.hpp>

namespace {

using namespace poprithms::memory::nest;

// We test that the computed reshape is
// 1) equivalent to the expected reshape
// 2) does not do more shattering than the expected reshape.
// In other words, we don't check for exact Region correspondence.
void assertReshape(const Region &r,
                   const Shape &to,
                   const DisjointRegions &expected) {
  auto reshaped = r.reshape(to);

  std::ostringstream oss;
  oss << "In assertReshape(r=" << r << ", to=" << to
      << ", expected=" << expected << "). Computed reshape=" << reshaped
      << '.';
  if (reshaped.size() > expected.size()) {
    oss << " As expected only contains " << expected.size()
        << ", but reshaped contains " << reshaped.size()
        << ", an error is being reported.";
    throw error(oss.str());
  }

  if (!reshaped.isValid()) {
    oss << " reshaped is not a valid DisjointRegs.";
    throw error(oss.str());
  }

  if (!expected.isValid()) {
    oss << " expected is not a valid DisjointRegs.";
    throw error(oss.str());
  }

  if (!Region::equivalent(reshaped, expected)) {
    oss << " reshaped and expected are not equivalent.";
    throw error(oss.str());
  }
}

void test0() {

  //                   .....
  //                   .....
  //                   11111
  // ..........        11111
  // 1111111111        11111
  // 1111111111  ==>   11111
  // 1111111111        11111
  // ..........        11111
  //                   .....
  //                   .....
  //
  //
  assertReshape(Region({5, 10}, {{{{3, 2, 1}}}, {{}}}),
                Shape({10, 5}),
                {Region({10, 5}, {{{{6, 4, 2}}}, {{}}})});
}

void test1() {

  //                   .....     .....     .....
  //                   .....     .....     .....
  //                   .1111     .....     .1111
  // ..........        1111.     1111.     .....
  // .11111111.        .1111     .....     .1111
  // .11111111.  ==>   1111.  =  1111.  +  .....
  // .11111111.        .1111     .....     .1111
  // ..........        1111.     1111.     .....
  //                   .....     .....     .....
  //                   .....     .....     .....
  //

  assertReshape(
      Region({5, 10}, {{{{3, 2, 1}}}, {{{8, 2, 1}}}}),
      Shape({10, 5}),
      DisjointRegions(
          Shape({10, 5}),
          {Region({10, 5}, {{{{6, 4, 2}, {1, 1, 0}}}, {{{4, 1, 1}}}}),
           Region({10, 5}, {{{{6, 4, 2}, {1, 1, 1}}}, {{{4, 1, 0}}}})}));
}

void test2() {

  // .....
  // .....      ..........
  // 1111.      1111.1111.
  // 1111. ==>  1111.1111.
  // 1111.      1111.1111.
  // 1111.      1111......
  // 1111.
  // 1111.
  // 1111.
  // .....

  assertReshape(
      Region({10, 5}, {{{{7, 3, 2}}}, {{{4, 1, 0}}}}),
      Shape({5, 10}),
      DisjointRegions(Shape({5, 10}),
                      {Region({5, 10}, {{{{3, 1, 1}}}, {{{4, 6, 5}}}}),
                       Region({5, 10}, {{{{4, 1, 1}}}, {{{4, 6, 0}}}})}));
}

} // namespace

int main() {
  test0();
  test1();
  test2();

  std::cout
      << Region({5, 10}, {{{{3, 2, 1}}}, {{{8, 2, 1}}}}).reshape({10, 5})
      << std::endl;
}
