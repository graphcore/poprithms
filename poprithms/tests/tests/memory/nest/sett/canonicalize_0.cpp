// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/sett.hpp>

namespace {
using namespace poprithms::memory::nest;
void assertDepth(const Sett &p, uint64_t d) {
  if (p.recursiveDepth_u64() != d) {
    std::ostringstream oss;
    oss << "Failure in Sett test of recursiveDepth. Expected " << p
        << " to have depth " << d << ", not " << p.recursiveDepth();
    throw error(oss.str());
  }
}

void assertStripe(const Sett &p, uint64_t d, const Stripe &s) {
  if (p.recursiveDepth_u64() <= d) {
    std::ostringstream oss;
    oss << "Failure in assertStripe, Sett " << p << " not deep enough, "
        << "expected deptth greater than " << d;
    throw error(oss.str());
  }

  if (p.atDepth(d) != s) {
    std::ostringstream oss;
    oss << "Expected " << s << " at depth " << d << " of " << p;
    throw error(oss.str());
  }
}

} // namespace

int main() {

  using namespace poprithms::memory::nest;

  // with a single Stripe of on() = 0: reduce to simplest case
  Sett p3{{{{100, 100, 0}, {20, 13, 7}, {0, 5, 3}, {2, 0, 1}}}};
  assertDepth(p3, 1);

  // second stripe is redundant, check that it is removed
  Sett p4{{{{10, 5, 3}, {12, 1, 0}, {1, 1, 2}}}};
  assertDepth(p4, 2);
  assertStripe(p4, 0, {10, 5, 3});
  assertStripe(p4, 1, {1, 1, 0});

  // ..xxxxxxxxxx..........xxxxxxxxxx..........xxxxxxxxxx
  // ..x...................x...................x.........
  Sett p5{{{{10, 10, 2}, {2, 18, 19}}}};
  assertDepth(p5, 1);
  assertStripe(p5, 0, {1, 19, 2});

  // ..xxxxxxxxxx..........xxxxxxxxxx..........xxxxxxxxxx
  // ..x........x..........x........x..........x........x
  Sett p6{{{{10, 10, 2}, {2, 8, 9}}}};
  assertDepth(p6, 2);

  // ..xxxxxxxxxx..........xxxxxxxxxx..........xxxxxxxxxx
  // ..x...................x...................x.........
  p6 = Sett{{{{10, 10, 2}, {2, 9, 10}}}};
  assertDepth(p6, 1);
  assertStripe(p6, 0, Stripe(1, 19, 2));

  // ..xxxxxxxxxx..........xxxxxxxxxx..........xxxxxxxxxx
  // ...xx..................xx..................xx.......
  p6 = Sett{{{{10, 10, 2}, {2, 9, 1}}}};
  assertDepth(p6, 1);
  assertStripe(p6, 0, Stripe(2, 18, 3));

  // ..xxxxxxxxxx..........xxxxxxxxxx..........xxxxxxxxxx
  // ...xx..................xx..................xx.......
  p6 = Sett{{{{10, 10, 2}, {2, 8, 1}}}};
  assertDepth(p6, 1);
  assertStripe(p6, 0, Stripe(2, 18, 3));

  // ..xxxxxxxxxx..........xxxxxxxxxx..........xxxxxxxxxx
  // ...xx..................xx..................xx.......
  p6 = Sett{{{{10, 10, 2}, {2, 7, 1}}}};
  assertDepth(p6, 1);
  assertStripe(p6, 0, Stripe(2, 18, 3));

  // ....xxxxxxxxxxxxxxx..... parent
  // ..xxxxxxxxxxxxxxxxxx..xx child
  p6 = Sett{{{{10, 100, 4}, {13, 2, -2}}}};
  assertDepth(p6, 1);
  assertStripe(p6, 0, Stripe(10, 100, 4));

  // ....xxxxxxxxxxxxxxx..... parent
  // ..xxxxxxxxxxxxxxxx....xx child
  p6 = Sett{{{{10, 100, 4}, {11, 2, -2}}}};
  assertDepth(p6, 1);
  assertStripe(p6, 0, Stripe(9, 101, 4));

  // ....xxxxxxxxxxxxxxx.....  parent
  // ..xxxx..............xxxx  child
  p6 = Sett{{{{10, 100, 4}, {4, 9, -2}}}};
  assertDepth(p6, 1);
  assertStripe(p6, 0, Stripe(2, 108, 4));

  // ....x...................  parent
  // ..xxxx..............xxxx  child
  p6 = Sett{{{{1, 100, 4}, {4, 9, -2}}}};
  assertDepth(p6, 1);
  assertStripe(p6, 0, Stripe(1, 100, 4));

  // ....x......... parent  (1,100,4)
  // xx....xx....xx child   (2,4,2)
  p6 = Sett{{{{1, 100, 4}, {2, 4, 2}}}};
  assertDepth(p6, 1);
  if (p6.atDepth(0).on() != 0 || p6.atDepth(0).period() <= 0) {
    throw error("Error in testing basic Sett");
  }

  // .....xxxxxxxxxx..........xxxxxxxxxx..........
  //      .....xxxxx
  p6 = Sett({{{{10, 10, 5}, {5, 5, 5}}}});
  assertDepth(p6, 1);
  assertStripe(p6, 0, Stripe(5, 15, 10));

  // .....xxxxxxxxxx..........xxxxxxxxxx..........
  //      x....xxxxxx
  p6 = Sett({{{{10, 10, 5}, {6, 4, 5}}}});
  assertDepth(p6, 2);

  // .....xxxxxxxxxx..........xxxxxxxxxx..........
  //      .....xxxxxx
  p6 = Sett({{{{10, 10, 5}, {6, 5, 5}}}});
  assertDepth(p6, 1);
  assertStripe(p6, 0, Stripe(5, 15, 10));

  // .....xxxxxxxxxx..........xxxxxxxxxx..........
  //      ...xxxx...xxxx
  p6 = Sett({{{{10, 10, 5}, {4, 3, 3}}}});
  assertDepth(p6, 1);
  assertStripe(p6, 0, Stripe(4, 16, 8));

  // .....xxxxxxxxxx..........xxxxxxxxxx..........
  //      ..xxxx...xxxx
  p6 = Sett({{{{10, 10, 5}, {4, 3, 2}}}});
  assertDepth(p6, 2);

  // .....xxxxxxxxxx..........xxxxxxxxxx..........
  //    .xxxxxxxxxxxx.
  p6 = Sett({{{{10, 1000, 17}, {12, 1, -1}}}});
  assertDepth(p6, 1);
  assertStripe(p6, 0, Stripe(10, 1000, 17));

  p6 = Sett({{{{10, 1000, 17}, {12, 100, 70}}}});
  assertDepth(p6, 1);
  if (p6.atDepth(0).on() != 0) {
    std::ostringstream oss;
    oss << "Failed in test of " << p6;
    throw error(oss.str());
  }

  // .....x...........x..... (1,11,5)
  //      xx..xxxx..         (4,2,4)
  //      x.  x.x.           (1,1,0)
  p6 = Sett({{{{1, 11, 5}, {4, 2, 4}, {1, 1, 0}}}});
  assertDepth(p6, 1);
  assertStripe(p6, 0, {1, 11, 5});

  return 0;
}
