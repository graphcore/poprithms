// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>

#include <poprithms/memory/nest/error.hpp>
#include <poprithms/memory/nest/logging.hpp>
#include <poprithms/memory/nest/sett.hpp>

int main() {

  using namespace poprithms::memory::nest;
  Sett sett0{{{1944, 0, 0},
              {324, 1134, 162},
              {162, 0, 0},
              {70, 36, 90},
              {22, 16, 23}}};
  Sett sett1{{{160, 1784, 324}, {68, 144, 196}, {16, 60, 64}}};

  poprithms::logging::enableDeltaTime(true);

  // compare time to compute intersection and disjointedness
  poprithms::logging::Logger l("timing");
  l.setLevel(poprithms::logging::Level::Info);
  l.info("compute intersect from main");
  auto computedIntersection = sett0.intersect(sett1);
  l.info("compute disjoint from main");
  auto computedDisjoint = sett0.disjoint(sett1);
  l.info("return from main");

  auto intersectionEmpty = std::all_of(computedIntersection.cbegin(),
                                       computedIntersection.cend(),
                                       [](auto x) { return x.alwaysOff(); });

  if (intersectionEmpty != computedDisjoint) {
    throw error("Disagreemment between disjoint and intersect");
  }
  return 0;
}
