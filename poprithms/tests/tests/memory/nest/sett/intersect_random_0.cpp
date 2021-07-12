// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <random>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/memory/nest/sett.hpp>
#include <poprithms/util/unisort.hpp>
#include <testutil/memory/nest/randomsett.hpp>

namespace {

int max0 = 50;

using namespace poprithms::memory::nest;

void assertAgreement(const poprithms::memory::nest::Sett &p0,
                     const poprithms::memory::nest::Sett &p1) {

  auto U = p0.smallestCommonMultiple(p1) + 100;

  auto intersect = p0.intersect(p1);
  auto ons0      = p0.getOns(0, U);
  auto ons1      = p1.getOns(0, U);
  decltype(ons1) interOns;
  for (auto x : intersect) {
    for (auto v : x.getOns(0, U)) {
      interOns.push_back(v);
    }
  }
  interOns = poprithms::util::unisorted(interOns);

  decltype(ons0) intersectMan;
  if (!ons0.empty() && !ons1.empty()) {
    auto iter0 = ons0.cbegin();
    auto iter1 = ons1.cbegin();
    while (iter0 != ons0.cend() && iter1 != ons1.cend()) {
      if (*iter0 == *iter1) {
        intersectMan.push_back(*iter0);
        ++iter0;
        ++iter1;
      } else if (*iter0 < *iter1) {
        ++iter0;
      } else {
        ++iter1;
      }
    }
  }

  if (p0.n(0, U) != static_cast<int64_t>(ons0.size()) ||
      p1.n(0, U) != static_cast<int64_t>(ons1.size())) {
    throw poprithms::test::error("Failed in piggy-back test for Sett::n(.)");
  }

  if (intersectMan != interOns) {
    std::ostringstream oss;
    oss << "\n\nFailed in random intersect test. \nsett0 = " << p0
        << ", \n     with number of ons = " << ons0.size()
        << " and \nsett1 = " << p1
        << ", \n     with number of ons = " << ons1.size() << ".\n"
        << "The computed intersection has number of ons  = "
        << interOns.size() << "."
        << "\nThe baseline computed intersection has number of ons = "
        << intersectMan.size() << ".";
    throw poprithms::test::error(oss.str());
  }

  // we also check disjoint
  if (interOns.empty() != p0.disjoint(p1) ||
      interOns.empty() != p1.disjoint(p0)) {
    std::ostringstream oss;
    oss << "Failed to compute disjoint correctly, for " << p1 << " and "
        << p0;
    throw poprithms::test::error(oss.str());
  }
}

} // namespace

int main() {

  bool shorten = true;

  std::cout << "test number\n-------------\n";
  for (uint64_t ti = 0; ti < 1024; ++ti) {
    std::cout << ti << ' ';
    if (ti % 16 == 15) {
      std::cout << std::endl;
    }

    std::mt19937 gen(ti + 12);
    auto depth0 = 0 + gen() % 4;
    auto depth1 = 0 + gen() % 4;

    bool canonicalize0 = gen() % 1;
    bool canonicalize1 = gen() % 1;

    auto p0 = poprithms::memory::nest::getRandom(
        shorten, depth0, canonicalize0, ti + 100, max0);

    auto p1 = poprithms::memory::nest::getRandom(
        shorten, depth1, canonicalize1, ti + 200, max0);

    assertAgreement(p0, p1);
  }

  return 0;
}
