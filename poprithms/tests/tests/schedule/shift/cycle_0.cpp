// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <string>

#include <poprithms/schedule/shift/error.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace {
using namespace poprithms::schedule::shift;

void test0() {

  Graph g;

  const auto connect = [&g](const std::vector<OpAddress> &addresses) {
    for (uint64_t i = 0; i < addresses.size(); ++i) {
      g.insertConstraint(addresses[i], addresses[(i + 1) % addresses.size()]);
    }
  };

  const auto component2 = g.insertOps({"20_alpha", "21_beta"});
  connect(component2);

  const auto component3 = g.insertOps({"30_gamma", "31_delta", "32_epsilon"});
  connect(component3);

  const auto component5 =
      g.insertOps({"50_zeta", "51_eta", "52_theta", "53_iota", "54_kappa"});
  connect(component5);

  const auto inter = g.insertOp("between components");
  g.insertConstraints({{component2[0], inter}, {inter, component3[0]}});

  bool caught{false};
  try {
    ScheduledGraph sg(std::move(g),
                      KahnTieBreaker::RANDOM,
                      TransitiveClosureOptimizations::allOff(),
                      RotationTermination::preStart());
  } catch (const poprithms::error::error &e) {
    caught                    = true;
    const std::string message = e.what();

    // check that the message has something about Strongly Connected Component
    // in it:
    const auto F = message.find("omponent");
    if (F == std::string::npos) {
      throw error("Message should be about Connected Components");
    }
  }
  if (!caught) {
    throw error("Cycle not detected");
  }
}
} // namespace

int main() { test0(); }
