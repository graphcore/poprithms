// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/format/format_fwd.hpp>

#include <testutil/schedule/shift/randomgraph.hpp>

#include <poprithms/error/error.hpp>
#include <poprithms/schedule/shift/graph.hpp>
#include <poprithms/schedule/shift/scheduledgraph.hpp>

namespace {

using namespace poprithms::schedule::shift;

void testNoWrites() {

  uint64_t nOps{73};
  auto g0 = getRandomGraph(nOps, 3, 6, 1011);

  for (auto fw : {FileWriter::None(), FileWriter(".", 0)}) {

    auto sg = ScheduledGraph::fromCache(
        std::move(g0),
        Settings({KahnTieBreaker::FIFO, {}},
                 TransitiveClosureOptimizations::allOff(),
                 Settings::defaultRotationTermination(),
                 RotationAlgo::RIPPLE,
                 1011),
        fw,
        nullptr,
        nullptr);

    if (boost::filesystem::exists(FileWriter::finalDirName(0, nOps, 0))) {
      throw poprithms::test::error(
          "Filewriter with maxWritesPerBin of 0 should not create directory");
    }
  }
}
} // namespace

int main() {
  testNoWrites();
  return 0;
}
