#include <iterator>
#include <numeric>
#include <string>
#include <poprithms/schedule/anneal/annealusings.hpp>
#include <poprithms/schedule/anneal/error.hpp>
#include <testutil/schedule/anneal/branch_doubling_generator.hpp>

namespace poprithms {
namespace schedule {
namespace anneal {

poprithms::schedule::anneal::Graph getBranchDoublingGraph(uint64_t nBranches,
                                                          uint64_t offset) {
  using namespace poprithms::schedule::anneal;

  Graph g;
  auto root = g.insertOp("root");

  std::vector<std::vector<OpAddress>> branchIds(nBranches);
  std::vector<OpAddress> branchEnds;
  std::vector<int> branchLengths;
  for (uint64_t branch = 0; branch < nBranches; ++branch) {
    auto nPostRoot    = g.nOps() - 1;
    auto branchLength = branch == 0 ? 3 : nPostRoot + offset;
    branchLengths.push_back(branchLength);
    for (uint64_t depth = 0; depth < branchLength; ++depth) {
      auto id = "Op_" + std::to_string(branch) + "_" + std::to_string(depth);
      branchIds[branch].push_back(g.insertOp(id));
      if (depth == 0) {
        g.insertConstraint(root, branchIds[branch][depth]);
      } else {
        g.insertConstraint(branchIds[branch][depth - 1],
                           branchIds[branch][depth]);
      }
    }
    auto id = "End_" + std::to_string(branch);
    branchEnds.push_back(g.insertOp(id));
    g.insertConstraint(branchIds[branch].back(), branchEnds[branch]);
    if (branch != 0) {
      g.insertConstraint(branchEnds[branch - 1], branchEnds[branch]);
    }
  }

  for (OpAddress a = 0; a < g.nOps(); ++a) {
    auto alloc = g.insertAlloc(1.0);
    g.insertOpAlloc(a, alloc);
    for (auto i : g.getOp(a).getOuts()) {
      g.insertOpAlloc(i, alloc);
    }
  }

  //  //ensure final branch runs last
  auto alloc = g.insertAlloc(100.0);
  g.insertOpAlloc(branchIds.back()[0], alloc);
  g.insertOpAlloc(branchEnds.back(), alloc);

  return g;
}

namespace {

std::vector<int> getBranchLengths(const Graph &g) {
  std::vector<int> lengths;
  for (const auto &op : g.getOps()) {
    auto dbs = op.getDebugString();
    if (dbs.find("Op") != std::string::npos) {
      auto a0 = std::find(dbs.begin(), dbs.end(), '_');
      auto a1 = std::find(std::next(a0), dbs.end(), '_');
      std::string b0{std::next(a0), a1};
      std::string b1{std::next(a1), dbs.end()};
      auto branch = std::stoi(b0);
      auto index  = std::stoi(b1);
      lengths.resize(branch + 1, 0);
      lengths[branch] = std::max(lengths[branch], index + 1);
    }
  }
  return lengths;
}

} // namespace

void assertGlobalMinimumBranchDoubling(const Graph &g,
                                       int nBranches,
                                       int offset) {
  auto branchLengths = getBranchLengths(g);

  std::vector<std::string> expected{"root"};
  if (offset < 0) {
    for (auto b = 0; b < nBranches; ++b) {
      for (auto i = 0; i < branchLengths[b]; ++i) {
        expected.push_back("Op_" + std::to_string(b) + "_" +
                           std::to_string(i));
      }
      expected.push_back("End_" + std::to_string(b));
    }
  }
  if (offset > 0) {
    for (int b = nBranches - 2; b >= 0; --b) {
      for (auto i = 0; i < branchLengths[b]; ++i) {
        expected.push_back("Op_" + std::to_string(b) + "_" +
                           std::to_string(i));
      }
    }
    for (int b = 0; b < nBranches - 1; ++b) {
      expected.push_back("End_" + std::to_string(b));
    }
    for (auto i = 0; i < branchLengths[nBranches - 1]; ++i) {
      expected.push_back("Op_" + std::to_string(nBranches - 1) + "_" +
                         std::to_string(i));
    }
    expected.push_back("End_" + std::to_string(nBranches - 1));
  }

  if (expected.size() != g.nOps()) {
    throw error("Expected vector is not the correct length");
  }

  auto livenessString = g.getLivenessString();
  for (auto i = 0; i < g.nOps(); ++i) {
    auto dbs = g.getOp(g.scheduleToOp(i)).getDebugString();
    if (dbs != expected[i]) {
      throw error("Unexpected");
    }
  }
}

} // namespace anneal
} // namespace schedule
} // namespace poprithms
