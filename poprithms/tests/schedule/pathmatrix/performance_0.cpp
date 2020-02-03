#include <chrono>
#include <iostream>
#include <random>
#include <testutil/schedule/pathmatrix/pathmatrixcommandlineoptions.hpp>
#include <poprithms/schedule/pathmatrix/error.hpp>
#include <poprithms/schedule/pathmatrix/pathmatrix.hpp>

using namespace poprithms::schedule::pathmatrix;

int main(int argc, char **argv) {

  auto opts = PathMatrixCommandLineOptions().getCommandLineOptionsMap(
      argc,
      argv,
      {"N", "E", "D"},
      {"Number of Ops",
       "Number of out edges per Op",
       "Maximum inter-index edge length"});
  auto N = std::stoi(opts.at("N"));
  auto E = std::stoi(opts.at("E"));
  auto D = std::stoi(opts.at("D"));

  std::vector<std::vector<OpId>> fwd(N);
  std::mt19937 gen(1012);
  std::vector<uint64_t> indices(N);
  std::iota(indices.begin(), indices.end(), 0);

  if (E > D) {
    throw error("E cannot be larger than D");
  }
  if (D > N - 10) {
    throw error("D cannot be larger than N - 10");
  }

  auto nRando = N - D - 1;
  for (uint64_t i = 0; i < nRando; ++i) {
    fwd[i].reserve(E);
    std::sample(indices.begin() + i + 1,
                indices.begin() + i + 1 + D,
                std::back_inserter(fwd[i]),
                E,
                gen);
  }
  for (uint64_t i = nRando; i < N - 1; ++i) {
    fwd[i].push_back(i + 1);
  }

  auto start = std::chrono::high_resolution_clock::now();
  PathMatrix fem(fwd);
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = stop - start;

  bool printConnectivity{false};
  if (printConnectivity) {
    std::cout << "\nConstraint Map. v[i][j] = 1 iff i->j is a constraint. \n"
              << std::endl;
    for (uint64_t from = 0; from < fem.nOps_u64(); ++from) {
      for (uint64_t to = 0; to < fem.nOps_u64(); ++to) {
        std::cout << fem.constrained(from, to);
      }
      std::cout << std::endl;
    }
  }

  std::cout << "Total time to construct PathMatrix = " << elapsed.count()
            << " [s]" << std::endl;
  return 0;
};
