#include <array>
#include <bitset>
#include <chrono>
#include <iostream>
#include <vector>

#include <testutil/schedule/pathmatrix/pathmatrixcommandlineoptions.hpp>

// Motivation for going with BitSetSize = 512;

template <uint64_t NBits> void count(uint64_t repeat) {
  std::bitset<NBits> x0;
  auto start = std::chrono::high_resolution_clock::now();
  uint64_t sum{0};
  for (int i = 0; i < repeat; ++i) {
    sum += x0.count();
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> countElapsed = stop - start;
  std::cout << "@bitfield=" << NBits << " : "
            << (repeat * NBits) / countElapsed.count() << std::endl;
}

void simpleLoop(uint64_t repeat) {
  int j      = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; ++i) {
    j += 1;
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  std::cout << repeat / elapsed.count() << std::endl;
}

template <uint64_t NBits> void add(uint64_t repeat) {
  std::bitset<NBits> x0;
  std::bitset<NBits> x1;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; ++i) {
    x0 &= x1;
  }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> countElapsed = stop - start;
  std::cout << "@bitfield=" << NBits << " : "
            << (repeat * NBits) / countElapsed.count() << std::endl;
}

int main(int argc, char **argv) {

  using namespace poprithms::schedule::pathmatrix;
  auto opts = PathMatrixCommandLineOptions().getCommandLineOptionsMap(
      argc,
      argv,
      {"repeat"},
      {"Number of iterations to loop for in bencharks"});
  auto repeat = std::stoi(opts.at("repeat"));
  std::cout << "Simple Loop, integer adds per second:\n";
  simpleLoop(repeat);

  std::cout << "\nCount number of bits set, bits counted per second: \n";
  count<1>(repeat);
  count<2>(repeat);
  count<4>(repeat);
  count<8>(repeat);
  count<16>(repeat);
  count<32>(repeat);
  count<64>(repeat);
  count<128>(repeat);
  count<256>(repeat);
  count<512>(repeat);
  count<1024>(repeat);
  count<2048>(repeat);
  count<4096>(repeat);

  std::cout << "\nBitwise or, bits processed per second: \n";
  add<1>(repeat);
  add<2>(repeat);
  add<4>(repeat);
  add<8>(repeat);
  add<16>(repeat);
  add<32>(repeat);
  add<64>(repeat);
  add<128>(repeat);
  add<256>(repeat);
  add<512>(repeat);
  add<1024>(repeat);
  add<2048>(repeat);
  add<4096>(repeat);
  return 0;
}
