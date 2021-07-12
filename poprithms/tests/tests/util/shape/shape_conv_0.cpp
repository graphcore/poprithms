// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <array>
#include <iostream>

#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/printiter.hpp>

namespace {
using namespace poprithms::ndarray;

std::ostream &operator<<(std::ostream &ost, const std::vector<uint64_t> &x) {
  poprithms::util::append(ost, x);
  return ost;
}

void confirmConvShape(const Shape &data,
                      const Shape &kernel,
                      const Shape &expected,
                      const std::vector<uint64_t> &lowPrePads,
                      const std::vector<uint64_t> &uppPrePads,
                      const Dilations &dilations,
                      const Strides &strides) {

  const auto observed =
      data.convolve(kernel, lowPrePads, uppPrePads, dilations, strides);

  if (observed != expected) {
    std::ostringstream oss;
    oss << "Failure in confirmConvShape, expected " << data
        << ".convolve(kernel=" << kernel << ", lowPrePads=" << lowPrePads
        << ", uppPrePads=" << uppPrePads << ", dilations=" << dilations.get()
        << ", strides=" << strides.get() << ')' << " But observed "
        << observed;
    throw poprithms::test::error(oss.str());
  }
}

void multiChannelTest() {

  const auto out =
      Shape({2, 3, 4, 5})
          .batchedMultiChannelConvolve({10, 1, 5}, {}, {}, {}, {});
  const auto expected = Shape({2, 10, 4, 1});
  if (out != expected) {
    std::ostringstream oss;
    oss << "Observed Shape in multiChannelTest: " << out << ", but expected "
        << expected << '.';
  }
}

} // namespace

int main() {

  confirmConvShape({3, 3}, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {}, {});

  confirmConvShape({3, 3}, {3, 3}, {2, 2}, {1, 0}, {0, 1}, {}, {});

  confirmConvShape(
      {5, 5}, {3, 3}, {2, 2}, {0, 0}, {0, 0}, {}, Strides({2, 2}));

  confirmConvShape(
      {4, 4}, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {}, Strides({2, 2}));

  confirmConvShape(
      {4, 4}, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {}, Strides({2, 2}));

  confirmConvShape(
      {5, 5}, {2, 2}, {1, 1}, {0, 0}, {0, 0}, Dilations({4, 4}), {});

  confirmConvShape(
      {5, 5}, {2, 2}, {0, 2}, {0, 0}, {0, 0}, Dilations({5, 3}), {});

  multiChannelTest();
  return 0;
}
