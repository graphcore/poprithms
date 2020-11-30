// Copyright (c) 2015 Graphcore Ltd. All rights reserved.
#include "./include/numpyformatter.hpp"

#include <algorithm>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace compute {
namespace host {

void NumpyFormatter::append(const std::vector<std::string> &stringElements1d,
                            std::ostream &outStream,
                            const ndarray::Shape &shape,
                            uint64_t abbreviationThreshold) {

  if (stringElements1d.size() != shape.nelms_u64()) {
    std::ostringstream oss;
    oss << "stringElements1d -- the vector containing the "
        << "row major "
        << "string representation of the elements -- "
        << "has " << stringElements1d.size() << ". "
        << "It must have the same number of elements as " << shape
        << ", which is " << shape.nelms() << '.';
    throw error(oss.str());
  }

  if (stringElements1d.size() == 0) {
    outStream << "()";
    return;
  }

  if (shape.rank_u64() == 0) {
    outStream << "scalar(" << stringElements1d[0] << ")";
    return;
  }

  // Too many elements to fully represent:
  if (stringElements1d.size() > abbreviationThreshold) {
    const auto xi       = abbreviationThreshold / 2 - 1;
    const auto nOmmited = stringElements1d.size() - 2 * xi;
    std::vector<std::string> nv2;
    for (uint64_t i = 0; i < xi; ++i) {
      nv2.push_back(stringElements1d[i]);
    }
    nv2.push_back("...(" + std::to_string(nOmmited) + " more values)...");
    for (uint64_t i = stringElements1d.size() - xi;
         i != stringElements1d.size();
         ++i) {
      nv2.push_back(stringElements1d[i]);
    }
    poprithms::util::append(outStream, nv2);
    return;
  }

  auto strides = shape.getRowMajorStrides();
  strides.pop_back();
  std::reverse(strides.begin(), strides.end());

  // The lines of the final string representation:
  std::vector<std::vector<std::string>> rows;
  rows.reserve(shape.rank_u64());
  for (uint64_t i = 0; i < stringElements1d.size(); ++i) {
    if (i % static_cast<uint64_t>(strides[0]) == 0) {
      uint64_t c = i == 0;
      for (auto s : strides) {
        if (i % static_cast<uint64_t>(s) == 0) {
          ++c;
        }
      }
      if (i != 0) {
        rows.back().push_back(std::string(c, ']'));
      }
      rows.push_back({});
      auto opening = std::string(shape.rank_u64() - c, ' ');
      opening.resize(shape.rank_u64(), '[');
      rows.back().push_back(opening);
    }
    rows.back().push_back(stringElements1d[i]);
  }
  rows.back().push_back(std::string(shape.rank_u64(), ']'));

  // pad the strings, for a cleaner vertical alignment:
  for (uint64_t i = 0; i < rows[0].size(); ++i) {
    uint64_t m = 0;
    for (uint64_t j = 0; j < rows.size(); ++j) {
      m = std::max<uint64_t>(m, rows[j][i].size());
    }
    for (uint64_t j = 0; j < rows.size(); ++j) {
      rows[j][i].resize(m + 1, ' ');
    }
  }

  for (auto r : rows) {
    for (auto v : r) {
      outStream << v;
    }
    outStream << '\n';
  }
}

} // namespace host
} // namespace compute
} // namespace poprithms
