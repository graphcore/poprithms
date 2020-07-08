// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>

#include <poprithms/util/error.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/shape.hpp>

namespace poprithms {
namespace util {

int64_t Shape::nelms() const {
  return std::accumulate(
      shp.begin(), shp.end(), int64_t(1), std::multiplies<int64_t>());
}

void Shape::assertFlatPoint(int64_t flatPoint) const {
  if (flatPoint >= nelms() || flatPoint < 0) {
    std::ostringstream oss;
    oss << "In Shape::assertFloatPoint, for " << *this;
    oss << ", and flatPoint = " << flatPoint
        << ". Expected flatPoint in range [0, nelms=" << nelms() << ").";
    throw error(oss.str());
  }
}

std::vector<int64_t> Shape::getRowMajorIndices(
    const std::vector<std::vector<int64_t>> &subIndices) const {
  if (rank_u64() != subIndices.size()) {
    std::ostringstream oss;
    oss << "In Shape::getRowMajorIndices for " << *this
        << ", invalid input of size " << subIndices.size() << '.';
    throw error(oss.str());
  }
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    for (const auto &y : subIndices[d]) {
      if (y >= shp[d]) {
        std::ostringstream oss;
        oss << " In Shape::getRowMajorIndices for " << *this << ". With dim("
            << d << ")=" << dim(d) << ", there is a value in subIndices[" << d
            << "] of " << y
            << ", which is not valid, it must be less than dim(" << d << ").";
        throw error(oss.str());
      }
    }
  }

  const auto outShape = fromPartials(subIndices);
  const auto nOutElms = outShape.nelms_u64();

  std::vector<int64_t> inds{0};
  std::vector<int64_t> prevInds;

  inds.reserve(nOutElms);
  prevInds.reserve(nOutElms);

  int64_t stride = 1;
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    auto dim = rank_u64() - d - 1;
    std::swap(inds, prevInds);
    inds.clear();
    for (const auto x : subIndices[dim]) {
      for (const auto y : prevInds) {
        inds.push_back(stride * x + y);
      }
    }
    stride *= shp[dim];
  }
  return inds;
}

std::vector<int64_t> Shape::getRowMajorPoint(int64_t flatPoint) const {
  assertFlatPoint(flatPoint);

  std::vector<int64_t> point(rank_u64(), 0);
  if (flatPoint == 0) {
    return point;
  }

  auto rem = flatPoint;
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    auto pi   = rank_u64() - d - 1;
    point[pi] = rem % shp[pi];
    rem /= shp[pi];
  }
  return point;

  // Example : {5, 7}, 20 : returns {2,6}
}

int64_t Shape::getRowMajorIndex(const std::vector<int64_t> &indices) const {
  const auto strides = getRowMajorStrides();
  return std::inner_product(
      strides.cbegin(), strides.cend(), indices.cbegin(), 0LL);
}

std::vector<int64_t> Shape::getRowMajorStrides() const {
  // Example   :: sh={5, 2, 3} --> strides={6, 3, 1}
  // Example   :: sh={}        --> strides={}
  // Example   :: sh={1744}    --> strides={1}
  // Example   :: sh={1,2,3,4} --> strides={24, 12, 4, 1}

  if (rank_u64() == 0ull) {
    return {};
  }

  std::vector<int64_t> strides(rank_u64(), 1);
  std::partial_sum(shp.rbegin(),
                   shp.rend() - 1,
                   strides.rbegin() + 1,
                   std::multiplies<int64_t>());
  return strides;
}

std::vector<int64_t> Shape::getColMajorStrides() const {
  auto sh2 = *this;
  std::reverse(sh2.shp.begin(), sh2.shp.end());
  auto strides = sh2.getRowMajorStrides();
  std::reverse(strides.begin(), strides.end());
  return strides;
}

Shape Shape::numpyBinary(const Shape &rhs) const {
  assertNumpyBroadcastable(shp, rhs.shp);
  return numpyBinary(shp, rhs.shp);
}

void Shape::assertNumpyBroadcastable(const std::vector<int64_t> &a,
                                     const std::vector<int64_t> &b) {
  bool aIsLonger      = a.size() > b.size();
  const auto &longer  = aIsLonger ? a : b;
  const auto &shorter = aIsLonger ? b : a;
  const auto delta    = longer.size() - shorter.size();
  for (auto i = delta; i < longer.size(); ++i) {
    if (shorter[i - delta] != longer[i] && shorter[i - delta] != 1 &&
        longer[i] != 1) {
      std::ostringstream oss;
      oss << "Failure in "
          << "Shape::assertNumpyBroadcastable, "
          << "with a=" << a << " and b=" << b << ". "
          << "Failed at index " << i << '.';
      throw error(oss.str());
    }
  }
}

std::vector<bool> Shape::numpyWhereToExpand(const Shape &targetShape) const {

  if (rank_u64() > targetShape.rank_u64()) {
    std::ostringstream oss;
    oss << "`from' larger than `to' in Shape::numpyWhereToExpand for "
        << *this << ", where targetShape=" << targetShape << '.';
    throw error(oss.str());
  }

  const auto &from = shp;
  const auto &to   = targetShape.shp;
  const auto delta = to.size() - from.size();

  std::vector<bool> wh;
  wh.reserve(from.size());

  for (uint64_t i = 0; i < from.size(); ++i) {

    // no expansion
    if (from[i] == to[i + delta]) {
      wh.push_back(false);
    }

    // expansion
    else if (from[i] == 1) {
      wh.push_back(true);
    }

    else {
      std::ostringstream oss;
      oss << "Invalid arguments in "
          << "Shape::numpyWhereToExpand, where from=" << *this
          << " and to=" << targetShape << " : not numpy-expandable.";
      throw error(oss.str());
    }
  }
  return wh;
}

void Shape::append(std::ostream &os) const {
  poprithms::util::append(os, shp);
}

std::ostream &operator<<(std::ostream &os, const Shape &sh) {
  sh.append(os);
  return os;
}

} // namespace util
} // namespace poprithms
