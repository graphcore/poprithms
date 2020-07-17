// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>

#include <poprithms/util/error.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/shape.hpp>

namespace poprithms {
namespace util {

void Shape::assertValidDimension(uint64_t d) const {
  if (d >= rank_u64()) {
    std::ostringstream oss;
    oss << "Failure in assertValidDimension for shape=" << *this
        << ", failure with invalid dimension= " << d;
    throw error(oss.str());
  }
}

std::vector<int64_t>
Shape::concatPartitionPoints(const std::vector<Shape> &inShapes,
                             uint64_t axis) {
  std::vector<int64_t> ps(1, 0LL);
  for (const auto &s : inShapes) {
    ps.push_back(ps.back() + s.dim(axis));
  }
  return ps;
}

Shape Shape::squeeze() const {
  std::vector<int64_t> squeezed;
  for (const auto s : get()) {
    if (s != 1) {
      squeezed.push_back(s);
    }
  }
  return squeezed;
}

Shape Shape::broadcast(int64_t N, uint64_t dimension) const {
  assertValidDimension(dimension);
  auto s = get();
  s[dimension] *= N;
  return s;
}

Shape Shape::unsqueeze(uint64_t d) const {
  assertValidDimension(d);
  auto s = get();
  s.insert(std::next(s.begin(), d), 1LL);
  return s;
}

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

namespace {
template <class T> Shape fromPartials(const T &t) {
  std::vector<int64_t> shp_(t.size());
  for (uint64_t d = 0; d < t.size(); ++d) {
    shp_[d] = static_cast<int64_t>(t[d].size());
  }
  return Shape(shp_);
}
} // namespace

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

Shape Shape::concat(const Shape &rhs, uint64_t axis) const {
  return concat({*this, rhs}, axis);
}

Shape Shape::concat(const std::vector<Shape> &inShapes, uint64_t concatAxis) {
  if (inShapes.empty()) {
    throw error("Empty vector of Shapes in Shape::concat");
  }
  const auto &s0 = inShapes[0];
  if (s0.rank_u64() <= concatAxis) {
    std::ostringstream oss;
    oss << "Invalid rank of Tensor (" << s0.rank_u64()
        << ") in Shape::concat.with concatAxis=" << concatAxis;
    throw error(oss.str());
  }
  auto concatAxisSize = s0.dim(concatAxis);
  for (auto iter = std::next(inShapes.cbegin()); iter < inShapes.cend();
       ++iter) {
    const auto &i1 = *iter;
    s0.assertConcattable(i1, concatAxis);
    concatAxisSize += i1.dim(concatAxis);
  }

  auto outShape        = s0.get();
  outShape[concatAxis] = concatAxisSize;
  return outShape;
}

bool Shape::concattable(const Shape &rhs, uint64_t axis) const {
  if (axis >= rank_u64()) {
    return false;
  }
  if (rhs.rank_u64() != rank_u64()) {
    return false;
  }
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    if (d != axis && dim(d) != rhs.dim(d)) {
      return false;
    }
  }
  return true;
}

void Shape::assertConcattable(const Shape &rhs, uint64_t axis) const {
  if (!concattable(rhs, axis)) {
    std::ostringstream oss;
    oss << "Failure in Shape::assertConcattable, with axis = " << axis
        << ". The Shapes are :"
        << "\n      " << *this << "\n      " << rhs << '.';
    throw error(oss.str());
  }
}

Shape Shape::slice(const Lower &l, const Upper &u) const {
  assertBoundsAreValid(l, u);
  std::vector<int64_t> out(rank_u64(), 0);
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    out[d] = u[d] - l[d];
  }
  return Shape(out);
}

void Shape::assertBoundsAreValid(const Lower &l, const Upper &u) const {

  std::ostringstream ss;

  // same rank for lower and upper
  if (l.size() != u.size() || u.size() != rank_u64()) {
    ss << "lower and upper must both be of size "
       << " " << rank_u64() << ". This ia not true for lower=" << l
       << " and upper=" << u << '.';
    throw error(ss.str());
  }

  // lower less than or equal to upper
  for (auto i = 0ul; i < rank_u64(); ++i) {
    if (l[i] > u[i]) {
      ss << "lower bound cannot excede upper bound. "
         << "This for lower=" << l << " and upper=" << u << '.';
      throw error(ss.str());
    }

    if (dim(i) < u[i]) {
      ss << "lower bound cannot excede upper bound. "
         << "This for lower=" << l << " and upper=" << u << '.';
      throw error(ss.str());
    }
  }
}

} // namespace util
} // namespace poprithms
