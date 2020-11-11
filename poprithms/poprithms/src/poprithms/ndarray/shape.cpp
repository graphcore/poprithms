// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>
#include <sstream>

#include <poprithms/ndarray/error.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace ndarray {

namespace {
std::ostream &operator<<(std::ostream &ost, const std::vector<uint64_t> &x) {
  util::append(ost, x);
  return ost;
}
} // namespace

uint64_t Shape::dimProduct_u64(int64_t l, int64_t u) const {
  return static_cast<uint64_t>(dimProduct(l, u));
}

Shape Shape::unsqueeze(const std::vector<uint64_t> &dims) const {
  const auto R0 = dims.size() + rank_u64();
  std::vector<int64_t> outShape(R0, -1);

  // Set outShape to be 1 in all dimensions in dims.
  for (auto oneDim : dims) {
    if (oneDim >= R0) {
      std::ostringstream oss;
      oss << "Invalid dimensions, " << *this << ".unsqueeze(dims=" << dims
          << "). The output shape has rank_u64() + dims.size() = "
          << rank_u64() << " + " << dims.size() << " = " << R0
          << ". Therefore, the dimension " << oneDim << "is too large. ";
      throw error(oss.str());
    }
    if (outShape[oneDim] == 1) {
      std::ostringstream oss;
      oss << "Invalid dimensions, " << *this << ".unsqueeze(dims=" << dims
          << "). Dimensions must be unique. ";
      throw error(oss.str());
    }
    outShape[oneDim] = 1;
  }

  // Fill in the dimensions from this Shape
  uint64_t index{0};
  for (auto d : get()) {
    while (outShape[index] == 1) {
      ++index;
    }
    outShape[index] = d;
    ++index;
  }

  return Shape(outShape);
}

void Shape::assertValidDimension(uint64_t d) const {
  if (d >= rank_u64()) {
    std::ostringstream oss;
    oss << "Failure in assertValidDimension for shape=" << *this
        << ", failure with invalid dimension= " << d;
    throw error(oss.str());
  }
}

Shape Shape::prepend(int64_t dim0) const {
  decltype(shp) prepended{dim0};
  prepended.insert(prepended.cend(), shp.cbegin(), shp.cend());
  return prepended;
}

std::vector<int64_t> Shape::getCustomStridedRowMajorIndices(
    const std::vector<int64_t> &strides) const {
  std::vector<int64_t> out(nelms_u64(), 0);
  uint64_t nToCopy = 1;
  for (uint64_t d_ = rank_u64(); d_ != 0; --d_) {
    const auto d      = d_ - 1;
    const auto stride = strides[d];
    for (uint64_t copyNumber = 1; copyNumber < dim_u64(d); ++copyNumber) {
      const auto delta = stride * copyNumber;
      for (uint64_t localIndex = 0; localIndex < nToCopy; ++localIndex) {
        out[copyNumber * nToCopy + localIndex] = out[localIndex] + delta;
      }
    }
    nToCopy *= dim(d);
  }
  return out;
}

std::vector<std::array<Shape, 2>>
Shape::getPadShapes(const std::vector<uint64_t> &l,
                    const std::vector<uint64_t> &u) const {

  const auto R0 = rank_u64();

  if (l.size() != R0 || u.size() != R0) {
    std::ostringstream oss;
    oss << "In getPadShapes for Shape " << *this << ", which is of rank "
        << R0 << ". The lower and upper paddings must "
        << "be of the same rank. But l=" << l << " has size " << l.size()
        << ", and u=" << u << " has size " << u.size() << '.';
    throw error(oss.str());
  }

  std::vector<std::array<Shape, 2>> shapes;
  shapes.reserve(R0);
  auto current = get();
  for (uint64_t d = 0; d < R0; ++d) {
    auto lowPad = current;
    lowPad[d]   = l[d];
    auto uppPad = current;
    uppPad[d]   = u[d];
    shapes.push_back({lowPad, uppPad});
    current[d] = current[d] + l[d] + u[d];
  }

  return shapes;
}

std::vector<int64_t>
Shape::getExpandedRowMajorIndices(const Shape &to) const {
  if (rank_u64() < to.rank_u64()) {
    auto prepadded = std::vector<int64_t>(to.rank_u64() - rank_u64(), 1);
    prepadded.insert(prepadded.end(), shp.cbegin(), shp.cend());
    return Shape(prepadded).getExpandedRowMajorIndices(to);
  }
  auto strides     = getRowMajorStrides();
  const auto where = numpyWhereToExpand(to);
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    if (where[d]) {
      strides[d] = 0;
    }
  }
  return to.getCustomStridedRowMajorIndices(strides);
}

std::vector<int64_t>
Shape::getDimShuffledRowMajorIndices(const Permutation &p) const {
  return dimShuffle(p).getCustomStridedRowMajorIndices(
      p.apply(getRowMajorStrides()));
}

Shape Shape::dimShuffle(const Permutation &p) const {
  return {p.apply(get())};
}

std::vector<Shape::ConcatSource>
Shape::getRowMajorConcatSources(const Shapes &shapes, uint64_t axis) {

  const auto outShape = concat(shapes, axis);
  std::vector<Shape::ConcatSource> out(outShape.nelms_u64());
  const auto axis_i64 = static_cast<int64_t>(axis);

  // The number of times you loop through each source Shape.
  // This is 1 if the axis of concatenation is 0: each input Shape contributes
  // to a contiguous region of the output Shape.
  const auto nCopies = outShape.dimProduct_u64(0, axis_i64);

  std::vector<uint64_t> nContigs;
  nContigs.reserve(shapes.size());
  for (const auto &inShape : shapes) {
    nContigs.push_back(inShape.dimProduct_u64(axis_i64, inShape.rank_i64()));
  }

  uint64_t outIndex = 0;
  for (uint64_t i = 0; i < nCopies; ++i) {
    for (uint64_t shapeIndex = 0; shapeIndex < shapes.size(); ++shapeIndex) {
      for (uint64_t k = 0; k < nContigs[shapeIndex]; ++k) {
        auto inRowMajorIndex =
            static_cast<int64_t>(i * nContigs[shapeIndex] + k);
        out[outIndex] = {shapeIndex, inRowMajorIndex};
        ++outIndex;
      }
    }
  }
  return out;
}

std::vector<int64_t>
Shape::getRowMajorBlockOrdered(const Shape &blockShape) const {
  if (blockShape.rank_u64() != rank_u64()) {
    std::ostringstream oss;
    oss << "blockShape has rank " << blockShape.rank_u64()
        << " but this Shape has rank " << rank_u64()
        << ". They should be the same. ";
    throw error(oss.str());
  }

  for (auto l : blockShape.get()) {
    if (l < 1) {
      std::ostringstream oss;
      oss << "blockShape=" << blockShape;
      oss << ", all elements must be strictly positive.";
      throw error(oss.str());
    }
  }

  // The number of blocks in each dimension
  std::vector<int64_t> blocksPerDim;
  blocksPerDim.reserve(rank_u64());
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    blocksPerDim.push_back(dim(d) / blockShape.dim(d) +
                           (dim(d) % blockShape.dim(d) != 0));
  }
  const Shape blocks(blocksPerDim);

  std::vector<int64_t> blockOrdered;
  blockOrdered.reserve(nelms_u64());

  for (int64_t blockId = 0; blockId < blocks.nelms(); ++blockId) {
    const auto blockCoordinate = blocks.getRowMajorPoint(blockId);
    std::vector<int64_t> lower;
    lower.reserve(rank_u64());
    std::vector<int64_t> upper;
    upper.reserve(rank_u64());
    for (uint64_t d_ = 0; d_ < rank_u64(); ++d_) {
      lower.push_back(blockShape.dim(d_) * blockCoordinate[d_]);
      upper.push_back(std::min(dim(d_), lower.back() + blockShape.dim(d_)));
    }
    const auto nxt = getSlicedRowMajorIndices(lower, upper);
    blockOrdered.insert(blockOrdered.end(), nxt.cbegin(), nxt.cend());
  }
  return blockOrdered;
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

Shape Shape::squeeze(const std::vector<uint64_t> &dims) const {

  std::vector<bool> retain(rank_u64(), true);

  for (auto d : dims) {
    if (d >= rank_u64()) {
      std::ostringstream oss;
      oss << "Error in squeezing Shape " << *this << ". Dimension " << d
          << " exceeds permissible range end.";
      throw error(oss.str());
    }
    if (dim(d) != 1) {
      std::ostringstream oss;
      oss << "Error in squeezing Shape " << *this
          << ". The size of dimension " << d << " is " << dim(d)
          << ", but you can only squeeze on size-1 dimensions.";
      throw error(oss.str());
    }
    retain[d] = false;
  }
  std::vector<int64_t> squeezed;

  // There may be duplicates in dims. We reserve to optimize the case where
  // there are no duplicates.
  squeezed.reserve(rank_u64() - std::min<size_t>(dims.size(), rank_u64()));
  for (uint64_t i = 0; i < rank_u64(); ++i) {
    if (retain[i]) {
      squeezed.push_back(dim(i));
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

Shape Shape::resizeSingleDim(int64_t N, uint64_t dimension) const {
  assertValidDimension(dimension);
  auto s       = get();
  s[dimension] = N;
  return s;
}

Shape Shape::unsqueeze(uint64_t d) const {
  assertValidDimension(d);
  auto s = get();
  s.insert(std::next(s.cbegin(), d), 1LL);
  return s;
}

int64_t Shape::nelms() const {
  return std::accumulate(
      shp.cbegin(), shp.cend(), int64_t(1), std::multiplies<int64_t>());
}

int64_t Shape::dimProduct(int64_t l, int64_t u) const {
  return std::accumulate(shp.cbegin() + l,
                         shp.cbegin() + u,
                         int64_t(1),
                         std::multiplies<int64_t>());
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
  auto strides = reverse().getRowMajorStrides();
  std::reverse(strides.begin(), strides.end());
  return strides;
}

Shape Shape::reverse() const {
  auto s = get();
  std::reverse(s.begin(), s.end());
  return {std::move(s)};
}

std::vector<int64_t> Shape::getSlicedRowMajorIndices(const Lower &l,
                                                     const Upper &u) const {

  const auto outShape  = slice(l, u);
  const auto rmStrides = getRowMajorStrides();

  std::vector<int64_t> indices{0};
  indices.reserve(outShape.nelms_u64());

  std::vector<int64_t> nextIndices;
  nextIndices.reserve(outShape.nelms_u64());

  for (uint64_t d_ = rank_u64(); d_ != 0; --d_) {
    const auto d = d_ - 1;
    for (int64_t c = l[d]; c < u[d]; ++c) {
      const auto delta = c * rmStrides[d];
      for (auto i : indices) {
        nextIndices.push_back(i + delta);
      }
    }
    std::swap(indices, nextIndices);
    nextIndices.clear();
  }
  return indices;
}

std::vector<int64_t>
Shape::gatherRowMajorIndices(uint64_t dimension,
                             const std::vector<int64_t> &where) const {
  if (dimension >= rank_u64()) {
    std::ostringstream oss;
    oss << "Invalid dimension (" << dimension << ") for Shape " << *this
        << ", dimension must be less than rank (" << rank_u64() << ").";
    throw error(oss.str());
  }

  for (auto w : where) {
    if (w >= dim(dimension)) {
      std::ostringstream oss;
      oss << "Invalid gather index (" << w << ") along dimension "
          << dimension << ", for Shape " << *this << ", which has size "
          << dim(dimension) << " along dimension " << dimension
          << ". Gather indices must be less than this, but " << w
          << " >= " << dim(dimension) << ". ";
      throw error(oss.str());
    }
  }

  const auto outerSize = dimProduct(0, dimension);
  const auto innerSize = dimProduct(dimension + 1, rank_u64());

  std::vector<int64_t> indices;
  indices.reserve(outerSize * innerSize * dim(dimension));

  for (int64_t outerIndex = 0; outerIndex < outerSize; ++outerIndex) {
    for (auto w : where) {
      int64_t baseIndex = (w + dim(dimension) * outerIndex) * innerSize;
      for (int64_t innerIndex = 0; innerIndex < innerSize; ++innerIndex) {
        indices.push_back(baseIndex + innerIndex);
      }
    }
  }

  return indices;
}

std::vector<int64_t>
Shape::gatherColMajorIndices(uint64_t dimension,
                             const std::vector<int64_t> &where) const {
  return reverse().gatherRowMajorIndices(rank_u64() - dimension - 1, where);
}

std::vector<int64_t> Shape::getSlicedColMajorIndices(const Lower &l,
                                                     const Upper &u) const {
  auto l2 = l;
  std::reverse(l2.begin(), l2.end());
  auto u2 = u;
  std::reverse(u2.begin(), u2.end());
  return reverse().getSlicedRowMajorIndices(l2, u2);
}

Shape Shape::numpyBinary(const Shape &rhs) const {
  assertNumpyBroadcastable(shp, rhs.shp);
  return numpyBinary(shp, rhs.shp);
}

Shape Shape::numpyVariadic(const std::vector<Shape> &shapes) {
  if (shapes.empty()) {
    throw error("Empty container of shapes not allowed in numpyVariadic");
  }
  return std::accumulate(
      shapes.cbegin(),
      shapes.cend(),
      shapes[0],
      [](const auto &a, const auto &b) { return a.numpyBinary(b); });
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

void Shape::assertCanExpandTo(const Shape &to) const {
  // This call does all the work required to generate a clean error message if
  // this Shape cannot be expanded to "to".
  const auto indices = numpyWhereToExpand(to);
  (void)indices;
}

void Shape::assertSameNumberOfElements(const Shape &rhs) const {
  if (nelms_u64() != rhs.nelms_u64()) {
    std::ostringstream oss;
    oss << "Failed in Shape::assertSameNumberOfElements, "
        << "where this Shape is " << *this << " with " << nelms_u64()
        << " elements, and rhs is " << rhs << ", with " << rhs.nelms_u64()
        << " elements. ";
    throw error(oss.str());
  }
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
        << ") in Shape::concat.with concatAxis=" << concatAxis
        << " and shape = \n      { ";
    for (const auto &x : inShapes) {
      oss << x << ' ';
    }
    oss << '}';

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
      ss << "lower bound cannot exceed upper bound. "
         << "This for lower=" << l << " and upper=" << u << '.';
      throw error(ss.str());
    }

    if (dim(i) < u[i]) {
      ss << "Upper bound cannot exceed dimension size (in dimension " << i
         << ") "
         << "This for Shape = " << *this << ", lower=" << l
         << " and upper=" << u << '.';
      throw error(ss.str());
    }
  }
}

Shape Shape::matmul(const Shape &a, const Shape &b) {

  const auto aRank = a.rank_u64();
  const auto bRank = b.rank_u64();

  if (aRank == 0 || bRank == 0) {
    std::ostringstream oss;
    oss << "rank-0 Shape not allowed in Shape::matmul: "
        << " a = " << a << ", b = " << b << '.';
    throw error(oss.str());
  }

  // If the first argument is 1-D, it is promoted to a matrix by prepending a
  // 1 to its dimensions. After matrix multiplication the prepended 1 is
  // removed.
  if (aRank == 1) {
    auto o        = matmul({{1, a.dim(0)}}, b).get();
    const auto bv = o.back();
    o.pop_back();
    if (!o.empty()) {
      o.back() = bv;
    }
    return o;
  }

  //  If the second argument is 1-D, it is promoted to a matrix by appending a
  //  1 to its dimensions. After matrix multiplication the appended 1 is
  //  removed.
  if (bRank == 1) {
    auto o = matmul(a, {b.dim(0), 1}).get();
    o.pop_back();
    return o;
  }

  //  If either argument is N-D, N > 2, it is treated as a stack of matrices
  //  residing in the last two indexes and broadcast accordingly.
  if (a.get().back() != *(b.get().cend() - 2)) {
    std::ostringstream oss;
    oss << "Reduction dimension sizes do not agree in matmul, a = " << a
        << ", b = " << b << " (" << a.get().back()
        << " != " << *(b.get().cend() - 2) << ").";
    throw error(oss.str());
  }

  // numpy shape broadcasting:
  auto outShape = Shape{{a.get().cbegin(), a.get().cend() - 2}}
                      .numpyBinary({{b.get().cbegin(), b.get().cend() - 2}})
                      .get();

  // actual matmul shape:
  outShape.push_back(*(a.get().cend() - 2));
  outShape.push_back(b.get().back());
  return outShape;
}

Shape Shape::flattenTo2d(uint64_t axis) const {
  if (axis > rank_u64()) {
    std::ostringstream oss;
    oss << "Invalid axis (" << axis << ") in flattenTo2d, for Shape " << *this
        << " which is of rank " << rank_u64() << ". "
        << "axis must in range [0, " << axis << "].";
    throw error(oss.str());
  }
  return {dimProduct(0, axis), dimProduct(axis, rank_u64())};
}

} // namespace ndarray
} // namespace poprithms
