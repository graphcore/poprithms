// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <numeric>
#include <ostream>
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

Shape Shape::batchedMultiChannelConvolve(
    const Shape &kernel,
    const std::vector<uint64_t> &lowPrePads,
    const std::vector<uint64_t> &uppPrePads,
    const Dilations &dilations,
    const Strides &strides) const {

  if (rank_u64() < 2 || kernel.rank_u64() < 1) {
    std::ostringstream oss;
    oss << "Invalid ranks in batchedMultiChannelConvolve. "
        << "This (data) Shape is " << *this << ", and the Shape of kernel is "
        << kernel << ". This must be rank>=2 and kernel must be rank>=1. ";
    throw error(oss.str());
  }

  // 1) spatial convolution of the suffixes of this and kernel
  // 2) prepend batch-size, and output channels.
  return fromDim(2)
      .convolve(kernel.fromDim(1), lowPrePads, uppPrePads, dilations, strides)
      .prepend({dim(0), kernel.dim(0)});
}

Shape Shape::pool(const Shape &window,
                  const std::vector<uint64_t> &lowPrePads,
                  const std::vector<uint64_t> &uppPrePads,
                  const Dilations &dilations,
                  const Strides &strides,
                  const RoundMode m) const {

  const auto R0 = rank_u64();

  // For all empty inputs, give them the default values:

  auto window_ = window.rank_u64() == 0 ? singleton(R0) : window;

  const auto lowPrePads_ =
      lowPrePads.empty() ? std::vector<uint64_t>(R0, 0) : lowPrePads;

  const auto uppPrePads_ =
      uppPrePads.empty() ? std::vector<uint64_t>(R0, 0) : uppPrePads;

  const auto dilations_ =
      dilations.empty() ? Dilations(std::vector<uint64_t>(R0, 1)) : dilations;

  const auto strides_ =
      strides.empty() ? Strides(std::vector<uint64_t>(R0, 1)) : strides;

  // Check that all inputs now have the correct rank:

  if (window_.rank_u64() != R0 || lowPrePads_.size() != R0 ||
      uppPrePads_.size() != R0 || dilations_.size() != R0 ||
      strides_.size() != R0) {
    std::ostringstream oss;
    oss << "Invalid inputs to pool for this Shape " << *this
        << ", which has rank " << rank_u64()
        << ". Expected all inputs to have either size 0, or size "
        << rank_u64() << ". But the sizes of the inputs are:"
        << "\n    window    : " << window.rank_u64()
        << "\n    lowPrePads: " << lowPrePads.size()
        << "\n    uppPrePads: " << uppPrePads.size()
        << "\n    dilations : " << dilations.size()
        << "\n    strides   : " << strides.size() << '.' << " The window is "
        << window << ".";
    throw error(oss.str());
  }

  // populate dimension by dimension:

  std::vector<int64_t> out;
  out.reserve(R0);
  for (uint64_t d = 0; d < R0; ++d) {
    out.push_back(pool1d(dim(d),
                         window_.dim(d),
                         lowPrePads_.at(d),
                         uppPrePads_.at(d),
                         dilations_.at(d),
                         strides_.at(d),
                         m));
  }
  return out;
}

uint64_t Shape::pool1d(uint64_t data,
                       uint64_t window,
                       uint64_t lowPad,
                       uint64_t uppPad,
                       Dilation dilation,
                       Stride stride,
                       RoundMode m) {

  /**
   * Example 0 (most basic case)
   * --> data = 10, window=1, padSum = 0, dilation=1, stride=0
   *     out = 10
   *
   * Example 2 (scales as - dilation * (window - 1) when stride = 1)
   * --> data = 10, window=3, padSum = 0, dilation=1, stride=0
   *     out = 8
   *
   *     *..*..* (window = 3, dilation = 3)
   * --> data = 7, window=3, padSum = 0, dilation=3, stride=0
   *     out = 1
   *
   *
   * Example 3 (After dilation, striding is applied)
   * dddddd
   * kkk      (first out)
   *   kkk    (second out)
   *     kkk  (third out, only if RoundMode::Ceil)
   *
   * --> data = 6, window=3, padSum = 0, dilation=1, stride=2,
   *     RoundMode::Floor
   *     out = 2
   *
   * --> data = 6, window=3, padSum = 0, dilation=1, stride=2,
   *
   * RoundMode::Floor
   *     out = 3
   * */

  const auto padSum = lowPad + uppPad;

  auto x0 = data + padSum - 1 - dilation.val * (window - 1);
  x0 /= stride.get();
  x0 += (x0 % stride.get() == 0) * (m == RoundMode::Ceil) + 1;
  return x0;
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

Shape Shape::prepend(const Shape &dims0) const {
  auto outShape_    = dims0.get();
  const auto suffix = get();
  outShape_.insert(outShape_.end(), suffix.cbegin(), suffix.cend());
  return outShape_;
}

Shape Shape::append(int64_t dimEnd) const & {
  auto outShape_ = get();
  outShape_.push_back(dimEnd);
  return outShape_;
}

Shape &&Shape::append(int64_t dimEnd) && {
  shp.push_back(dimEnd);
  return std::move(*this);
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

Shape Shape::dimRoll(Dimension dimIdx, Dimension newIdx) const {
  return dimShuffle(
      Permutation::dimRoll(rank_u64(), {dimIdx.get(), newIdx.get()}));
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

Shape Shape::broadcast(int64_t N, uint64_t dimension) const & {
  assertValidDimension(dimension);
  auto s = get();
  s[dimension] *= N;
  return s;
}

Shape &&Shape::broadcast(int64_t N, uint64_t dimension) && {
  assertValidDimension(dimension);
  shp[dimension] *= N;
  return std::move(*this);
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

namespace {
std::vector<int64_t> getIndices(int64_t start,
                                const std::vector<int64_t> &strides,
                                const Shape &outShape) {

  std::vector<int64_t> indices{start};
  indices.reserve(outShape.nelms_u64());

  std::vector<int64_t> nextIndices;
  nextIndices.reserve(outShape.nelms_u64());

  for (uint64_t d_ = outShape.rank_u64(); d_ != 0; --d_) {
    const auto d = d_ - 1;
    for (uint64_t c = 0; c < outShape.dim_u64(d); ++c) {
      const auto delta = c * strides[d];
      for (auto i : indices) {
        nextIndices.push_back(i + delta);
      }
    }
    std::swap(indices, nextIndices);
    nextIndices.clear();
  }
  return indices;
}
} // namespace

std::vector<int64_t>
Shape::getReducedRowMajorIndices(const Shape &outShape_) const {

  assertCanReduceTo(outShape_);

  if (rank_u64() == 0) {
    return {0};
  }

  // prepend 1's, to bump the rank of the output Shape up to rank of this
  // Shape.
  const auto outShape =
      outShape_.prepend(Shape::singleton(rank_u64() - outShape_.rank_u64()));

  const auto outStrides = outShape.getRowMajorStrides();
  const auto inStrides  = getRowMajorStrides();

  // Example:
  //
  //   4 5 3 2  this Shape
  //      |
  //      v
  //   5 1 3 1  outShape.
  //

  std::vector<int64_t> indices{0};
  std::vector<int64_t> prevIndices;

  // Reserving the maximum number of elements indices will ever have.
  indices.reserve(nelms_u64());

  // Reserving the maximum number of elements prevIndices will ever have.
  prevIndices.reserve(dimProduct_u64(1, rank_i64()));

  for (uint64_t d_ = rank_u64(); d_ != 0; --d_) {
    const auto d = d_ - 1;

    std::swap(indices, prevIndices);
    indices.clear();
    for (uint64_t k = 0; k < dim_u64(d); ++k) {

      // If the axis d is a reduction axis, then elements in the input (this
      // Shape) get mapped to elements in the outShape independently of k.
      const auto delta = outShape.dim(d) == 1 ? 0 : outStrides[d] * k;

      for (auto p : prevIndices) {
        indices.push_back(p + delta);
      }
    }
  }

  return indices;
}

std::vector<int64_t> Shape::getSlicedRowMajorIndices(const Lower &l,
                                                     const Upper &u) const {
  assertSliceBoundsAreValid(l, u);
  int64_t start{0};
  auto strides = getRowMajorStrides();
  for (uint64_t i = 0; i < rank_u64(); ++i) {
    start += strides[i] * l[i];
  }
  return getIndices(start, strides, slice(l, u));
}

std::vector<int64_t> Shape::getSubSampledRowMajorIndices(
    const std::vector<uint64_t> &strides) const {
  const auto outShape = subSample(strides);
  auto rms            = getRowMajorStrides();
  for (uint64_t i = 0; i < rank_u64(); ++i) {
    rms[i] *= strides[i];
  }
  const auto indices = getIndices(0, rms, outShape);
  return indices;
}

std::vector<int64_t>
Shape::getSlicedRowMajorIndices(const NormalizedSliceParams &n) const {

  // initialize strides as if steps were are +1
  auto strides = getRowMajorStrides();

  // itialize start as if starts were all 0.
  int64_t start{0};

  for (uint64_t i = 0; i < rank_u64(); ++i) {
    start += strides[i] * n.start(i);
    strides[i] *= n.step(i);
  }
  return getIndices(start, strides, slice(n));
}

std::vector<int64_t>
Shape::getReversedRowMajorIndices(const std::vector<uint64_t> &dims) const {
  auto strides = getRowMajorStrides();
  std::vector<bool> mustReverse(rank_u64(), false);
  for (auto d : dims) {
    if (d >= rank_u64()) {
      std::ostringstream oss;
      oss << "Invalid dimension '" << d
          << "' in getReversedRowMajorIndices for Shape " << *this
          << ". Expected all dimensions to be less than the rank, "
          << rank_u64() << ". Invalid call " << *this
          << ".getReversedRowMajorIndices(dims=";
      poprithms::util::append(oss, dims);
      oss << ").";
      throw error(oss.str());
    }
    mustReverse[d] = !mustReverse[d];
  }

  if (nelms() == 0) {
    return {};
  }

  int64_t start{0};
  for (uint64_t i = 0; i < rank_u64(); ++i) {
    if (mustReverse[i]) {
      start += (dim(i) - 1) * strides[i];
      strides[i] *= -1;
    }
  }
  return getIndices(start, strides, *this);
}

Shape Shape::subSample(const std::vector<uint64_t> &strides) const {

  if (strides.size() != rank_u64()) {
    std::ostringstream oss;
    oss << "Invalid call in Shape::subSample, " << *this
        << ".subSample(strides = ";
    util::append(oss, strides);
    oss << "). strides' length should be the same as the rank of this Shape, "
        << strides.size() << " != " << rank_u64() << '.';
    throw error(oss.str());
  }

  auto inShape = get();
  decltype(inShape) outShape;
  outShape.reserve(rank_u64());
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    if (strides[d] < 1) {
      std::ostringstream oss;
      oss << "Invalid strides in " << *this << ".subSample(strides = ";
      util::append(oss, strides);
      oss << "). All stride values must be strictly greater than 0. ";
      throw error(oss.str());
    }
    outShape.push_back(inShape[d] / strides[d] +
                       (inShape[d] % strides[d] != 0));
  }
  return outShape;
}

void Shape::validateGatherIndices(uint64_t dimension,
                                  const std::vector<int64_t> &where) const {
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

  validateGatherIndices(dimension, where);

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

std::vector<int64_t> Shape::gatherRowMajorIndices(
    const std::vector<std::vector<int64_t>> &where) const {

  if (where.size() != rank_u64()) {
    std::ostringstream oss;
    oss << "Invalid vector `where`, of size " << where.size()
        << ". Expected its size to be the same as this Shape's rank, "
        << rank_u64() << '.';
    throw error(oss.str());
  }

  for (uint64_t dimension = 0; dimension < where.size(); ++dimension) {
    validateGatherIndices(dimension, where[dimension]);
  }

  std::vector<int64_t> indices{0};
  std::vector<int64_t> prevIndices;

  const auto rmStrides = getRowMajorStrides();

  for (uint64_t i = 0; i < rank_u64(); ++i) {
    auto d = rank_u64() - i - 1;

    std::swap(indices, prevIndices);
    indices.clear();
    indices.reserve(prevIndices.size() * where[d].size());

    for (auto w : where[d]) {
      const auto delta = w * rmStrides[d];
      for (auto p : prevIndices) {
        indices.push_back(p + delta);
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
          << "with `a`=" << a << " and `b`=" << b << ". "
          << "Failed at index " << i << '.'
          << " `a` and `b` cannot be numpy added. ";
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
std::string Shape::str() const {
  std::ostringstream ost;
  append(ost);
  return ost.str();
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

Shape Shape::pad(const Lower &l, const Upper &u) const {

  const auto throwBadParams = [this, &l, &u]() {
    std::ostringstream oss;
    oss << "Invalid Lower or Upper in Shape::pad. "
        << "This Shape is " << *this << ", Lower is " << l << " of rank "
        << l.size() << ", and Upper is " << u << ", of size " << u.size()
        << '.';
    throw error(oss.str());
  };

  if (l.size() != rank_u64() || u.size() != rank_u64()) {
    throwBadParams();
  }

  std::vector<int64_t> out;
  out.reserve(rank_u64());
  for (uint64_t i = 0; i < rank_u64(); ++i) {
    if (l[i] < 0 || u[i] < 0) {
      throwBadParams();
    }
    out.push_back(dim(i) + l[i] + u[i]);
  }
  return out;
}

Shape Shape::addToDims(const std::vector<int64_t> &d) const {

  if (d.size() != rank_u64()) {
    std::ostringstream oss;
    oss << "Invalid values in Shape::addToDims. "
        << "This Shape is " << *this << ", values"
        << ": type should have the same rank but do not. ";
    throw error(oss.str());
  }

  std::vector<int64_t> out;
  out.reserve(rank_u64());
  for (uint64_t i = 0; i < rank_u64(); ++i) {
    out.push_back(dim(i) + d[i]);
    if (out.back() < 0) {
      std::ostringstream oss;
      oss << "Invalid values in Shape::addToDims. "
          << "In dimension " << i << ", Shape's dim is " << dim(i)
          << ", and the delta value is " << d[i]
          << ". Adding these together results in a negative dimension"
          << ", which is not allowed. ";
      throw error(oss.str());
    }
  }
  return out;
}

Shape Shape::slice(const Lower &l, const Upper &u) const {
  assertSliceBoundsAreValid(l, u);
  std::vector<int64_t> out(rank_u64(), 0);
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    out[d] = u[d] - l[d];
  }
  return Shape(out);
}

std::pair<Shape::Lower, Shape::Upper>
Shape::getFullSliceBounds(Dimension d, uint64_t l, uint64_t u) const {
  if (d.get() >= rank_u64()) {
    std::ostringstream oss;
    oss << "Cannot slice this Shape, " << *this << ", in Dimension "
        << d.get() << ", as it is only of rank " << rank_u64() << '.';
    throw error(oss.str());
  }

  Lower lows(rank_u64(), 0);
  lows[d.get()] = static_cast<int64_t>(l);

  Upper upps    = get();
  upps[d.get()] = static_cast<int64_t>(u);

  return {lows, upps};
}

Shape Shape::slice(Dimension d, uint64_t l, uint64_t u) const & {
  const auto lowsUpps = getFullSliceBounds(d, l, u);
  return slice(std::get<0>(lowsUpps), std::get<1>(lowsUpps));
}

Shape &&Shape::slice(Dimension d, uint64_t l, uint64_t u) && {
  assertValidDimension(d.get());
  assertSliceBoundsAreValid(d, l, u);
  shp[d.get()] = u - l;
  return std::move(*this);
}

namespace {
// Number of elements in [start, end) with stride of step. step may be
// negative, and start may be greater than end. The return value cannot be
// negative though.
int64_t getSize(int64_t start, int64_t end, int64_t step) {

  // If the step takes you further away from end, then return 0.
  if ((start < end) != (step > 0)) {
    return 0ll;
  }
  uint64_t delta_u64  = start < end ? end - start : start - end;
  uint64_t step_u64   = step > 0 ? step : -step;
  const auto size_u64 = delta_u64 / step_u64 + (delta_u64 % step_u64 != 0);
  return static_cast<int64_t>(size_u64);
}
} // namespace

Shape Shape::slice(const Starts &starts,
                   const Ends &ends,
                   const Steps &steps,
                   const Dims &dims) const {
  const auto n = getNormalizedSliceParams(starts, ends, steps, dims);
  return slice(n);
}

void Shape::validateNormalizedSliceParams(
    const NormalizedSliceParams &n) const {
  const auto getBase = [&n, this]() {
    std::ostringstream oss;
    oss << "Invalid call " << *this
        << ".validateNormalizedSliceParams(n=" << n << "). ";
    return oss.str();
  };
  if (n.size() != rank_u64()) {
    std::ostringstream oss;
    oss << getBase() << "Expected n to be of size " << rank_u64()
        << ", the size of this Shape. ";
    throw error(oss.str());
  }
  for (uint64_t i = 0; i < rank_u64(); ++i) {
    if (n.step(i) == 0) {
      std::ostringstream oss;
      oss << getBase() << "Steps must all be non-zero.";
      throw error(oss.str());
    }
    if (n.start(i) < 0 || n.start(i) >= dim(i)) {
      std::ostringstream oss;
      oss << getBase() << "Starts must be in [0, dim).";
      throw error(oss.str());
    }
    if (n.end(i) < -1 || n.end(i) > dim(i)) {
      std::ostringstream oss;
      oss << getBase() << "Ends must be in [-1, dim+1).";
      throw error(oss.str());
    }
  }
}

Shape Shape::slice(const NormalizedSliceParams &n) const {
  validateNormalizedSliceParams(n);
  std::vector<int64_t> shape_(rank_u64());
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    shape_[d] = getSize(n.start(d), n.end(d), n.step(d));
  }
  return shape_;
}

std::vector<int64_t> Shape::getSlicedRowMajorIndices(const Starts &starts,
                                                     const Ends &ends,
                                                     const Steps &steps,
                                                     const Dims &dims) const {
  return getSlicedRowMajorIndices(
      getNormalizedSliceParams(starts, ends, steps, dims));
}

void Shape::assertSliceBoundsAreValid(const Lower &l, const Upper &u) const {

  // same rank for lower and upper
  if (l.size() != u.size() || u.size() != rank_u64()) {
    std::ostringstream ss;
    ss << "lower and upper must both be of size "
       << " " << rank_u64() << ". This ia not true for lower=" << l
       << " and upper=" << u << '.';
    throw error(ss.str());
  }

  // lower less than or equal to upper
  for (auto i = 0ul; i < rank_u64(); ++i) {
    assertSliceBoundsAreValid(Dimension(i), l[i], u[i]);
  }
}

void Shape::assertSliceBoundsAreValid(Dimension d,
                                      uint64_t l,
                                      uint64_t u) const {

  std::ostringstream ss;

  if (l > u) {
    ss << "lower bound cannot exceed upper bound. "
       << "This for lower=" << l << " and upper=" << u << '.';
    throw error(ss.str());
  }

  if (dim_u64(d.get()) < u) {
    ss << "Failure in Shape::assertSliceBoundsAreValid. "
       << "Upper bound cannot exceed dimension size "
       << "(in dimension " << d.get() << ") "
       << "This for Shape = " << *this << ", lower=" << l
       << " and upper=" << u << '.';
    throw error(ss.str());
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

namespace {
// some numpy experiments:
//
// In [5]: X = np.arange(4)
// In [6]: X
// Out[6]: array([0, 1, 2, 3])
//
// In [7]: X[-100:100:1] #start:end:step
// Out[7]: array([0, 1, 2, 3])
//
// In [8]: X[90:100:1]
// Out[8]: array([], dtype=int64)
//
// In [9]: X[-100:1:1]
// Out[9]: array([0])
//
// In [10]: X[-1:100:1]
// Out[10]: array([3])
//
// In [11]: X[-1:100:1]
//
// Map \a v to [low, high), pivoting on pivot
int64_t normalize(int64_t v, uint64_t pivot, int64_t low, int64_t high) {
  if (v < 0) {
    v += pivot;
  }
  if (v < low) {
    v = low;
  }
  if (v >= high) {
    v = high - 1;
  }
  return v;
}
} // namespace

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

Shape::NormalizedSliceParams::NormalizedSliceParams(const Starts &starts_,
                                                    const Ends &ends_,
                                                    const Steps &steps_,
                                                    const Dims &dims_,
                                                    const Shape &shape) {

  // return a string, summarizing the input parameters. Used for error
  // messages.
  auto getBase = [&shape, &starts_, &ends_, &steps_, &dims_]() {
    std::ostringstream oss;
    oss << "In NormalizedSliceParams constructor, with "
        << "starts=" << starts_.get() << ", ends=" << ends_.get()
        << ", steps=" << steps_.get() << ", dims=" << dims_.get()
        << ", shape=" << shape.get() << ". ";
    return oss.str();
  };

  const auto R_u64 = shape.rank_u64();
  const auto R_i64 = shape.rank_i64();

  if (starts_.size() != ends_.size()) {
    std::ostringstream oss;
    oss << getBase() << "starts and ends must be same size. ";
    throw error(oss.str());
  }

  const auto NIn = starts_.size();

  // Dims
  // If no dimensions are provided, the ONNX spec stipulates that is
  // implicitly all dimensions of the Shape. That is, starts and ends are for
  // all dimensions.
  if (dims_.empty()) {
    if (NIn != R_u64) {
      std::ostringstream oss;
      oss << getBase()
          << "As dims is empty, starts and ends must both be of size "
          << R_u64 << ", the rank of the Shape. ";
      throw error(oss.str());
    }
  }

  // If dimensions are provided, they correspond to values in starts and ends.
  else {
    if (NIn != dims_.size()) {
      std::ostringstream oss;
      oss << getBase()
          << "As dims is non-empty, starts and ends must both be of size "
          << dims_.size() << ", the size of dims. ";
      throw error(oss.str());
    }
  }

  // Empty steps implies step size 1 in every dimension
  if (!steps_.empty() && steps_.size() != NIn) {
    std::ostringstream oss;
    oss << getBase() << "As steps is non-empty, it must be of size " << NIn
        << ", the size of starts and ends. ";
  }
  if (std::any_of(steps_.vals.cbegin(), steps_.vals.cend(), [](auto x) {
        return x == 0;
      })) {
    std::ostringstream oss;
    oss << getBase() << "Invalid steps, all steps must be non-zero";
    throw error(oss.str());
  }

  // At this point in the code, all 4 vectors are of size NIn.
  // Normalize the elements of dims, to be positive.
  auto dims = dims_.vals;
  std::vector<bool> dimsSeen(R_u64, false);
  for (auto &d : dims) {
    if (d < -R_i64 || d >= R_i64) {
      std::ostringstream oss;
      oss << getBase() << "Dimension " << d << " is not in the range ["
          << -R_i64 << ", " << R_i64 << "). ";
      throw error(oss.str());
    }
    if (d < 0) {
      d += R_i64;
    }

    // At this point in the code, we know that d is non-negative.
    const auto d_u64 = static_cast<uint64_t>(d);
    if (dimsSeen[d_u64]) {
      std::ostringstream oss;
      oss << getBase() << "Repeated dimension, " << d_u64 << ".";
      throw error(oss.str());
    }
    dimsSeen[static_cast<uint64_t>(d)] = true;
  }
  if (dims.empty()) {
    dims.resize(R_u64);
    std::iota(dims.begin(), dims.end(), 0);
  }

  // At this point, all 4 vectors are of the same length, and all elements of
  // dims are positive. We initialize the normalized parameters to defaults:
  starts = std::vector<int64_t>(R_u64, 0);
  ends   = shape.get();
  steps  = std::vector<int64_t>(R_u64, 1);

  for (uint64_t i = 0; i < NIn; ++i) {
    const auto d = static_cast<uint64_t>(dims[i]);

    const auto start = starts_.vals[i];
    starts[d]        = normalize(start, shape.dim_u64(d), 0, shape.dim(d));

    const auto end = ends_.vals[i];
    ends[d]        = normalize(end, shape.dim_u64(d), -1, shape.dim(d) + 1);

    if (!steps_.empty()) {
      const auto step = steps_.vals[i];
      steps[d]        = step;
    }
  }
}

Shape::NormalizedSliceParams
Shape::getNormalizedSliceParams(const Starts &starts,
                                const Ends &ends,
                                const Steps &steps,
                                const Dims &dims) const {
  return NormalizedSliceParams(starts, ends, steps, dims, *this);
}

void Shape::NormalizedSliceParams::append(std::ostream &ost) const {
  ost << "starts=" << starts << ",ends=" << ends << ",steps=" << steps;
}

std::ostream &operator<<(std::ostream &ost,
                         const Shape::NormalizedSliceParams &n) {
  n.append(ost);
  return ost;
}

void Shape::assertConcattable(const Shapes &shapes_, uint64_t axis_) {
  if (shapes_.size() == 0) {
    throw error("Failed in assertConcattable, where there are 0 Shapes. "
                "Concatenation requires at least one Shape. ");
  }

  for (uint64_t i = 1; i < shapes_.size(); ++i) {
    shapes_[i].assertConcattable(shapes_[0], axis_);
  }
}

std::vector<uint64_t>
Shape::getCanonicalReverseIndices(const std::vector<uint64_t> &where) const {

  std::vector<bool> flips(rank_u64(), false);
  for (auto d : where) {
    if (d < rank_u64()) {
      flips[d] = !flips[d];
    } else {
      std::ostringstream oss;
      oss << "Invalid index " << d
          << " in getCanonicalReverseIndices for Shape " << *this
          << ", which is of rank " << rank_u64() << '.';
      throw error(oss.str());
    }
  }

  std::vector<uint64_t> flipped;
  for (uint64_t d = 0; d < rank_u64(); ++d) {
    if (flips[d]) {
      flipped.push_back(d);
    }
  }
  return flipped;
}

bool Shape::canReduceTo(const Shape &outShape) const {
  const auto outRank = outShape.rank_u64();

  // cannot reduce to higher rank
  if (outRank > rank_u64()) {
    return false;
  }

  const auto deltaRank = rank_u64() - outRank;

  for (uint64_t d = 0; d < outShape.rank_u64(); ++d) {
    if (outShape.dim(d) != 1) {
      if (outShape.dim(d) != dim(d + deltaRank)) {
        // cannot reduce, as neither is 1.
        return false;
      }
    }
  }
  return true;
}

void Shape::assertCanReduceTo(const Shape &outShape) const {
  if (!canReduceTo(outShape)) {
    std::ostringstream oss;
    oss << "Cannot reduce from " << *this << " to " << outShape << ".";
    throw error(oss.str());
  }
}

Shape Shape::scale(Stride s, Dimension d) const & {
  assertValidDimension(d.get());
  auto x = get();
  x[d.get()] *= s.get();
  return x;
}

Shape &&Shape::scale(Stride s, Dimension d) && {
  assertValidDimension(d.get());
  shp[d.get()] *= s.get();
  return std::move(*this);
}

bool Shape::isSqueezed() const {
  return std::all_of(shp.cbegin(), shp.cend(), [](auto x) { return x != 1; });
}

std::vector<uint64_t> Shape::singletonDimensions() const {
  std::vector<uint64_t> singletons;
  for (uint64_t i = 0; i < rank_u64(); ++i) {
    if (dim(i) == 1) {
      singletons.push_back(i);
    }
  }
  return singletons;
}

std::vector<uint64_t> Shape::nonSingletonDimensions() const {
  std::vector<uint64_t> nonSingletons;
  for (uint64_t i = 0; i < rank_u64(); ++i) {
    if (dim(i) != 1) {
      nonSingletons.push_back(i);
    }
  }
  return nonSingletons;
}

void Shape::assertValidFlatten(uint64_t from, uint64_t to) const {

  if (from >= to || to > rank_u64()) {
    std::ostringstream oss;
    oss << "Invalid call for this Shape: " << *this
        << ". Call to flatten(from = " << from << ", to = " << to
        << ") is invalid as it does not satisfy the requirement that "
        << "0 <= from < to <= rank. ";
    throw error(oss.str());
  }
}

Shape Shape::flatten(uint64_t from, uint64_t to) const & {

  assertValidFlatten(from, to);

  std::vector<int64_t> dims;
  dims.reserve(rank_u64() - (to - from - 1));
  dims.insert(dims.end(), shp.cbegin(), shp.cbegin() + from);
  dims.push_back(dimProduct(from, to));
  dims.insert(dims.end(), shp.begin() + to, shp.cend());
  return dims;
}

Shape &&Shape::flatten(uint64_t from, uint64_t to) && {
  assertValidFlatten(from, to);
  shp[from]         = dimProduct(from, to);
  const auto offset = to - from - 1;
  for (uint64_t i = to; i < rank_u64(); ++i) {
    shp[i - offset] = shp[i];
  }
  shp.resize(rank_u64() - offset);
  return std::move(*this);
}

} // namespace ndarray
} // namespace poprithms
