// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <memory/alias/error.hpp>

#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/alias/node.hpp>
#include <poprithms/memory/alias/tensor.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace poprithms {
namespace memory {
namespace alias {

namespace {

TensorIds getIds(const Tensors &tensors) {
  TensorIds tensorIds;
  tensorIds.reserve(tensors.size());
  for (const auto &tensor : tensors) {
    tensorIds.push_back(tensor.id());
  }
  return tensorIds;
}

std::vector<TensorId>
getAllIds(const Tensors &tensors_, const Tensor &toInsert, uint64_t index) {
  if (index > tensors_.size()) {
    std::ostringstream oss;
    oss << "Failure in \n    getAllIds(tensors_ of size " << tensors_.size()
        << ", index = " << index
        << "): argument 'index' cannot exceed the size of tensors_. ";
    throw error(oss.str());
  }
  std::vector<TensorId> allIds;
  const auto insertionIter = std::next(tensors_.cbegin(), index);
  allIds.reserve(tensors_.size() + 1);
  for (auto iter = tensors_.cbegin(); iter != insertionIter; ++iter) {
    allIds.push_back(iter->id());
  }
  allIds.push_back(toInsert.id());
  for (auto iter = insertionIter; iter != tensors_.cend(); ++iter) {
    allIds.push_back(iter->id());
  }
  return allIds;
}
} // namespace

std::ostream &operator<<(std::ostream &oss, const Tensor &x) {
  oss << "tensor:" << x.id();
  return oss;
}

Tensor Tensor::slice(uint64_t start, uint64_t end, Dimension sliceDim) const {
  if (sliceDim.get() >= rank_u64()) {
    std::ostringstream oss;
    oss << "Cannot slice this Tensor of rank " << rank_u64()
        << " in dimension " << sliceDim.get() << '.';
    throw error(oss.str());
  }

  // The lower and upper bounds to slice between are:
  // lower = (0, ..., 0,              start, ..., 0)
  // upper = (d0, ... d_{sliceDim-1}, end,   .... d_{rank-1})

  Lower l(rank_u64(), 0);
  l[sliceDim.get()] = start;
  Upper u           = shape().get();
  u[sliceDim.get()] = end;
  return slice(l, u);
}

Tensors Tensor::slices(const Intervals &intervals, uint64_t dim) const {
  Tensors tensors;
  tensors.reserve(intervals.size());
  for (const auto &interval : intervals) {
    tensors.push_back(slice(interval.l(), interval.u(), Dimension(dim)));
  }
  return tensors;
}

Tensors Tensor::slices(const std::vector<Intervals> &intervals,
                       uint64_t dim) const {
  Tensors tensors;
  tensors.reserve(intervals.size());

  for (const auto &intervalSeq : intervals) {
    auto seqTensors = slices(intervalSeq, dim);
    tensors.push_back({pgraph->concat(getIds(seqTensors), dim), pgraph});
  }
  return tensors;
}

Tensor concat(Tensors &&tensors, uint64_t axis) {
  if (tensors.size() == 0) {
    throw error("Cannot concatenate an empty vector of Tensors");
  }
  const auto b = tensors.back();
  tensors.pop_back();
  return b.concat(tensors, tensors.size(), axis);
}

Tensor concat(const Tensors &tensors, uint64_t axis) {
  auto tensors_ = tensors;
  return concat(std::move(tensors_), axis);
}

Tensor settfill(Tensors &&tensors, const DisjointRegions &regions) {
  if (tensors.size() == 0) {
    throw error("Cannot settfill an empty vector of Tensors");
  }
  const auto b = tensors.back();
  tensors.pop_back();
  return b.settfill(tensors, tensors.size(), regions);
}

Tensor settfill(const Tensors &tensors, const DisjointRegions &regions) {
  auto tensors_ = tensors;
  return settfill(std::move(tensors_), regions);
}

void Tensor::toAllocation(Color c) { pgraph->toAllocation(id(), c); }

void Tensor::toIdentityFrom(Tensor src) {
  if (src.pgraph != pgraph) {
    throw error(
        "Cannot call Tensor::toIdentityFrom for Tensors in different Graphs");
  }
  pgraph->toIdentity(src.id(), id());
}

Tensor
Tensor::concat(const Tensors &tensors_, uint64_t index, uint64_t axis) const {
  if (tensors_.empty()) {
    return *this;
  }
  return {pgraph->concat(getAllIds(tensors_, *this, index), axis), pgraph};
}

Tensor Tensor::settfill(const Tensors &tensors_,
                        uint64_t index,
                        const DisjointRegions &regions) const {
  if (tensors_.empty()) {
    return *this;
  }
  return {pgraph->settfill(getAllIds(tensors_, *this, index), regions),
          pgraph};
}

Tensors Tensor::getNonDisjoint() const {
  const auto dj = pgraph->allAliases(id());
  Tensors tens;
  tens.reserve(dj.size());
  for (auto id : dj) {
    tens.push_back({id, pgraph});
  }
  return tens;
}

const Shape &Tensor::shape() const { return pgraph->shape(id()); }

Tensor Tensor::concatFirstDim(const Tensors &tensors_, uint64_t index) const {
  return concat(tensors_, index, 0);
}

Tensor Tensor::flatten() const { return reshape(shape().flatten()); }

Tensor Tensor::reshape(const Shape &to) const {
  return {pgraph->reshape(id(), to), pgraph};
}

Tensor Tensor::upsample(uint64_t scale, uint64_t dim) const {
  auto broadcasted =
      reshape(shape().unsqueeze(dim + 1)).broadcast(scale, dim + 1);
  return broadcasted.reshape(broadcasted.shape().flatten(dim, dim + 2));
}

Tensor Tensor::subscript(uint64_t index) const {
  auto slicedTensor = slice(index, index + 1, Dimension(0));
  return slicedTensor.reshape(slicedTensor.shape().squeeze({0}));
}

Tensor Tensor::index(const std::vector<uint64_t> &indices) const {
  uint64_t rank = shape().rank_u64();
  if (indices.size() > rank) {
    std::ostringstream oss;
    oss << "Number of indices (= " << indices.size()
        << ") exceeds rank (= " << rank << ").";
    throw error(oss.str());
  }

  Lower begin(rank, 0);
  Upper end = shape().get();
  for (uint64_t dim = 0; dim < indices.size(); ++dim) {
    begin[dim] = indices[dim];
    end[dim]   = indices[dim] + 1;
  }
  return slice(begin, end).reshape(shape().fromDim(indices.size()));
}

Tensor Tensor::expand(const Shape &to) const {
  return {pgraph->expand(id(), to), pgraph};
}

Tensor Tensor::reverse(uint64_t dimension) const {
  return reverse(std::vector<uint64_t>(1, dimension));
}

Tensor Tensor::reverse(const std::vector<uint64_t> &dimensions) const {
  return {pgraph->reverse(id(), dimensions), pgraph};
}

Tensor Tensor::broadcast(int64_t N, uint64_t dimension) const {
  // If shape = (a,b,c), dimension = 1:
  //               \.
  //             (a,1,b,c) -> (a,N,b,c) -> (a,N*b,c)
  const auto unsqueezedShape = shape().unsqueeze(dimension);
  const auto unsqueezedArr   = reshape(unsqueezedShape);

  const auto expandedShape = unsqueezedShape.broadcast(N, dimension);
  const auto expandedArr   = unsqueezedArr.expand(expandedShape);

  const auto broadcastShape = shape().broadcast(N, dimension);
  return expandedArr.reshape(broadcastShape);
}

Tensor Tensor::subsample(int64_t stride, uint64_t dimension) const {
  if (stride < 1 || dimension >= rank_u64()) {
    std::ostringstream oss;
    oss << "Invalid call to Tensor::subsample(stride=" << stride
        << ", dimension=" << dimension
        << "). Stride must be strictly positive, "
        << " dimension must be less than rank (" << rank_u64() << "). ";
    throw error(oss.str());
  }

  return {
      pgraph->settSample(
          id(), Region::fromStripe(shape(), dimension, {1, stride - 1, 0})),
      pgraph};
}

bool Tensor::isRowMajorSetContiguous() const {
  return pgraph->isRowMajorSetContiguous(id());
}

bool Tensor::containsAliases() const { return pgraph->containsAliases(id()); }

bool Tensor::containsColor(Color c) const {
  return pgraph->containsColor(id(), c);
}

bool Tensor::intersectsWith(const Tensor &rhs) const {
  return pgraph->areAliased(id(), rhs.id());
}

Tensor Tensor::squeeze() const { return reshape(shape().squeeze()); }

Tensor Tensor::slice(const Lower &l, const Upper &u) const {
  return {pgraph->settSample(id(), Region::fromBounds(shape(), l, u)),
          pgraph};
}

Tensor Tensor::settSample(const Region &r) const {
  return {pgraph->settSample(id(), r), pgraph};
}

Tensor Tensor::dimShuffle(const Permutation &perm) const {
  return {pgraph->dimShuffle(id(), perm), pgraph};
}

Tensor Tensor::clone() const { return {pgraph->clone(id()), pgraph}; }

Tensor Tensor::concatFinalDim(const Tensors &tensors_, uint64_t index) const {
  if (tensors_.empty()) {
    return *this;
  }
  const auto rank0 = tensors_[0].shape().rank_u64();
  return concat(tensors_, index, std::max<uint64_t>(rank0, 1) - 1);
}

} // namespace alias
} // namespace memory
} // namespace poprithms
