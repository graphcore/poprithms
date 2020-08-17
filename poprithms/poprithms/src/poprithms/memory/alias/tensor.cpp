// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/graph.hpp>
#include <poprithms/memory/alias/node.hpp>
#include <poprithms/memory/alias/tensor.hpp>
#include <poprithms/memory/nest/region.hpp>

namespace poprithms {
namespace memory {
namespace alias {

std::ostream &operator<<(std::ostream &oss, const Tensor &x) {
  oss << "tensor:" << x.id();
  return oss;
}

Tensor concat(std::vector<Tensor> &&tensors, uint64_t axis) {
  if (tensors.size() == 0) {
    throw error("Cannot concatenate an empty vector of Tensors");
  }
  const auto b = tensors.back();
  tensors.pop_back();
  return b.concat(tensors, tensors.size(), axis);
}

Tensor concat(const std::vector<Tensor> &tensors, uint64_t axis) {
  auto tensors_ = tensors;
  return concat(std::move(tensors_), axis);
}

void Tensor::toAllocation(Color c) { pgraph->toAllocation(id(), c); }

void Tensor::toIdentityFrom(Tensor src) {
  if (src.pgraph != pgraph) {
    throw error(
        "Cannot call Tensor::toIdentityFrom for Tensors in different Graphs");
  }
  pgraph->toIdentity(src.id(), id());
}

Tensor Tensor::concat(const std::vector<Tensor> &tensors_,
                      uint64_t index,
                      uint64_t axis) const {

  if (tensors_.empty()) {
    return *this;
  }
  if (index > tensors_.size()) {
    std::ostringstream oss;
    oss << "Failure in \n    Tensor::concat(tensors_ of size "
        << tensors_.size() << ", index = " << index << " axis = " << axis
        << "): argument 'index' cannot excede the size of tensors_. ";
    throw error(oss.str());
  }
  std::vector<TensorId> allIds;
  const auto insertionIter = std::next(tensors_.cbegin(), index);
  allIds.reserve(tensors_.size() + 1);
  for (auto iter = tensors_.cbegin(); iter != insertionIter; ++iter) {
    allIds.push_back(iter->id());
  }
  allIds.push_back(id());
  for (auto iter = insertionIter; iter != tensors_.cend(); ++iter) {
    allIds.push_back(iter->id());
  }
  return {pgraph->concat(allIds, axis), pgraph};
}

std::vector<Tensor> Tensor::getNonDisjoint() const {
  const auto dj = pgraph->allAliases(id());
  std::vector<Tensor> tens;
  tens.reserve(dj.size());
  for (auto id : dj) {
    tens.push_back({id, pgraph});
  }
  return tens;
}

const Shape &Tensor::shape() const { return pgraph->shape(id()); }

Tensor Tensor::hstack(const std::vector<Tensor> &tensors_,
                      uint64_t index) const {
  return concat(tensors_, index, 0);
}

Tensor Tensor::flatten() const { return reshape(shape().flatten()); }

Tensor Tensor::reshape(const Shape &to) const {
  return {pgraph->reshape(id(), to), pgraph};
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
      pgraph->settsample(
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
  return {pgraph->settsample(id(), Region::fromBounds(shape(), l, u)),
          pgraph};
}

Tensor Tensor::settsample(const Region &r) const {
  return {pgraph->settsample(id(), r), pgraph};
}

Tensor Tensor::dimshuffle(const Permutation &perm) const {
  return {pgraph->dimshuffle(id(), perm), pgraph};
}

Tensor Tensor::clone() const { return {pgraph->clone(id()), pgraph}; }

Tensor Tensor::vstack(const std::vector<Tensor> &tensors_,
                      uint64_t index) const {
  if (tensors_.empty()) {
    return *this;
  }
  const auto rank0 = tensors_[0].shape().rank_u64();
  return concat(tensors_, index, std::max<uint64_t>(rank0, 1) - 1);
}

} // namespace alias
} // namespace memory
} // namespace poprithms
