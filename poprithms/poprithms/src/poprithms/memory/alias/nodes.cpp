// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>

#include <poprithms/memory/alias/error.hpp>
#include <poprithms/memory/alias/nodes.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace alias {

std::ostream &operator<<(std::ostream &ost, const std::vector<int64_t> &x) {
  poprithms::util::append(ost, x);
  return ost;
}
std::string Permute::typeString() const {
  std::ostringstream oss;
  oss << "Permute " << permutation();
  return oss.str();
}

std::string Reverse::typeString() const {
  std::ostringstream oss;
  oss << "Reverse ";
  util::append(oss, dimensions());
  return oss.str();
}

std::string SettSample::typeString() const {
  std::ostringstream oss;
  oss << "SettSample " << region().setts();
  ;
  return oss.str();
}

SettSample::SettSample(const Node::State &state,
                       const Origins &oris,
                       const Shape &inShape,
                       const Lower &l,
                       const Upper &u)
    : Node(state, oris), region_(Region::fromBounds(inShape, l, u)) {}

std::unique_ptr<Node> Concat::clone(const State &state,
                                    const Origins &oris) const {
  return std::make_unique<Concat>(state, oris, axis());
}

std::unique_ptr<Node> Identity::clone(const State &state,
                                      const Origins &oris) const {
  return std::make_unique<Identity>(state, oris);
}

std::unique_ptr<Node> Allocate::clone(const State &state,
                                      const Origins &oris) const {
  return std::make_unique<Allocate>(state, oris, color());
}

std::unique_ptr<Node> SettSample::clone(const State &state,
                                        const Origins &oris) const {
  return std::make_unique<SettSample>(state, oris, region());
}

std::unique_ptr<Node> Reshape::clone(const State &state,
                                     const Origins &oris) const {
  return std::make_unique<Reshape>(state, oris);
}

std::unique_ptr<Node> Permute::clone(const State &state,
                                     const Origins &oris) const {
  return std::make_unique<Permute>(state, oris, permutation());
}

std::unique_ptr<Node> Expand::clone(const State &state,
                                    const Origins &oris) const {
  return std::make_unique<Expand>(state, oris);
}

std::unique_ptr<Node> Reverse::clone(const State &state,
                                     const Origins &oris) const {
  return std::make_unique<Reverse>(state, oris, dimensions());
}

DisjointRegions Concat::getInRegions(InIndex inIndex,
                                     const DisjointRegions &outRegs) const {
  return outRegs.slice(getLowerSlice(inIndex), getUpperSlice(inIndex));
}

std::vector<int64_t> Concat::getLowerSlice(InIndex i) const {
  std::vector<int64_t> x(shape().rank_u64(), 0LL);
  x[axis()] = partitionPoints_[i.get()];
  return x;
}
std::vector<int64_t> Concat::getUpperSlice(InIndex i) const {
  auto x    = shape().get();
  x[axis()] = partitionPoints_[i.get() + 1];
  return x;
}

DisjointRegions
SettSample::getInRegions(InIndex, const DisjointRegions &outRegs) const {
  return outRegs.settFillInto(region());
}

DisjointRegions Reverse::getInRegions(InIndex,
                                      const DisjointRegions &outRegs) const {
  return outRegs.reverse(dimensions());
}

DisjointRegions Allocate::getInRegions(InIndex,
                                       const DisjointRegions &) const {
  throw error("Invalid call to Allocate::getInRegions: not supported");
}

DisjointRegions Reshape::getInRegions(InIndex,
                                      const DisjointRegions &outRegs) const {
  return outRegs.reshape(inShape(0));
}

DisjointRegions Expand::getInRegions(InIndex,
                                     const DisjointRegions &outRegs) const {
  return outRegs.reduce(inShape(0));
}

DisjointRegions Permute::getInRegions(InIndex,
                                      const DisjointRegions &outRegs) const {
  return outRegs.permute(permutation().inverse());
}

} // namespace alias
} // namespace memory
} // namespace poprithms
