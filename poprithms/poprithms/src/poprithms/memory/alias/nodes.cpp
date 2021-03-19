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
std::string DimShuffle::typeString() const {
  std::ostringstream oss;
  oss << "DimShuffle " << permutation();
  return oss.str();
}

std::string Reverse::typeString() const {
  std::ostringstream oss;
  oss << "Reverse ";
  util::append(oss, dimensions());
  return oss.str();
}

std::string Allocate::typeString() const {
  return "Allocate(" + std::to_string(color_.get()) + ")";
}

std::string SettSample::typeString() const {
  std::ostringstream oss;
  oss << "SettSample(" << region().setts() << ')';
  return oss.str();
}

std::string SettFill::typeString() const {
  std::ostringstream oss;
  oss << "SettFill(" << regions() << ')';
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

std::unique_ptr<Node> SettFill::clone(const State &state,
                                      const Origins &oris) const {
  return std::make_unique<SettFill>(state, oris, regions());
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

std::unique_ptr<Node> DimShuffle::clone(const State &state,
                                        const Origins &oris) const {
  return std::make_unique<DimShuffle>(state, oris, permutation());
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

DisjointRegions SettFill::getInRegions(InIndex inIndex,
                                       const DisjointRegions &outRegs) const {
  return outRegs.settSample(region(inIndex));
}

SettFill::SettFill(const State &ob,
                   const Origins &oris,
                   const DisjointRegions &regions__)
    : Node(ob, oris), regions_(regions__) {

  // confirm number of input Tensors is the same as the number of Regions:
  if (ob.ins.size() != regions_.size()) {
    std::ostringstream oss;
    oss << "ids and regions of different sizes in SettFill constructor. ";
    oss << "ids = ";
    util::append(oss, ob.ins);
    oss << ", and regions = " << regions_;
    throw error(oss.str());
  }

  // confirm no intersections between Regions:
  regions_.confirmValid();

  // confirm a complete partition:
  if (regions_.totalElms() != regions_.shape().nelms()) {
    std::ostringstream oss;
    oss << "The SettFills region must pack together to cover the full Shape."
        << "This for regions = " << regions() << ", which has "
        << regions_.totalElms() << ". The containing Shape has "
        << regions_.shape().nelms() << ". The respective regions shapes [ ";
    for (const auto &r : regions_.get()) {
      oss << r.totalElms() << ' ';
    }
    oss << "] . ";
    throw error(oss.str());
  }

  // confirm Shapes match exactly:
  for (uint64_t i = 0; i < ob.ins.size(); ++i) {
    if (ob.inShapes[i].get() != region(i).nelms()) {
      std::ostringstream oss;
      oss << "The " << i << "'th input Tensor has Shape " << ob.inShapes[i]
          << ", which cannot map to Region " << region(i)
          << " as it has number of elements (in each dimension) of ";
      util::append(oss, region(i).nelms());
      throw error(oss.str());
    }
  }
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

DisjointRegions
DimShuffle::getInRegions(InIndex, const DisjointRegions &outRegs) const {
  return outRegs.dimShuffle(permutation().inverse());
}

} // namespace alias
} // namespace memory
} // namespace poprithms
