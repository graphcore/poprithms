// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <numeric>
#include <sstream>
#include <variant>

#include <memory/chain/disjointregionsmapper.hpp>
#include <memory/chain/error.hpp>
#include <memory/chain/hosttensormapper.hpp>
#include <memory/chain/op.hpp>

#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/chain/type.hpp>
#include <poprithms/util/copybyclone_impl.hpp>

namespace poprithms {
namespace memory {
namespace chain {

DisjointRegions Chain::apply(const DisjointRegions &rIn) const {
  return apply<DisjointRegionsMapper, DisjointRegions>(rIn);
}

DisjointRegions Chain::apply(const DisjointRegions &rIn,
                             uint64_t nOpsToApply) const {
  return apply<DisjointRegionsMapper, DisjointRegions>(rIn, nOpsToApply);
}

compute::host::Tensor Chain::apply(const compute::host::Tensor &t) const {

  if (t.shape() != inShape()) {
    std::ostringstream oss;
    oss << "Cannot apply this Chain, whose entry shape is " << inShape()
        << ", to a host::Tensor of Shape " << t.shape();
    throw error(oss.str());
  }
  return apply<HostTensorMapper, compute::host::Tensor>(t);
}

class Chain::Ops {
public:
  bool operator==(const Ops &rhs) const { return ops == rhs.ops; }
  bool operator!=(const Ops &rhs) const { return !operator==(rhs); }
  std::unique_ptr<Ops> clone() const { return std::make_unique<Ops>(*this); }
  std::vector<Op> ops;
};

template <typename X> void Chain::append(Type t, const Shape &o, const X &a) {
  ops_.uptr->ops.push_back({t, o, a});
}

void Chain::append(const Op &op) {
  ops_.uptr->ops.push_back(op);
  verifyOp(nOps() - 1);
}

Chain Chain::mirror() const {
  Chain rev(outShape());

  for (uint64_t i = 0; i < nOps(); ++i) {
    const auto toRevIndex  = nOps() - i - 1;
    const auto &toRev      = op(toRevIndex);
    const auto fwdOutShape = toRev.outShape();
    const auto fwdInShape =
        toRevIndex == 0 ? inShape() : outShape(toRevIndex - 1);
    switch (toRev.type()) {
    case Type::Reshape: {
      rev.reshape(fwdInShape);
      break;
    }
    case Type::Expand: {
      rev.reduce(fwdInShape);
      break;
    }
    case Type::Reduce: {
      rev.expand(fwdInShape);

      break;
    }
    case Type::SettSample: {
      rev.settFillInto(toRev.attr().region());
      break;
    }
    case Type::SettFillInto: {
      rev.settSample(toRev.attr().region());
      break;
    }
    case Type::Reverse: {
      rev.reverse(toRev.attr().dimensions());
      break;
    }
    case Type::DimShuffle: {
      rev.dimShuffle(toRev.attr().permutation().inverse());
      break;
    }
    }
  }

  return rev;
}

void Chain::verifyOp(uint64_t i) const {

  const auto &op_ = op(i);

  switch (op_.type()) {
  case Type::Reverse: {
    for (auto d : op_.attr().dimensions().get()) {
      if (d > inShape(i).rank_u64()) {
        throw error("Invalid reverse dimension " + std::to_string(d) +
                    " for reverse with input shape " + inShape(i).str());
      }
    }
    break;
  }
  case Type::Reshape: {
    outShape(i).assertSameNumberOfElements(inShape(i));
    break;
  }
  case Type::SettFillInto: {
    if (inShape(i) != region(i).nelms()) {
      std::ostringstream oss;
      oss << "Chain::settFillInto: Region r has nelms="
          << Shape(region(i).nelms()) << ". This should match inShape, "
          << inShape(i) << ". This for Chain\n"
          << *this;
      throw error(oss.str());
    }

    if (outShape(i) != region(i).shape()) {
      std::ostringstream oss;
      oss << "Chain::settFillInto: outShape is " << outShape(i)
          << " but region shape is " << region(i).shape()
          << ". They should be the same. ";
      throw error(oss.str());
    }
    break;
  }
  case Type::SettSample: {
    if (outShape(i) != region(i).nelms()) {
      std::ostringstream oss;
      oss << "Chain::settSample: Region's nelms, " << Shape(region(i).nelms())
          << " is not the same as outShape, " << outShape(i);
      throw error(oss.str());
    }
    if (inShape(i) != region(i).shape()) {
      std::ostringstream oss;
      oss << "Chain::settSample: inShape is " << inShape(i)
          << " but region shape is " << region(i).shape()
          << ". They should be the same. ";
      throw error(oss.str());
    }
    break;
  }
  case Type::DimShuffle: {
    break;
  }
  case Type::Expand: {
    inShape(i).assertCanExpandTo(outShape(i));
    if (inShape(i).rank_u64() != outShape(i).rank_u64()) {
      throw error("Expand Ops in the Chain class must be rank preserving");
    }
    break;
  }
  case Type::Reduce: {
    inShape(i).assertCanReduceTo(outShape(i));
    if (inShape(i).rank_u64() != outShape(i).rank_u64()) {
      throw error("Reduce Ops in the Chain class must be rank preserving");
    }
    break;
  }
  }
}

void Chain::reshape(const Shape &s) { append(Type::Reshape, s, s); }

void Chain::reduce(const Shape &s) {

  // do the reduction in 2 parts.
  // 1) rank-preserving reduction
  // 2) reshape
  const auto surplus = outShape().rank_u64() - s.rank_u64();
  const auto iShape  = s.prependOnes(surplus);
  append(Type::Reduce, iShape, iShape);
  append(Type::Reshape, s, s);
}

void Chain::expand(const Shape &s) {

  // do the expansion in 2 parts.
  // 1) reshape
  // 2) rank-preserving expand.
  const auto surplus = s.rank_u64() - outShape().rank_u64();
  const auto iShape  = outShape().prependOnes(surplus);
  append(Type::Reshape, iShape, iShape);
  append(Type::Expand, s, s);
}

void Chain::settSample(const Region &r) {
  append(Type::SettSample, {r.nelms()}, r);
}

void Chain::settSample(const std::vector<nest::Sett> &setts) {
  settSample(Region(outShape(), setts));
}

void Chain::settFillInto(const Region &r) {
  append(Type::SettFillInto, r.shape(), r);
}

void Chain::settFillInto(const Lower &l, const Upper &u) {
  const auto oShape = outShape().pad(l, u);
  settFillInto(Region::fromBounds(oShape, l, outShape().addToDims(l).get()));
}

void Chain::settFillInto(Stride s, Dimension d) {
  const auto oShape = outShape().scale(s, d);
  settFillInto(Region::fromStrideAndDim(oShape, s, d));
}

void Chain::reverse(const Dimensions &d) {
  append(Type::Reverse,
         outShape(),
         Dimensions(outShape().getCanonicalReverseIndices(d.get())));
}

void Chain::reverse(Dimension d) { reverse(Dimensions({d})); }

void Chain::dimShuffle(const Permutation &p) {
  return append(Type::DimShuffle, outShape().dimShuffle(p), p);
}

bool Chain::tryMergeLastTwo() {
  if (nOps() < 2) {
    return false;
  }

  /**
   * A SettFill followed by a SettSample of the exact same Region results in
   * the initial region.
   *
   *  region ->
   *   V     region.setFill(region0) ->
   *          V      region.setFill(region0).settSample(region0).
   *                  V
   *
   *          .
   *   x      .       x
   *   x      x       x
   *   x      .       x
   *          .
   *          x
   *
   */

  if (type(nOps() - 2) == Type::SettFillInto &&
      type(nOps() - 1) == Type::SettSample &&
      region(nOps() - 1).equivalent(region(nOps() - 2))) {
    popBack();
    popBack();
    return true;
  }

  /**
   * A SettSample followed by a SettFillInto of the same Region results in
   * the initial region, if the filtering Region contains the full input.
   * */
  if (type(nOps() - 2) == Type::SettSample &&
      type(nOps() - 1) == Type::SettFillInto) {

    if (region(nOps() - 2).equivalent(region(nOps() - 1))) {

      // Apply all but last 2 Ops to the full input:
      const auto regs = apply(Region::createFull(inShape(0)), nOps() - 2);

      // If the sampling region contains 'regs', then it will contain all
      // inputs.
      if (DisjointRegions({region(nOps() - 2)}).contains(regs)) {
        popBack();
        popBack();
        return true;
      }
    }
  }

  // Case of 2 contiguous Ops of the same type. They can sometimes be merged.
  if (type(nOps() - 1) == type(nOps() - 2)) {

    const auto t     = type(nOps() - 1);
    const auto shape = outShape(nOps() - 1);
    switch (t) {

    // Merging a sub-Chain of Reshapes, or of Expands, or of Reduces is
    // simple: Just jump straight to the final Shape.
    case Type::Reshape:
    case Type::Expand:
    case Type::Reduce: {
      popBack();
      popBack();
      append(t, shape, shape);
      return true;
    }

    // Merging DimShuffles consists of composing (multiplying) all of the
    // Permutations together.
    case Type::DimShuffle: {
      const auto p = permutation(nOps() - 2).mul(permutation(nOps() - 1));
      popBack();
      popBack();
      dimShuffle(p);
      return true;
    }

    // Merging Reverses consists of simply concatenating the Dimensions of
    // reversal.
    case Type::Reverse: {
      const auto dims = dimensions(nOps() - 2).append(dimensions(nOps() - 1));
      popBack();
      popBack();
      reverse(dims);
      return true;
    }

    // Merging SettSamples (slices, subSamples):
    // Starting at the end of the sub-Chain, fill the sampling Region into
    // the preceding SettSample's Region.
    case Type::SettSample: {
      auto merged = region(nOps() - 1).settFillInto(region(nOps() - 2));

      if (merged.size() > 1) {
        // It's not possible to merge these SettSamples, their Regions are
        // not compatible for merging (they are "co-prime").
        break;
      }
      popBack();
      popBack();
      settSample(merged.at(0));
      return true;
    }

    // Merging SettFillInto:
    case Type::SettFillInto: {
      auto merged = region(nOps() - 2).settFillInto(region(nOps() - 1));
      if (merged.size() > 1) {
        // It's not possible to merge these SettFillIntos, their Regions are
        // not compatible for merging (they are "co-prime").
        break;
      }
      popBack();
      popBack();
      settFillInto(merged.at(0));
      return true;
    }
    }
  }

  return false;
}

bool Chain::isIdentity(uint64_t opIndex) const {
  switch (type(opIndex)) {
  case Type::Reshape:
  case Type::Expand:
  case Type::Reduce: {
    return inShape(opIndex) == outShape(opIndex);
  }
  case Type::DimShuffle: {
    return permutation(opIndex).isIdentity();
  }
  case Type::Reverse: {
    return dimensions(opIndex).empty();
  }

  case Type::SettFillInto:
  case Type::SettSample:
    return region(opIndex).full();
  }

  throw error("Exited switch in Chain::isIdentity without returning");
}

// TODO(T32931) check for "high probability" of equivalence.
//   bool Chain::randomRegionMappingEquivalent(const Chain &rhs,
//                                      uint64_t nRandomRegions) const;

void Chain::canonicalize() {

  // Check if the full Region gets mapped to the empty Region, if it does,
  // it can be represented as a simple mask.
  if (apply(DisjointRegions::createFull(inShape())).empty()) {
    ops_.uptr->ops.clear();
    mask(Region::createEmpty(outShape()));
    return;
  }

  bool changed{true};

  // Iteratively try canonicalization passes, until the Chain is unchanged.
  // It is guaranteed that this does terminate, as there are no circular
  // sequences of passes: they all either reduce the number of Ops, or move
  // the Chain towards alphabetical order.
  //

  while (changed) {
    changed = false;

    // Store and clear the current Ops.
    auto oldOps = ops_.uptr->ops;
    ops_.uptr->ops.clear();

    for (const auto &oldOp : oldOps) {

      // Push the old Op into the new Chain. The passes which follow will
      // try and remove / move it.
      append(oldOp);

      // remove identities.
      while (nOps() > 0 && isIdentity(nOps() - 1)) {
        popBack();
        changed = true;
      }

      // merge or remove the back 2 Ops.
      bool blocked = false;
      while (!blocked) {
        auto merged = tryMergeLastTwo();
        if (merged) {
          changed = true;
        } else {
          blocked = true;
        }
      }

      // try bubbling the Op backwards, towards the front of the Chain.
      // we try and keep Ops in alphabetical order. This makes the pass
      // which merges compatible Ops more likely to succeed, so that even
      // though this pass does not reduce the number of Ops in this Chain,
      // it makes it more likely for other Op-reducing passes to succeed.

      if (nOps() > 1) {
        uint64_t current = nOps() - 1;
        bool bubbled{true};
        while (bubbled) {
          bubbled = tryBubbleBack(current);
          changed = changed || bubbled;
          current -= 1;
        }
      }
    }
  }

  for (uint64_t i = 0; i < nOps(); ++i) {
    verifyOp(i);
  }
}

bool Chain::tryBubbleBack(uint64_t i1) {

  // Cannot bubble back if already at the front.
  if (i1 == 0) {
    return false;
  }

  const auto i0 = i1 - 1;

  auto &op0 = op(i0);
  auto &op1 = op(i1);

  const auto inShape0 = inShape(i0);

  const auto t0 = op0.type();
  const auto t1 = op1.type();

  // Only back if (op0, op1) are not lexicographically sorted.
  // TODO(T53918): make this a chain attribute, which the user can set.
  if (static_cast<uint64_t>(t0) <= static_cast<uint64_t>(t1)) {
    return false;
  }

  switch (t1) {

    // Try bubbling DimShuffle back.
  case Type::DimShuffle: {
    return Op::bubbleDimShuffleBack(inShape0, op0, op1);
  }

  case Type::Expand: {
    return Op::bubbleExpandBack(inShape0, op0, op1);
  }

  case Type::Reduce: {
    return Op::bubbleReduceBack(inShape0, op0, op1);
  }

  case Type::Reshape: {
    return Op::bubbleReshapeBack(inShape0, op0, op1);
  }

  case Type::Reverse: {
    return Op::bubbleReverseBack(inShape0, op0, op1);
  }

  case Type::SettSample: {
    return Op::bubbleSettSampleBack(inShape0, op0, op1);
  }

  case Type::SettFillInto: {
    return Op::bubbleSettFillIntoBack(inShape0, op0, op1);
  }
  }

  throw error("Unhandled type in tryBubbleBack:  " + getTypeString(t1));
}

Shape Chain::inShape(uint64_t opIndex) const {
  if (opIndex == 0) {
    return inShape();
  }
  return outShape(opIndex - 1);
}

Shape Chain::outShape() const {
  if (nOps() == 0) {
    return inShape();
  }
  return outShape(nOps() - 1);
}

void Chain::append(std::ostream &ost, uint64_t opIndex) const {
  const auto shape = outShape(opIndex);
  ost << getTypeString(type(opIndex)) << '(';
  switch (type(opIndex)) {
  case Type::Reshape:
  case Type::Expand:
  case Type::Reduce: {
    ost << shape << ')';
    break;
  }
  case Type::DimShuffle: {
    ost << permutation(opIndex) << ')';
    break;
  }
  case Type::Reverse: {
    ost << dimensions(opIndex) << ')';
    break;
  }
  case Type::SettSample:
  case Type::SettFillInto: {
    ost << region(opIndex).setts() << ')';
    break;
  }
  }
}

void Chain::appendCompact(std::ostream &ost) const {
  ost << '(';
  for (uint64_t i = 0; i < nOps(); ++i) {
    append(ost, i);
    if (i != nOps() - 1) {
      ost << ',';
    }
  }
  ost << ')';
}

void Chain::append(std::ostream &ost) const {
  if (nOps() == 0) {
    ost << "(empty" << inShape() << ')';
    return;
  }

  const std::string opening = inShape().str() + std::string(" ----> ");
  const std::string spacey  = std::string(opening.size(), ' ');
  ost << opening;
  for (uint64_t i = 0; i < nOps(); ++i) {
    if (i != 0) {
      ost << '\n' << spacey;
    }
    append(ost, i);
  }
  ost << " ----> " << outShape();
}

std::ostream &operator<<(std::ostream &ost, const Chain &ch) {
  ch.append(ost);
  return ost;
}

Shape Chain::outShape(uint64_t opIndex) const {
  return op(opIndex).outShape();
}

uint64_t Chain::nOps() const { return ops_.uptr->ops.size(); }

void Chain::slice(const Lower &l, const Upper &u) {
  settSample(Region::fromBounds(outShape(), l, u));
}

void Chain::subSample(Stride s, Dimension d) {
  settSample(Region::fromStrideAndDim(outShape(), s, d));
}

Shape Chain::inShape() const { return inShape_; }

bool Chain::operator==(const Chain &rhs) const {
  if (inShape() != rhs.inShape()) {
    return false;
  }
  return ops_ == rhs.ops_;
}

void Chain::confirmNotEqual(const Chain &rhs) const {
  if (*this == rhs) {
    std::ostringstream oss;
    oss << "Failed in confirmNotEqual. "
        << "This Chain is \n"
        << *this << ", rhs Chain is \n"
        << rhs << '.';
    throw error(oss.str());
  }
}

void Chain::confirmEqual(const Chain &rhs, const std::string &ctxt) const {
  if (*this != rhs) {
    std::ostringstream oss;
    oss << "Failed in confirmEqual. "
        << "This Chain is \n"
        << *this << ", rhs Chain is \n"
        << rhs << '.';
    if (!ctxt.empty()) {
      oss << "\nContext: " << ctxt;
    }
    throw error(oss.str());
  }
}

std::vector<uint64_t> Chain::where(Type t) const {
  std::vector<uint64_t> where_;
  for (uint64_t i = 0; i < nOps(); ++i) {
    if (type(i) == t) {
      where_.push_back(i);
    }
  }
  return where_;
}

const Op &Chain::op(uint64_t i) const { return ops_.uptr->ops[i]; }
Op &Chain::op(uint64_t i) { return ops_.uptr->ops[i]; }

Region Chain::region(uint64_t id) const { return op(id).attr().region(); }
Permutation Chain::permutation(uint64_t id) const {
  return op(id).attr().permutation();
}
Dimensions Chain::dimensions(uint64_t id) const {
  return op(id).attr().dimensions();
}
Type Chain::type(uint64_t id) const { return op(id).type(); }

Chain::~Chain()             = default;
Chain::Chain(const Chain &) = default;
Chain::Chain(Chain &&)      = default;

Chain &Chain::operator=(const Chain &) = default;
Chain &Chain::operator=(Chain &&) = default;

void Chain::append(const Chain &rhs) {

  if (outShape() != rhs.inShape()) {

    std::ostringstream oss;
    oss << "Failure in Chain::append, "
        << "where the end of this Chain has Shape " << outShape()
        << ", and the start of rhs has Shape " << rhs.inShape()
        << ". These are not compatible. ";
    throw error(oss.str());
  }

  ops_.uptr->ops.insert(ops_.uptr->ops.end(),
                        rhs.ops_.uptr->ops.cbegin(),
                        rhs.ops_.uptr->ops.cend());
}

void Chain::mask(const Region &r) {
  settSample(r);
  settFillInto(r);
}

Chain Chain::canonicalized() const {
  auto c = *this;
  c.canonicalize();
  return c;
}

void Chain::popBack() { ops_.uptr->ops.pop_back(); }

Chain::Chain(const Shape &s) : ops_(std::make_unique<Ops>()), inShape_(s) {}

} // namespace chain
} // namespace memory

namespace util {
template class CopyByClone<memory::chain::Chain::Ops>;
}

} // namespace poprithms
