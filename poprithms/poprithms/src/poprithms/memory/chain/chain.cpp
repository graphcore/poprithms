// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <numeric>
#include <sstream>
#include <unordered_set>
#include <variant>

#include <memory/chain/disjointregionsmapper.hpp>
#include <memory/chain/error.hpp>
#include <memory/chain/op.hpp>

#include <poprithms/compute/host/tensormapper.hpp>
#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/chain/type.hpp>
#include <poprithms/util/copybyclone_impl.hpp>
#include <poprithms/util/printiter.hpp>

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
  return apply<compute::host::TensorMapper, compute::host::Tensor>(t);
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

bool Chain::tryMakeIdentitiesWithPrevious(uint64_t indexSecond) {

  if (indexSecond == 0) {
    throw error("The second index in the pair cannot be 0 (as the first "
                "would then be -1). ");
  }

  if (indexSecond >= nOps()) {
    throw error(
        "The second index in the pair exceeds or equals the chain length");
  }

  auto indexFirst = indexSecond - 1;
  auto &op0       = op(indexFirst);
  auto &op1       = op(indexSecond);

  const auto inShape0  = inShape(indexFirst);
  const auto outShape0 = inShape(indexSecond);
  const auto outShape1 = outShape(indexSecond);

  auto makeTwoIdentities = [&inShape0, &outShape0, &outShape1, &op0, &op1]() {
    (void)outShape0;
    if (inShape0 != outShape1) {
      throw error("To create 2 identities, the first input shape must equal "
                  "the final output shape");
    }

    // reshape to same shape is an identity op
    op0 = {Type::Reshape, inShape0, inShape0};
    op1 = {Type::Reshape, inShape0, inShape0};
  };

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

  if (type(indexFirst) == Type::SettFillInto &&
      type(indexSecond) == Type::SettSample &&
      region(indexSecond).equivalent(region(indexFirst))) {
    makeTwoIdentities();
    return true;
  }

  // as above, but Expand instead of SettFillInto.
  if (type(indexFirst) == Type::Expand &&
      type(indexSecond) == Type::SettSample && inShape0 == outShape1) {
    makeTwoIdentities();
    return true;
  }

  /**
   * A SettSample followed by a SettFillInto of the same Region results in
   * the initial region, if the filtering Region contains the full input.
   * */
  if (type(indexFirst) == Type::SettSample &&
      type(indexSecond) == Type::SettFillInto) {

    if (region(indexFirst).equivalent(region(indexSecond))) {

      // Apply all but last 2 Ops to the full input:
      const auto regs = apply(Region::createFull(inShape(0)), indexFirst);

      // If the sampling region contains 'regs', then it will contain all
      // inputs.
      if (DisjointRegions({region(indexFirst)}).contains(regs)) {
        makeTwoIdentities();
        return true;
      }
    }
  }

  // Case of 2 contiguous Ops of the same type. They can sometimes be merged.
  if (type(indexSecond) == type(indexFirst)) {

    const auto t     = type(indexSecond);
    const auto shape = outShape(indexSecond);
    switch (t) {

    // Merging a sub-Chain of Reshapes, or of Expands, or of Reduces is
    // simple: Just jump straight to the final Shape.
    case Type::Reshape:
    case Type::Expand:
    case Type::Reduce: {
      op0 = {Type::Reshape, inShape0, inShape0};
      op1 = {t, shape, shape};
      return true;
    }

    // Merging DimShuffles consists of composing (multiplying) all of the
    // Permutations together.
    case Type::DimShuffle: {
      const auto p = permutation(indexFirst).mul(permutation(indexSecond));
      op0          = {Type::Reshape, inShape0, inShape0};
      op1          = {Type::DimShuffle, shape, p};
      return true;
    }

    // Merging Reverses consists of simply concatenating the Dimensions of
    // reversal.
    case Type::Reverse: {
      auto dims = dimensions(indexFirst).append(dimensions(indexSecond));
      dims      = Dimensions(shape.getCanonicalReverseIndices(dims.get()));
      op0       = {Type::Reshape, inShape0, inShape0};
      op1       = {Type::Reverse, shape, dims};
      return true;
    }

    // Merging SettSamples (slices, subSamples):
    // Starting at the end of the sub-Chain, fill the sampling Region into
    // the preceding SettSample's Region.
    case Type::SettSample: {
      auto merged = region(indexSecond).settFillInto(region(indexFirst));

      if (merged.size() > 1) {
        // It's not possible to merge these SettSamples, their Regions are
        // not compatible for merging (they are "co-prime").
        break;
      }

      op0 = {Type::Reshape, inShape0, inShape0};
      op1 = {Type::SettSample, shape, merged.at(0)};
      return true;
    }

    // Merging SettFillInto:
    case Type::SettFillInto: {
      auto merged = region(indexFirst).settFillInto(region(indexSecond));
      if (merged.size() > 1) {
        // It's not possible to merge these SettFillIntos, their Regions are
        // not compatible for merging (they are "co-prime").
        break;
      }
      op0 = {Type::Reshape, inShape0, inShape0};
      op1 = {Type::SettFillInto, shape, merged.at(0)};
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

void Chain::canonicalize(const Types &targetOrder) {

  // assert that #targetOrder has no duplicates and is not missing any Types
  // in this Chain.
  {
    std::unordered_set<Type> targetOrderSet(targetOrder.cbegin(),
                                            targetOrder.cend());
    for (const auto &op : ops_.uptr->ops) {
      if (targetOrderSet.count(op.type()) == 0) {
        std::ostringstream oss;
        oss << "The op type " << op.type()
            << " is present in this chain, but is missing from the target "
               "list.";
        poprithms::util::append(oss, targetOrder);
        throw error(oss.str());
      }
    }

    if (targetOrderSet.size() != targetOrder.size()) {
      std::ostringstream oss;
      oss << "There is duplication in the target order vector ";
      util::append(oss, targetOrder);
      oss << '.';
      throw error(oss.str());
    }
  }

  auto getOrderPosition = [&targetOrder](Type t) -> uint64_t {
    for (uint64_t i = 0; i < targetOrder.size(); ++i) {
      if (targetOrder[i] == t) {
        return i;
      }
    }
    throw error("Unexpected, should have found the type");
  };

  // Check if the full Region gets mapped to the empty Region. If it does,
  // it can be represented as a simple mask. This starts by checking for an op
  // of type SettFillInto, because without an op of this type it is impossible
  // to map to the empty region (early+fast exit if present).
  if (contains(Type::SettFillInto) &&
      apply(DisjointRegions::createFull(inShape())).empty()) {
    ops_.uptr->ops.clear();
    mask(Region::createEmpty(outShape()));
    return;
  }

  bool changed{true};

  // Iteratively try canonicalization passes, until the Chain is unchanging.
  // It is guaranteed that this does terminate, as there are no circular
  // sequences of passes: they all either reduce the number of Ops, or move
  // the Chain towards the specified lexicographic order.
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

      // try bubbling the Op backwards, towards the front of the Chain.
      // We try and keep Ops in the specified order. This makes the pass
      // which merges compatible Ops more likely to succeed, so that even
      // though this pass does not reduce the number of Ops in this Chain,
      // it makes it more likely for other Op-reducing passes to succeed.

      auto processBack = [this, &getOrderPosition]() -> bool {
        uint64_t current = nOps() - 1;
        bool keepBubbling{true};
        bool changed_{false};
        while (keepBubbling && current != 0) {

          auto madeToIdentity = tryMakeIdentitiesWithPrevious(current);
          if (madeToIdentity) {
            return true;
          }

          else {

            // Only back if (op0, op1) are not in specified type order
            if (getOrderPosition(type(current - 1)) <=
                getOrderPosition(type(current))) {
              keepBubbling = false;
            }

            else {
              keepBubbling = tryBubbleBack(current);
            }
            changed_ = changed_ || keepBubbling;
            current -= 1;
          }
        }
        return changed_;
      };

      if (nOps() > 1) {
        changed = changed || processBack();
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
  const auto t1       = op1.type();

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

bool Chain::contains(Type t) const {

  for (uint64_t i = 0; i < nOps(); ++i) {
    if (type(i) == t) {
      return true;
    }
  }
  return false;
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

Chain Chain::canonicalized(const Types &targetOrder) const {
  auto c = *this;
  c.canonicalize(targetOrder);
  return c;
}

Chain Chain::canonicalized() const {
  auto c = *this;
  c.canonicalize();
  return c;
}

void Chain::canonicalize() {
  canonicalize(TypeOrders::reverseAlphabetical());
  canonicalize(TypeOrders::alphabetical());
}

void Chain::popBack() { ops_.uptr->ops.pop_back(); }

Chain::Chain(const Shape &s) : ops_(std::make_unique<Ops>()), inShape_(s) {}

} // namespace chain
} // namespace memory

namespace util {
template class CopyByClone<memory::chain::Chain::Ops>;
}

} // namespace poprithms
