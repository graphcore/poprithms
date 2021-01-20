// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <numeric>
#include <sstream>
#include <variant>

#include <poprithms/memory/chain/chain.hpp>
#include <poprithms/memory/chain/error.hpp>
#include <util/copybyclone_impl.hpp>

namespace poprithms {
namespace memory {
namespace chain {

// C++ 17, so this cannot appear in chain.hpp as poprithms is C++11 API.
using Variant = std::variant<Shape, Region, Permutation, Dimensions>;

/** An attribute of an Op. The attribute is one of
 *     Shape,
 *     Region,
 *     Permutation, and
 *     Dimensions.
 *
 * The use of std::variant ensures that
 * sizeof(Attr) =
 * max(sizeof(Shape), sizeof(Region), sizeof(Permutation), sizeof(Dimensions)
 * */
class Chain::Attr {
public:
  template <class T> Attr(const T &t) : v(t) {}

  const Shape &shape() const { return std::get<Shape>(v); }
  const Region &region() const { return std::get<Region>(v); }
  const Permutation &permutation() const { return std::get<Permutation>(v); }
  const Dimensions &dimensions() const { return std::get<Dimensions>(v); }
  const Variant &var() const { return v; }

private:
  Variant v;
};

class Chain::Op {
public:
  Op(Type t, const Shape &o, const Attr &a)
      : type_(t), outShape_(o), attr_(a) {}
  Type type() const { return type_; }
  Shape outShape() const { return outShape_; }
  const Attr &attr() const { return attr_; }

  bool operator==(const Op &) const;
  bool operator!=(const Op &rhs) const { return !operator==(rhs); }

private:
  Type type_;
  Shape outShape_;
  Attr attr_;
};

class Chain::Ops {
public:
  bool operator==(const Ops &rhs) const { return ops == rhs.ops; }
  bool operator!=(const Ops &rhs) const { return !operator==(rhs); }
  std::unique_ptr<Ops> clone() const { return std::make_unique<Ops>(*this); }
  std::vector<Op> ops;
};

template <typename X>
Chain &Chain::append(Type t, const Shape &o, const X &a) {
  ops_.uptr->ops.push_back({t, o, a});
  return *this;
}

Chain &Chain::append(const Op &op) {
  ops_.uptr->ops.push_back(op);
  return *this;
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

Chain &Chain::reshape(const Shape &s) {
  outShape().assertSameNumberOfElements(s);
  return append(Type::Reshape, s, s);
}

Chain &Chain::reduce(const Shape &s) {
  outShape().assertCanReduceTo(s);
  return append(Type::Expand, s, s);
}

Chain &Chain::expand(const Shape &s) {
  outShape().assertCanExpandTo(s);
  return append(Type::Reduce, s, s);
}

Chain &Chain::settSample(const Region &r) {
  if (outShape() != r.shape()) {
    std::ostringstream oss;
    oss << "Chain::settSample: Region's Shape, " << r.shape()
        << " is not the same as outShape(), " << outShape();
    throw error(oss.str());
  }
  return append(Type::SettSample, {r.nelms()}, r);
}

Chain &Chain::settSample(const std::vector<nest::Sett> &setts) {
  return settSample(Region(outShape(), setts));
}

Chain &Chain::settFillInto(const Region &r) {
  if (outShape() != r.nelms()) {
    std::ostringstream oss;
    oss << "Chain::settFillInto: Region r has nelms=" << Shape(r.nelms())
        << ". This should match outShape, " << outShape();
    throw error(oss.str());
  }
  return append(Type::SettFillInto, r.shape(), r);
}

Chain &Chain::settFillInto(const Lower &l, const Upper &u) {
  const auto oShape = outShape().pad(l, u);
  return settFillInto(
      Region::fromBounds(oShape, l, outShape().addToDims(l).get()));
}

Chain &Chain::settFillInto(Stride s, Dimension d) {
  const auto oShape = outShape().scale(s, d);
  return settFillInto(Region::fromStrideAndDim(oShape, s, d));
}

Chain &Chain::reverse(const Dimensions &d) {
  return append(Type::Reverse,
                outShape(),
                Dimensions(outShape().getCanonicalReverseIndices(d.get())));
}

Chain &Chain::dimShuffle(const Permutation &p) {
  return append(Type::DimShuffle, outShape().dimShuffle(p), p);
}

Chain Chain::removeIdentity() const {
  Chain after(inShape());
  for (uint64_t i = 0; i < nOps(); ++i) {
    if (!isIdentity(i)) {
      after.append(op(i));
    }
  }
  return after;
}

Chain Chain::mergeContiguousSameType() const {
  Chain after(inShape());
  uint64_t start{0};
  while (start < nOps()) {

    // from "start", move "stop" forward until an Op of a different type is
    // found.
    auto stop = start + 1;
    while (stop < nOps() && type(stop) == type(start)) {
      ++stop;
    }

    // The interval [start, stop) contains at least 2 Ops of the same type.
    if (stop - start > 1) {
      const auto t     = type(start);
      const auto shape = outShape(stop - 1);
      switch (t) {

      // Merging a sub-Chain of Reshapes, or of Expands, or of Reduces is
      // simple: Just jump straight to the final Shape.
      case Type::Reshape:
      case Type::Expand:
      case Type::Reduce: {
        after.append(type(start), shape, shape);
        break;
      }

      // Merging DimShuffles consists of composing (multiplying) all of the
      // Permutations together.
      case Type::DimShuffle: {
        const Permutation p =
            std::accumulate(ops_.uptr->ops.cbegin() + start,
                            ops_.uptr->ops.cbegin() + stop,
                            Permutation::identity(shape.rank_u64()),
                            [](const Permutation &a, const Op &b) {
                              return a.mul(b.attr().permutation());
                            });
        after.dimShuffle(p);
        break;
      }

      // Merging Reverses consists of simply concatenating the Dimensions of
      // reversal.
      case Type::Reverse: {
        std::vector<uint64_t> allDims;
        for (uint64_t i = start; i < stop; ++i) {
          const auto nxt = dimensions(i).get();
          allDims.insert(allDims.end(), nxt.cbegin(), nxt.cend());
        }
        after.reverse(Dimensions(allDims));
        break;
      }

      // Merging SettSamples (slices, subSamples):
      // Starting at the end of the sub-Chain, fill the sampling Region into
      // the preceding SettSample's Region.
      case Type::SettSample: {
        uint64_t current = stop - 1;
        std::vector<Region> rev;
        auto merged = region(current);
        while (current > start) {
          auto nxt = merged.settFillInto(region(current - 1));
          if (nxt.size() == 1) {
            merged = nxt.at(0);
          } else {
            rev.push_back(merged);
            merged = region(current - 1);
          }
          --current;
        }
        rev.push_back(merged);
        for (auto pr = rev.crbegin(); pr != rev.crend(); ++pr) {
          after.settSample(*pr);
        }
        break;
      }

      // Merging SettFillInto:
      case Type::SettFillInto: {
        uint64_t current = start;
        auto merged      = region(current);
        while (current < stop - 1) {
          auto nxt = merged.settFillInto(region(current + 1));
          if (nxt.size() == 1) {
            merged = nxt.at(0);
          } else {
            after.settFillInto(merged);
            merged = region(current + 1);
          }
          ++current;
        }
        after.settFillInto(merged);
      }
      }
    } else {
      after.append(op(start));
    }
    start = stop;
  }
  return after;
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

// TODO(T32929) improve complexity of this algorithm.
Chain Chain::canonicalize() const {

  auto c0 = *this;
  auto n0 = nOps();
  bool converged{false};
  while (!converged) {
    converged = true;
    auto c1   = c0.removeIdentity().mergeContiguousSameType();
    // TODO(T32930) a pass to sort the Ops.

    auto n1 = c1.nOps();

    if (n1 != n0) {
      converged = false;
    }
    c0 = c1;
    n0 = n1;
  }
  return c0;
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
  switch (type(opIndex)) {
  case Chain::Type::Reshape: {
    ost << "Reshape(" << shape << ')';
    break;
  }
  case Chain::Type::Expand: {
    ost << "Expand(" << shape << ')';
    break;
  }
  case Chain::Type::Reduce: {
    ost << "Reduce(" << shape << ')';
    break;
  }
  case Chain::Type::DimShuffle: {
    ost << "DimShuffle(" << permutation(opIndex) << ')';
    break;
  }
  case Chain::Type::Reverse: {
    ost << "Reverse(" << dimensions(opIndex) << ')';
    break;
  }
  case Chain::Type::SettSample: {
    ost << "SettSample(" << region(opIndex).setts() << ')';
    break;
  }
  case Chain::Type::SettFillInto: {
    ost << "SettFillInto(" << region(opIndex).setts() << ')';
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

Chain &Chain::slice(const Lower &l, const Upper &u) {
  return settSample(Region::fromBounds(outShape(), l, u));
}

Chain &Chain::subSample(Stride s, Dimension d) {
  return settSample(Region::fromStrideAndDim(outShape(), s, d));
}

Shape Chain::inShape() const { return inShape_; }

namespace {

template <typename> struct tag {};

template <typename T, typename V> struct getIndex;

template <typename T, typename... Ts>
struct getIndex<T, std::variant<Ts...>>
    : std::integral_constant<size_t,
                             std::variant<tag<Ts>...>(tag<T>()).index()> {};

template <typename T> struct getVariantIndex : getIndex<T, Variant> {};

} // namespace

bool Chain::Op::operator==(const Op &rhs) const {
  if (type() != rhs.type()) {
    return false;
  }

  switch (attr().var().index()) {
  case (getVariantIndex<Region>()): {
    return attr().region().equivalent(rhs.attr().region());
  }
  case (getVariantIndex<Shape>()): {
    return attr().shape() == rhs.attr().shape();
  }
  case (getVariantIndex<Permutation>()): {
    return attr().permutation() == rhs.attr().permutation();
  }
  case (getVariantIndex<Dimensions>()): {
    return attr().dimensions() == rhs.attr().dimensions();
  }
  }

  throw error("Exited switch in Chain::operator== without returning");
}

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

void Chain::confirmEqual(const Chain &rhs) const {
  if (*this != rhs) {
    std::ostringstream oss;
    oss << "Failed in confirmEqual. "
        << "This Chain is \n"
        << *this << ", rhs Chain is \n"
        << rhs << '.';
    throw error(oss.str());
  }
}

const Chain::Op &Chain::op(uint64_t i) const { return ops_.uptr->ops[i]; }
Chain::Op &Chain::op(uint64_t i) { return ops_.uptr->ops[i]; }

Region Chain::region(uint64_t id) const { return op(id).attr().region(); }
Permutation Chain::permutation(uint64_t id) const {
  return op(id).attr().permutation();
}
Dimensions Chain::dimensions(uint64_t id) const {
  return op(id).attr().dimensions();
}
Chain::Type Chain::type(uint64_t id) const { return op(id).type(); }

Chain::~Chain()             = default;
Chain::Chain(const Chain &) = default;
Chain::Chain(Chain &&)      = default;

Chain &Chain::operator=(const Chain &) = default;
Chain &Chain::operator=(Chain &&) = default;

Chain &Chain::append(const Chain &rhs) {

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

  return *this;
}

Chain::Chain(const Shape &s) : ops_(std::make_unique<Ops>()), inShape_(s) {}

} // namespace chain
} // namespace memory

namespace util {
template class CopyByClone<memory::chain::Chain::Ops>;
}
} // namespace poprithms
