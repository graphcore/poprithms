// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <sstream>

#include <poprithms/memory/unwind/error.hpp>
#include <poprithms/memory/unwind/path.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

Path::Path(const TensorId &src,
           const chain::Chain &chain,
           const TensorId &dst)
    : src_(src), chain_(chain), dst_(dst),
      dstRegions_(chain.apply(DisjointRegions::createFull(chain.inShape()))) {

  if (chain.outShape() != dstRegions_.shape()) {
    std::ostringstream oss;
    oss << "Incompatible Chain "
        << "and destination DisjointRegions in Path constructor. ";
    oss << "Chain = \n" << chain << " and dstRegions = \n" << dstRegions_;
    throw error(oss.str());
  }
}

std::string Path::str() const {
  std::ostringstream oss;
  append(oss);
  return oss.str();
}

void Path::append(std::ostream &ost) const {
  ost << "Source=" << src() << ",  Destination=" << dst() << ",  Chain=";
  chain().appendCompact(ost);
  ost << ", To=" << dstRegions();
}

std::ostream &operator<<(std::ostream &ost, const Path &p) {
  p.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const Paths &ps) {
  for (const auto &p : ps) {
    ost << "\n   " << p;
  }
  return ost;
}

void Link::append(std::ostream &ost) const {
  if (isFwd()) {
    ost << "in=" << inIndex() << "->op=" << opId() << "->out=" << outIndex();
  } else {
    ost << "out=" << outIndex() << "->op=" << opId() << "->in=" << inIndex();
  }
}

} // namespace unwind
} // namespace memory
} // namespace poprithms
