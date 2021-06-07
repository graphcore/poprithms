// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

#include <poprithms/common/multiout/error.hpp>
#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/common/multiout/op.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>
#include <util/copybyclone_impl.hpp>

namespace poprithms {
namespace common {
namespace multiout {

void Graph::confirmValidTensorId(const TensorId &tId) const {

  if (tId.opId().get() >= static_cast<int64_t>(ops_.size())) {
    std::ostringstream oss;
    oss << "Failure in confirmValidTensorId(TensorId=" << tId << "). "
        << "In total only " << ops_.size()
        << " Ops have ever been created in this Graph. ";
    throw error(oss.str());
  }

  if (!ops_[tId.opId().get()].uptr) {
    std::ostringstream oss;
    oss << "Failure in confirmValidTensorId(TensorId=" << tId << "). "
        << "The Op " << tId.opId() << " no longer exists.";
    throw error(oss.str());
  }

  if (tId.outIndex().get() >= op(tId.opId()).nOutTensors()) {
    std::ostringstream oss;
    oss << "Failure in confirmValidTensorId(TensorId=" << tId
        << "). Invalid output index (" << tId.outIndex() << ") as Op "
        << tId.opId() << "(" << op(tId.opId()) << ") only has "
        << op(tId.opId()).nOutTensors() << " output Tensors. ";
    throw error(oss.str());
  }
}

uint64_t Graph::nInTensors(OpId id) const { return op(id).nInTensors(); }
uint64_t Graph::nOutTensors(OpId id) const { return op(id).nOutTensors(); }

OpId Graph::insertMultioutOp(std::unique_ptr<Op> createdOp) {

  for (uint64_t inIndex = 0; inIndex < createdOp->nInTensors(); ++inIndex) {
    const auto inTensor = createdOp->inTensorId(inIndex);
    const auto sourceId = inTensor.opId();
    auto &source        = op(sourceId);
    source.insertConsumptionId(inTensor.outIndex(),
                               {createdOp->id(), InIndex(inIndex)});
  }

  ops_.emplace_back(std::move(createdOp));

  const auto newId = ops_.back().uptr->id();

  live_.insert(/* hint = */ live_.end(), newId);

  return newId;
}

OpIds Graph::opIds(const TensorIds &tids) {
  OpIds ids_(tids.size());
  for (uint64_t i = 0; i < tids.size(); ++i) {
    ids_[i] = tids[i].opId();
  }
  return ids_;
}

void Graph::setName(const OpId id, const std::string &name) {
  op(id).setName(name);
}

const std::string &Graph::getName(OpId opId) const {
  return op(opId).getName();
}

bool Graph::operator==(const Graph &rhs) const {
  return getName() == rhs.getName() && ops_ == rhs.ops_ &&
         live_ == rhs.live_ && typeid(*this) == typeid(rhs) &&
         multiOutTypeSpecificEqualTo(rhs);
}

const Op &Graph::op(OpId a) const {
  if (a.get() >= nOps_i64()) {
    std::ostringstream oss;
    oss << "The number of Ops in this Graph is " << nOps()
        << ", so there is no Op with OpId " << a << '.';
    throw error(oss.str());
  }

  const auto &opPtr = ops_[static_cast<uint64_t>(a.get())].uptr;

  if (!opPtr) {
    throw error("nullptr in op(" + std::to_string(a.get()) + ").");
  }
  return *opPtr;
}

// See Scott Meyers' "Effective C++"
Op &Graph::op(OpId id) {
  return const_cast<Op &>(static_cast<const Graph &>(*this).op(id));
}

std::vector<Shape> Graph::shapes(const TensorIds &ids) const {
  std::vector<Shape> shapes;
  shapes.reserve(ids.size());
  for (const auto &id : ids) {
    shapes.push_back(shape(id));
  }
  return shapes;
}

Shape Graph::shape(const TensorId &tid) const {
  return op(tid.opId()).outShape(tid.outIndex());
}

ConsumptionIds Graph::consumptionIds(const TensorId &id) const {
  const auto cs = op(id.opId()).consumptionIds(id.outIndex());
  return cs;
}

TensorIds Graph::setDifference(const TensorIds &a_, const TensorIds &b_) {
  auto a = a_;
  auto b = b_;
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());

  TensorIds diff;
  diff.reserve(a.size() - b.size());

  std::set_difference(a.cbegin(),
                      a.cend(),
                      b.cbegin(),
                      b.cend(),
                      std::inserter(diff, diff.begin()));

  return diff;
}

uint64_t Graph::nOutTensors(const OpIds &ids) const {
  uint64_t n = 0;
  for (auto i : ids) {
    n += op(i).nOutTensors();
  }
  return n;
}

uint64_t Graph::nMultioutRows(const OpIds &opIds) const {
  return nOutTensors(opIds) + nWithZeroOutputs(opIds);
}

std::vector<std::string> Graph::getOpNames() const {
  std::vector<std::string> names;
  names.reserve(nOps());
  for (uint64_t i = 0; i < nOps(); ++i) {
    names.push_back(op(i).getName());
  }
  return names;
}

std::string Graph::typeString(OpId id) const { return op(id).typeString(); }

void Graph::verifyTensorId(const TensorId &tId) const {
  if (tId.opId() >= nOps()) {
    std::ostringstream oss;
    oss << "Failure in verifyTensorId(TensorId=" << tId
        << "), in this Graph, which only has " << nOps() << " Ops. "
        << "This TensorId has (creator) OpId " << tId.opId()
        << ", which was expected to be less than " << nOps() << ". ";
    throw error(oss.str());
  }

  const auto nOuts = op(tId.opId()).nOutTensors();
  if (tId.outIndex() >= nOuts) {
    std::ostringstream oss;
    oss << "Failure in verifyTensorId(TensorId=" << tId
        << "), as the creator Op, with OpId=" << tId.opId() << " only has "
        << nOuts << " outputs. ";
    throw error(oss.str());
  }
}

TensorIds Graph::tensorIds() const {
  TensorIds ids;
  ids.reserve(nTensors());
  for (uint64_t i = 0; i < nOps(); ++i) {
    for (uint64_t o = 0; o < op(i).nOutTensors(); ++o) {
      ids.push_back({i, o});
    }
  }
  return ids;
}

OpIds Graph::opIds() const { return OpIds(live_.cbegin(), live_.cend()); }

TensorIds Graph::outTensorIds(OpId id) const { return op(id).outTensorIds(); }

TensorIds Graph::inTensorIds(OpId id) const { return op(id).inTensorIds(); }

void Graph::setName(const TensorId &id, const std::string &name) {
  if (op(id.opId()).nOutTensors() != 1) {
    std::ostringstream oss;
    oss << "Cannot call Graph::setName(TensorId=" << id << ", name=" << name
        << "), because the Op creator has multiple outputs. "
        << "Call "
        << "setName(OpId=" << id.opId() << ", name=" << name << ") instead.";
    throw error(oss.str());
  }
  setName(id.opId(), name);
}

TensorIds Graph::inAndOutTensorIds(OpId i) const {
  TensorIds outs       = outTensorIds(i);
  TensorIds insAndOuts = inTensorIds(i);
  insAndOuts.insert(insAndOuts.end(), outs.cbegin(), outs.cend());
  return insAndOuts;
}

uint64_t Graph::nOpsWithZeroOutputs() const {
  return nWithZeroOutputs(opIds());
}

uint64_t Graph::nWithZeroOutputs(const OpIds &opIds) const {
  uint64_t N{0};
  for (auto i : opIds) {
    if (op(i).nOutTensors() == 0) {
      ++N;
    }
  }
  return N;
}

std::vector<util::StringColumn> Graph::getMultioutColumns() const {
  return getMultioutColumns(opIds());
}

std::vector<util::StringColumn>
Graph::getMultioutColumns(const OpIds &opIds) const {
  const auto nTens = nMultioutRows(opIds);

  using Strings = std::vector<std::string>;
  Strings opId(nTens, "");
  Strings name(nTens, "");
  Strings opType(nTens, "");
  Strings inTensors(nTens, "");
  Strings outIndex(nTens, "");
  Strings tensorShape(nTens, "");

  uint64_t ti = 0;
  for (auto i : opIds) {
    opId[ti]      = std::to_string(i.get());
    opType[ti]    = op(i).typeString();
    name[ti]      = op(i).getName();
    inTensors[ti] = util::getStr((op(i).inTensorIds()));
    for (uint64_t o = 0; o < op(i).nOutTensors(); ++o) {
      outIndex[ti]    = std::to_string(o);
      tensorShape[ti] = util::getStr(shape({i, o}).get());
      ++ti;
    }
    if (op(i).nOutTensors() == 0) {
      ++ti;
    }
  }

  std::vector<util::StringColumn> cols;
  cols.push_back({"OpId", opId});

  // only add debug strings if at least one has:
  if (std::any_of(name.cbegin(), name.cend(), [](const auto &x) {
        return !x.empty();
      })) {
    cols.push_back({"Name", name});
  }

  cols.push_back({"OpType", opType});
  cols.push_back({"InTensors", inTensors});

  if (std::any_of(outIndex.cbegin(), outIndex.cend(), [](const auto &x) {
        return (!x.empty() && x[0] != '0');
      })) {
    cols.push_back({"OutIndex", outIndex});
  }

  cols.push_back({"Shape", tensorShape});

  return cols;
}

} // namespace multiout
} // namespace common

namespace util {
template class CopyByClone<common::multiout::Op>;
}

} // namespace poprithms
