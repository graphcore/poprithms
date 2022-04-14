// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_CALLSTACK_COPYIN_HPP
#define POPRITHMS_PROGRAM_CALLSTACK_COPYIN_HPP

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/program/callstack/calleeindex.hpp>
#include <poprithms/program/callstack/calleetensorid.hpp>

namespace poprithms {
namespace program {
namespace callstack {

using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::InIndices;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;

/**
 * A helper class to connect a tensor across the scope of a calling op's
 * sub-graph and one of its callee sub-graphs.
 * */
class CopyIn {
public:
  /**
   * \param s The source of the copy into the callee
   *
   * \param d The destination of the copy, a tensor in the callee graph
   *
   * \param i The index within the calling op of the callee graph. For ops
   *          with just 1 callee graph, such as call ops and loop ops, this
   *          will always be 0.
   * */
  CopyIn(const TensorId &s, const TensorId &d, CalleeIndex i)
      : src_(s), dst_(d), index_(i) {}

  /**
   * Some additional constructors, useful for classes with a method #id which
   * returns a TensorId.
   * */
  template <typename T>
  CopyIn(const T &src, const T &dst, CalleeIndex i)
      : CopyIn(src.id(), dst.id(), i) {}

  template <typename T>
  CopyIn(const TensorId &src, const T &dst, CalleeIndex i)
      : CopyIn(src, dst.id(), i) {}

  template <typename T>
  CopyIn(const T &src, const TensorId &dst, CalleeIndex i)
      : CopyIn(src.id(), dst, i) {}

  /**
   * The source of the copy into the callee graph.
   **/
  TensorId src() const { return src_; }

  /**
   * The destination of the copy in the callee graph.
   * */
  TensorId dst() const { return dst_; }

  /**
   * The index of the callee graph in the calling op.
   * */
  CalleeIndex index() const { return index_; }

  uint32_t index_u32() const { return index_.get(); }

  void append(std::ostream &) const;

  bool operator==(const CopyIn &r) const { return t() == r.t(); }
  bool operator<(const CopyIn &r) const { return t() < r.t(); }
  bool operator!=(const CopyIn &r) const { return t() != r.t(); }
  bool operator<=(const CopyIn &r) const { return t() <= r.t(); }
  bool operator>(const CopyIn &r) const { return t() > r.t(); }
  bool operator>=(const CopyIn &r) const { return t() >= r.t(); }

private:
  std::tuple<TensorId, TensorId, CalleeIndex> t() const {
    return {src_, dst_, index_};
  }

  TensorId src_;
  TensorId dst_;
  CalleeIndex index_;
};

/**
 * A container of objects of type CopyIn. There is no constraint on the number
 * of CopyIn objects per CalleeIndex, and tensors in the calling scope can be
 * copied to multiple tensors in the callee scopes.
 * */
class CopyIns {

public:
  CopyIns() = default;
  CopyIns(const std::vector<CopyIn> &cis);

  /**
   * Construct a vector of CopyIns with sources #srcs (in calling sub-graph),
   * and destinations #dsts, in the callee sub-graph at index #i.
   * */
  static std::vector<CopyIn>
  zip(const TensorIds &srcs, const TensorIds &dsts, CalleeIndex i);

  /**
   * Construct a vector of CopyIns with sources #srcs (in calling sub-graph),
   * and destinations #dsts, where #dsts are in the sub-graphs at indices #is.
   * */
  static std::vector<CopyIn>
  zip(const TensorIds &srcs, const TensorIds &dsts, const CalleeIndices &cis);

  /**
   * Construct a vector of CopyIns with sources #srcs (in calling sub-graph),
   * and destinations #dsts. Each CalleeTensor in #dsts consists of a callee
   * sub-graph index, and an tensor in that sub-graph.
   * */
  static std::vector<CopyIn> zip(const TensorIds &srcs,
                                 const CalleeTensorIds &dsts);

  const std::vector<CopyIn> &copyIns() const { return copyIns_; }

  CalleeIndex calleeIndex(InIndex i) const {
    return copyIns_.at(i.get()).index();
  }

  bool empty() const { return copyIns_.empty(); }

  void append(std::ostream &) const;

  std::string str() const;

  bool operator==(const CopyIns &cIns) const {
    return copyIns_ == cIns.copyIns();
  }

  bool operator<(const CopyIns &cIns) const {
    return copyIns_ < cIns.copyIns();
  }

  bool operator>(const CopyIns &cIns) const {
    return copyIns_ > cIns.copyIns();
  }

  bool operator!=(const CopyIns &r) const { return !operator==(r); }
  bool operator>=(const CopyIns &r) const { return !operator<(r); }
  bool operator<=(const CopyIns &r) const { return !operator>(r); }

  /**
   * The sources of all of the copies.
   * */
  TensorIds srcIds() const;

  /**
   * The destinations of all of the copies.
   * */
  TensorIds dstIds() const;

  /**
   * \return true if the tensor #tId is a copy destination in the callee graph
   *         #ci.
   * */
  bool isDst(CalleeIndex ci, const TensorId &tId) const;

  /**
   * \return The copy destinations if inputs at indices #inIndices.
   * */
  CalleeTensorIds indexedDsts(const InIndices &inIndices) const;

  /**
   * \return the copy destinations for inputs at indices #inIndices.
   * */
  TensorIds dsts(const InIndices &inIndices) const;

  /**
   * \return The source of the copy to #tId in the callee graph #ci.
   *
   * \throw error if #tId is not a copy destination in callee graph #ci.
   * */

  TensorId src(CalleeIndex ci, const TensorId &tId) const;

  TensorId dst(InIndex i) const { return copyIns_.at(i.get()).dst(); }
  TensorId src(InIndex i) const { return copyIns_.at(i.get()).src(); }

  TensorIds srcs(CalleeIndex) const;
  TensorIds dsts(CalleeIndex) const;

  uint64_t nInTensors() const { return copyIns_.size(); }

  /**
   * \return the destination of the copy from #inCaller into sub-graph #ci.
   *
   * \throw error if #inCaller is not copied into #ci.
   * */
  TensorIds dsts(CalleeIndex, const TensorId &inCaller) const;

  /**
   * \return the input index at which the tensor #inCaller is copied into the
   *         callee #ci.
   *
   * If #inCaller is not copied into #ci, then an error is thrown.
   * */
  InIndices indicesOfSrc(CalleeIndex ci, const TensorId &inCaller) const;

  InIndex inIndex(CalleeIndex ci, const TensorId &inCallee) const;

private:
  // For each input index, the source and destination of the input copy.
  std::vector<CopyIn> copyIns_;

  /**
   * return true if, at every callee index, the destinations are all unique.
   * That is, there is no tensor in the callee graph which gets copied 2 from
   * multiple sources.
   * */
  bool destinationsUniqueAtAllIndices() const;
};

std::ostream &operator<<(std::ostream &, const CopyIn &);
std::ostream &operator<<(std::ostream &, const CopyIns &);

} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
