// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_CALLSTACK_COPYIN_HPP
#define POPRITHMS_PROGRAM_CALLSTACK_COPYIN_HPP

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/program/callstack/calleeindex.hpp>

namespace poprithms {
namespace program {
namespace callstack {

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
 * of CopyIn objects per CalleeIndex.
 * */
class CopyIns {

public:
  CopyIns() = default;

  CopyIns(const std::vector<CopyIn> &cis);

  /**
   * Construct a CopyIns object with sources #srcs (in calling sub-graph), and
   * destinations #dsts, in the callee sub-graph at index #i.
   * */
  static CopyIns
  zip(const TensorIds &srcs, const TensorIds &dsts, CalleeIndex i);

  /**
   * Construct a CopyIns object with sources #srcs (in calling sub-graph), and
   * destinations #dsts, where #dsts are in the sub-graphs at indices #is.
   * */
  static CopyIns
  zip(const TensorIds &srcs, const TensorIds &dsts, const CalleeIndices &is);

  const std::vector<CopyIn> &copyIns() const { return copyIns_; }

  /**
   * The sources of all of the CopyIns.
   * */
  TensorIds srcIds() const;

  /**
   * The destinations of all of the CopyIns.
   * */
  TensorIds dstIds() const;

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
   * \return true if the tensor #tId is a copy destination in the callee graph
   *         #ci.
   * */
  bool isDst(CalleeIndex ci, const TensorId &tId) const;

  /**
   * \return The source of the copy to #tId in the callee graph #ci.
   *
   * If #tId is not a copy destination in callee graph #ci, an error is
   * thrown.
   * */

  TensorId src(CalleeIndex ci, const TensorId &tId) const;

  /**
   * return true if, at every callee index, the destinations are all unique.
   * That is, there is no tensor in the callee graph which gets copied 2 from
   * multiple sources.
   * */
  bool destinationsUniqueAtAllIndices() const;
  void assertDestinationsUniqueAtAllIndices() const;

  /**
   * return true if, at every callee index, the sources are all unique. That
   * is, there is no tensor in the calling graph which gets copied more than
   * once in the callee graph. This is not a strict requirement in general.
   * */
  bool sourcesUniqueAtAllIndices() const;

private:
  std::vector<CopyIn> copyIns_;
};

std::ostream &operator<<(std::ostream &, const CopyIn &);
std::ostream &operator<<(std::ostream &, const CopyIns &);


} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
