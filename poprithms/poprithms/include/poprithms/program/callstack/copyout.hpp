// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_CALLSTACK_COPYOUT_HPP
#define POPRITHMS_PROGRAM_CALLSTACK_COPYOUT_HPP

#include <map>
#include <vector>

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/program/callstack/calleeindex.hpp>

namespace poprithms {
namespace program {
namespace callstack {

using poprithms::common::multiout::ContiguousOutIndexSubset;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::OutIndices;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;

/**
 * Encapsulation of the outputs of an op with callees. Description of what
 * we're modelling:
 *
 * For each OutIndex #o and for each CalleeIndex #ci, there is one TensorId,
 * which is the tensor in the sub-graph (callee) #ci. It is copied out of
 * the sub-graph #ci, if that path is taken.
 *
 * For example, suppose the op corresponds to a switch statement:
 *
 *  case 0: run callee x. return a, b (tensors in x)
 *  case 1: run callee y. return c, d (tensors in y)
 *  case 2: run callee z. return e, f (tensors in z).
 *
 * where x, y and z are the (not necessarily different) sub-graphs as
 * CalleeIndices 0, 1 and 2. A CopyOuts object describing this switch
 * statements will return the following values for certain method calls
 * (decribed further in the class definition).
 *
 *  outSource(OutIndex(1), CalleeIndex(2)) -> f.
 *  nCallees -> 3
 *  nOuTensors -> 2
 *  outSources(OutIndex(0)) -> (a,c,e)
 *  outSources(CalleeIndex(1)) -> (c,d).
 * */
class CopyOuts {
public:
  CopyOuts() = default;

  /**
   * Construct from a vector of vectors, indexed as [outIndex][calleeIndex].
   * That is, ts.size() is the number of outputs and ts[0].size() is the
   *number of callees.
   **/
  CopyOuts(const std::vector<TensorIds> &ts);

  /**
   * Construct from a map #m, where m[calleeIndex][outIndex] is the
   * #outIndex'th output of the #calleeIndex'th callee sub-graph.
   * */
  explicit CopyOuts(const std::map<CalleeIndex, TensorIds> &m);

  /**
   * The #o'th output of the #c'th callee.
   * */
  TensorId outSource(OutIndex o, CalleeIndex c) const;

  /**
   * The #o'th output of the #c'th callee.
   * */
  TensorId src(CalleeIndex c, OutIndex o) const { return outSource(o, c); }

  /**
   * The #o'th outputs of all callees.
   * */
  TensorIds outSources(OutIndex o) const { return outs.at(o.get()); }

  /**
   * All outputs of the #c'th callee.
   * */
  TensorIds outSources(CalleeIndex c) const;

  /**
   * The number of callees. Note that an error is thrown if there are no
   * outputs.
   * */
  uint64_t nCallees() const;

  uint64_t nOutTensors() const { return outs.size(); }

  bool operator==(const CopyOuts &rhs) const { return outs == rhs.outs; }
  bool operator<(const CopyOuts &rhs) const { return outs < rhs.outs; }
  bool operator>(const CopyOuts &rhs) const { return outs > rhs.outs; }
  bool operator!=(const CopyOuts &rhs) const { return !operator==(rhs); }
  bool operator<=(const CopyOuts &rhs) const { return !operator>(rhs); }
  bool operator>=(const CopyOuts &rhs) const { return !operator<(rhs); }

  std::string outSourcesString(OutIndex o) const;

  void append(std::ostream &) const;

  void reduce(const ContiguousOutIndexSubset &coin) { coin.reduce(outs); }

private:
  // indexed as outs[outIndex][calleeIndex].
  std::vector<TensorIds> outs;
};

std::ostream &operator<<(std::ostream &, const CopyOuts &);

} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
