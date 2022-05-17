// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_CALLSTACK_COPYOUT_HPP
#define POPRITHMS_PROGRAM_CALLSTACK_COPYOUT_HPP

#include <map>
#include <vector>

#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/program/callstack/calleeindex.hpp>

namespace poprithms {
namespace program {
namespace callstack {

using poprithms::common::multiout::ContiguousOutIndexSubset;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OptionalTensorId;
using poprithms::common::multiout::OptionalTensorIds;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::OutIndices;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;

/**
 * Encapsulation of the outputs of an op with callees. Description of what
 * we're modelling:
 *
 * For each OutIndex #o and for each CalleeIndex #ci, there is one optional
 * TensorId, which is the tensor in the sub-graph (callee) #ci. It is copied
 * out of the sub-graph #ci, if that path is taken.
 *
 * For example, suppose the op corresponds to a switch statement:
 *
 *  case 0: run callee x. return a, b (tensors in x)
 *  case 1: run callee y. return c, d (tensors in y)
 *  case 2: run callee z. return e, f (tensors in z).
 *
 * where x, y and z are the (not necessarily different) sub-graphs at
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
   * Create a CopyOuts object from optional TensorIds. #ts.size() is the
   * number of output indices, and ts[o][c] is the optional output of callee
   * graph #c at output index #o.
   * */
  static CopyOuts fromOptionals(const std::vector<OptionalTensorIds> &ts);

private:
  CopyOuts(const std::vector<OptionalTensorIds> &, bool checkRectangle);

public:
  /**
   * The #o'th output of the #c'th callee.
   * */
  TensorId outSource(OutIndex o, CalleeIndex c) const;

  /**
   * The #o'th outputs of all callees.
   * */
  TensorIds outSources(OutIndex o) const;

  /**
   * \return true if the callee #c at output index #o is set.
   * */
  bool hasValue(OutIndex o, CalleeIndex ci) const;

  /**
   * All outputs of the #c'th callee.
   * */
  TensorIds outSources(CalleeIndex c) const;

  /**
   * The outputs of the #c'th callee, at indices #outIndces.
   * */
  TensorIds outSources(CalleeIndex c, const OutIndices &outIndices) const;

  /**
   * The number of callees. Note that an error is thrown if there are no
   * outputs.
   * */
  uint64_t nCallees() const;

  /**
   * The output index of tensor #tId in the callee graph #ci.
   * */
  OutIndex outIndex(CalleeIndex ci, const TensorId &tId) const;

  /**
   * \return true if #tId is a copy source in callee graph #ci.
   * */

  bool isSource(CalleeIndex ci, const TensorId &tId) const;

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

  /**
   * Set the output of callee #ci at output index #o to #tId.
   * */
  void reset(OutIndex o, CalleeIndex ci, const TensorId &tId);

private:
  // indexed as outs[outIndex][calleeIndex].
  std::vector<OptionalTensorIds> outs;

  void verifyValidOutIndex(OutIndex) const;
  void assertValidCalleeIndex(CalleeIndex) const;
};

std::ostream &operator<<(std::ostream &, const CopyOuts &);

} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
