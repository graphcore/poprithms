// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_CALLSTACK_COPYMAP_HPP
#define POPRITHMS_PROGRAM_CALLSTACK_COPYMAP_HPP

#include <ostream>
#include <unordered_map>
#include <vector>

#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/opid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/program/callstack/calleeindex.hpp>
#include <poprithms/program/callstack/callstack.hpp>

namespace poprithms {
namespace program {
namespace callstack {

using poprithms::common::multiout::InIndex;
using poprithms::common::multiout::InIndices;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::OutIndices;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;

/**
 * For a tensor with id #tId, what are all the CallEvents where #tId is in a
 * callee and is copied into?
 *
 * This class helps answer this question. Essentially it reverses a mapping
 * from "call -> copy in destination " which is passed in the constructor (in
 * the form of a graph), to a mapping "tensor : all calls which copy to the
 * tensor ".
 * */
class CopyInMap {

public:
  CopyInMap() = default;
  /**
   * Construct the mapping m[tId] = {all CallEvents which copy into tId},
   * starting from a TGraph (template class) which has methods
   *
   *   (1) callees(OpId opId)  <-- sub-graphs of the #opId.
   *   (2) opIds()             <-- supserset of all ops with callees.
   *   (3) inCopies(OpId)      <-- all the copies into callees of op #opId.
   *
   * */

  template <class TGraph> CopyInMap(const TGraph &g) {

    // For all ops will callees and for all copies in, make a single entry
    // into the map for the destination of the copy:
    for (const auto &opId : g.opIds()) {
      const auto callees = g.callees(opId);
      if (!callees.empty()) {
        auto cis = g.inCopies(opId).copyIns();
        for (InIndex i = 0; i < cis.size(); ++i) {
          const auto ci = cis[i.get()];
          auto found    = m_.find(ci.src());
          auto sg       = callees.at(ci.index().get());

          // The calling op #opId calls into the sub-graph sg, which is its
          // ci'th subgraph:
          CallEvent e(opId, sg, ci.index());
          std::pair<CallEvent, InIndex> p(e, i);
          if (found == m_.cend()) {
            m_.insert({ci.src(), {p}});
          } else {
            found->second.push_back(p);
          }
        }
      }
    }
  }

  const std::vector<std::pair<CallEvent, InIndex>> &
  get(const TensorId &tId) const;

  uint64_t n(const TensorId &tId) const { return get(tId).size(); }

private:
  std::unordered_map<TensorId, std::vector<std::pair<CallEvent, InIndex>>> m_;
  std::vector<std::pair<CallEvent, InIndex>> empty_{};
};

/**
 * A mapping from tensors in callee graphs to all their copy-out destination.
 * Analagous to the CopyInMap class, but for tensors being copied out of
 * callee sub-graphs.  */
class CopyOutMap {

public:
  CopyOutMap() = default;

  template <class TGraph> CopyOutMap(const TGraph &g) {
    for (const auto &opId : g.opIds()) {
      const auto callees = g.callees(opId);
      if (!callees.empty()) {
        auto cot = g.outCopies(opId);
        for (CalleeIndex c = 0; c < callees.size(); ++c) {
          for (OutIndex o = 0; o < cot.nOutTensors(); ++o) {
            auto src = cot.outSource(o, c);
            CallEvent ce(opId, callees.at(c.get()), c);
            std::pair<CallEvent, OutIndex> p(ce, o);
            auto found = m_.find(src);
            if (found == m_.cend()) {
              m_.insert({src, {p}});
            } else {
              found->second.push_back(p);
            }
          }
        }
      }
    }
  }

  /**
   * Return all of the call events where tensor #tId is copied out of a
   * callee.
   * */
  const std::vector<std::pair<CallEvent, OutIndex>> &
  get(const TensorId &tId) const;

  uint64_t n(const TensorId &tId) const { return get(tId).size(); }

private:
  std::unordered_map<TensorId, std::vector<std::pair<CallEvent, OutIndex>>>
      m_;

  // #get returns by cref, this is used if there is no match of key in m_.
  std::vector<std::pair<CallEvent, OutIndex>> empty_{};
};
} // namespace callstack
} // namespace program
} // namespace poprithms

#endif
