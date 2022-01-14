// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "error.hpp"

#include <limits>
#include <set>
#include <sstream>

#include <poprithms/program/distributed/program.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace program {
namespace distributed {

using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;

Program::Program(CodeLocation cl)
    : cl_(cl), ipuCallId_(std::numeric_limits<uint32_t>::max()) {
  if (cl == CodeLocation::None) {
    std::ostringstream oss;
    oss << "Failed in Program(CodeLocation = " << cl
        << "). CodeLocation must be Host or Ipu.";
    throw error(oss.str());
  }
}

const Program &Sequence::at(ProgramIndex i) const {
  return programs_.at(i.get());
}

void Program::setIpuCallId(uint32_t c) {
  if (!isIpu()) {
    std::ostringstream oss;
    oss << "Invalid call to setIpuCallId(c = " << c
        << ") for Program which does not have CodeLocation::Ipu.";
    append(oss);
    throw error(oss.str());
  }
  ipuCallId_    = c;
  hasIpuCallId_ = true;
}

void Sequence::appendToNew(CodeLocation cl, OpId opId) {
  programs_.push_back(Program(cl));
  appendToBack(opId);
}

void Sequence::appendToBack(OpId opId) {
  auto found = programIndices_.find(opId);
  if (found != programIndices_.cend()) {
    std::ostringstream oss;
    oss << "The op " << opId
        << " already appears in a Program in this Sequence, "
        << "it has ProgramIndex " << found->second
        << ". Ops must appear at most once in a Sequence. ";
    throw error(oss.str());
  }
  programIndices_.emplace(opId, programs_.size() - 1);
  programs_.back().appendOp(opId);
}

void Sequence::setIpuCallId(ProgramIndex i, uint32_t callId) {
  programs_.at(i.get()).setIpuCallId(callId);
}

void Sequence::append(std::ostream &ost) const {
  poprithms::util::append(ost, programs_);
}

ProgramIndex Sequence::programIndex(OpId opId) const {
  auto found = programIndices_.find(opId);
  if (found == programIndices_.cend()) {
    std::ostringstream oss;
    oss << "Failed to find a Program with the Op " << opId
        << " in it, in this Sequence of Programs. "
        << "Failed to retrieve ProgramIndex.";
    throw error(oss.str());
  }
  return found->second;
}

Sequences::Sequences(const Helper &helper) {

  const SubGraphIds callable  = helper.userCallable();
  const SubGraphIds reachable = helper.userReachable();

  // map from reachable sub-graphs, to all ops which call them.
  std::map<SubGraphId, OpIds> callers;
  for (auto g : reachable) {
    for (auto op : helper.schedule(g)) {
      for (auto c : helper.callees(op)) {
        auto found = callers.find(c);
        if (found == callers.cend()) {
          callers.insert({c, {op}});
        } else {
          found->second.push_back(op);
        }
      }
    }
  }

  // initialize sequences_.
  for (SubGraphId sgId : reachable) {
    sequences_.emplace(sgId, Sequence());
    auto &programs = sequences_[sgId];
    for (auto opId : helper.schedule(sgId)) {
      const auto location = helper.codeLocation(opId);
      const bool isHost   = location == CodeLocation::Host;
      if (location != CodeLocation::None) {
        if (programs.empty() || isHost != programs.back().isHost()) {
          programs.appendToNew(
              isHost ? CodeLocation::Host : CodeLocation::Ipu, opId);
        } else {
          programs.appendToBack(opId);
        }
      }
    }
  }

  std::set<std::pair<SubGraphId, ProgramIndex>> engineProgramSet;

  // If a graph (sequence) is either:
  //   1) user callable, or
  //   2) the callee of a host op,
  // then all of it's ipu programs must be engine programs.
  //
  // 1) user callable
  for (auto sgId : callable) {
    auto &sequence_ = at(sgId);
    for (ProgramIndex i = 0; i < sequence_.nPrograms(); ++i) {
      if (!sequence_.at(i).isHost()) {
        engineProgramSet.insert({sgId, i});
      }
    }
  }

  // 2) callees of host op
  // sgId is called by ops:
  for (const auto &[callee, calledBy] : callers) {
    for (auto op : calledBy) {
      if (helper.codeLocation(op) == CodeLocation::Host) {
        for (ProgramIndex i = 0; i < at(callee).nPrograms(); ++i) {
          if (at(callee).at(i).isIpu()) {
            engineProgramSet.insert({callee, i});
          }
        }
      }
    }
  }

  // call ids are in increasing order.
  enginePrograms_ =
      EngProgs{engineProgramSet.cbegin(), engineProgramSet.cend()};
  for (uint64_t i = 0; i < enginePrograms_.size(); ++i) {
    auto local = enginePrograms_[i];
    at(local.first).setIpuCallId(local.second, i);
  }
}

std::ostream &operator<<(std::ostream &ost, const Sequence &gps) {
  gps.append(ost);
  return ost;
}

void Sequences::append(std::ostream &ost) const {
  for (const auto &[sgid, gp] : sequences_) {
    ost << "\n     SubGraphId=" << sgid.get_u64() << ": " << gp;
  }
  ost << "\n     Engine Programs:";
  append(ost, enginePrograms_);
}

void Sequences::append(std::ostream &ost, const EngProgs &eps) {
  std::vector<std::string> strs;
  strs.reserve(eps.size());
  for (const auto &x : eps) {
    std::ostringstream oss;
    oss << '(' << x.first.get_u64() << ',' << x.second.get() << ')';
    strs.push_back(oss.str());
  }
  poprithms::util::append(ost, strs);
}

std::ostream &operator<<(std::ostream &ost, const Sequences &gps) {
  gps.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const Program &gp) {
  gp.append(ost);
  return ost;
}

void Program::append(std::ostream &ost) const {
  ost << "(" << cl_ << ",ipuCallId="
      << (hasIpuCallId_ ? std::to_string(ipuCallId_) : "none") << ",ops=";
  poprithms::util::append(ost, opIds_);
  ost << ")";
}

bool Sequence::operator==(const Sequence &rhs) const {
  return programIndices_ == rhs.programIndices_ && programs_ == rhs.programs_;
}

bool Sequences::operator==(const Sequences &rhs) const {

  // safe to compare, as guaranteed to be sorted.
  return enginePrograms() == rhs.enginePrograms() &&
         sequences_ == rhs.sequences_;
}

} // namespace distributed
} // namespace program
} // namespace poprithms
