// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_DISTRIBUTED_PROGRAM_HPP
#define POPRITHMS_PROGRAM_DISTRIBUTED_PROGRAM_HPP

#include <map>
#include <ostream>
#include <vector>

#include <poprithms/program/distributed/helper.hpp>

namespace poprithms {
namespace program {
namespace distributed {

/**
 * A sequence of ops with the same CodeLocation (either Ipu or Host).
 *
 * This class corresponds more closely to poplar::program::Sequence than
 * to poplar::program::Program.
 *
 * See the distributed::Helper class for more information on the project, in
 * general.
 * */
class Program {

public:
  Program() = delete;
  Program(CodeLocation);

  /**
   * Append #opId to the sequence of ops.
   * */
  void appendOp(OpId opId) { opIds_.push_back(opId); }

  const OpIds &opIds() const { return opIds_; }
  uint64_t nOps() const { return opIds_.size(); }

  bool isHost() const { return cl_ == CodeLocation::Host; }
  bool isIpu() const { return cl_ == CodeLocation::Ipu; }
  bool isCodeless() const { return cl_ == CodeLocation::None; }

  /**
   * Set a call id. This is only valid if this Program has CodeLocation Ipu.
   * If this Program has another CodeLocation, an error is thrown.
   * */
  void setIpuCallId(uint32_t);

  uint64_t ipuCallId() const { return ipuCallId_; }

  bool hasIpuCallId() const { return hasIpuCallId_; }

  void append(std::ostream &) const;

  bool operator==(const Program &rhs) const { return tup() == rhs.tup(); }

private:
  std::tuple<CodeLocation, OpIds, bool, uint32_t> tup() const {
    return {cl_, opIds_, hasIpuCallId_, ipuCallId_};
  }
  CodeLocation cl_;
  OpIds opIds_;
  bool hasIpuCallId_{false};
  uint32_t ipuCallId_;
};
std::ostream &operator<<(std::ostream &, const Program &);

/**
 * The index of a Program within a Sequence
 * */
using ProgramIndex = poprithms::util::TypedInteger<'y', uint32_t>;

/**
 * A sequence of Programs, with CodeLocations alternating between Host and
 * Ipu.
 * */
class Sequence {
public:
  /**
   * Create a new Program in the sequence, initialized with the single op
   * #opId.
   * */
  void appendToNew(CodeLocation, OpId opId);

  /**
   * Append the op #opId to the Program at the back of the sequence (the most
   * recently added sequence).
   * */
  void appendToBack(OpId);

  /**
   * return true if this sequence is empty.
   * */
  bool empty() const { return programs_.empty(); }

  uint64_t nPrograms() const { return programs_.size(); }

  const Program &back() const { return programs_.back(); }

  /**
   * return the i'th Program that was added to this sequence.
   * */
  const Program &at(ProgramIndex i) const;

  /**
   * Set the call id of the #i'th program to #callId.
   * */
  void setIpuCallId(ProgramIndex i, uint32_t callId);

  /**
   * return the index of the op #opId within its Program. The Op must belong
   * to exactly 1 Program in this Sequence.
   * */
  ProgramIndex programIndex(OpId opId) const;

  void append(std::ostream &) const;

  bool operator==(const Sequence &rhs) const;

  const std::vector<Program> &programs() const { return programs_; }

private:
  // For each Op, the index within its Program that it appears.
  std::map<OpId, ProgramIndex> programIndices_;

  // Programs, in sequence.
  std::vector<Program> programs_;
};
std::ostream &operator<<(std::ostream &, const Sequence &);

/**
 * Multiple sequences.
 *
 * This class manages the decomposition of ops and programs between host and
 * ipu. It determines which ipu programs need to be dynamically executable
 * from host.
 *
 * It is constructed from a graph of ops, which is modelled by the absract
 * Helper class.
 * */
class Sequences {

public:
  Sequences(const Helper &);

  /**
   * return the sequence of programs of #sgId.
   * */
  Sequence &at(SubGraphId sgId) { return sequences_.at(sgId); }

  void append(std::ostream &) const;

  // The ipu programs which must be made executable in the poplar::Engine
  // constructor.
  using EngProgs = std::vector<std::pair<SubGraphId, ProgramIndex>>;
  const EngProgs &enginePrograms() const { return enginePrograms_; }

  bool operator==(const Sequences &rhs) const;

  static void append(std::ostream &, const EngProgs &);

  const std::map<SubGraphId, Sequence> &sequences() const {
    return sequences_;
  }

private:
  Sequences() = delete;
  EngProgs enginePrograms_;
  std::map<SubGraphId, Sequence> sequences_;
};

std::ostream &operator<<(std::ostream &, const Sequences &);

} // namespace distributed
} // namespace program
} // namespace poprithms

#endif
