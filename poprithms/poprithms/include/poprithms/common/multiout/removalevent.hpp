// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_MULTIOUT_REMOVALEVENT_HPP
#define POPRITHMS_COMMON_MULTIOUT_REMOVALEVENT_HPP

#include <ostream>
#include <string>

#include <poprithms/common/multiout/opid.hpp>

namespace poprithms {
namespace common {
namespace multiout {

/**
 * Record of an Op being removed from a Graph.
 * */
struct RemovalEvent {

  RemovalEvent() = delete;

  /**
   * \param opRemoved the op that was removed.
   *
   * \param nameOfOpRemoved the name of the op that was removed (optional).
   *
   * \param nOpsCreatedAtTimeOfRemoval the total number of ops which had
   *                                   been created when the op was removed.
   *                                   This is useful for ordering
   *                                   insertion/removal events of ops.
   *
   * \param contextOfRemoval the transformation, or another context, in which
   *                         the op was removed (optional).
   * */
  RemovalEvent(OpId opRemoved,
               const std::string &nameOfOpRemoved,
               uint64_t nOpsCreatedAtTimeOfRemoval,
               const std::string &contextOfRemoval);

  bool operator==(const RemovalEvent &) const;
  bool operator!=(const RemovalEvent &r) const { return !operator==(r); }

  // as per constructor args:
  OpId opId;
  std::string name;
  uint64_t totalOpsCreatedSoFar;
  std::string context;

  // string summary of the RemovalEvent:
  void append(std::ostream &) const;
  std::string str() const;
};

struct RemovalEvents {

  RemovalEvents() = default;
  RemovalEvents(std::vector<RemovalEvent> &&es) : events(std::move(es)) {}

  bool operator==(const RemovalEvents &rhs) const {
    return events == rhs.events;
  }
  bool operator!=(const RemovalEvents &rhs) const { return !operator==(rhs); }
  size_t size() const { return events.size(); }

  /**
   * Access a RemovalEvent. An error is thrown if there is no RemovalEvent
   * for #opId.
   * */
  RemovalEvent event(OpId opId) const;

  /**
   * \return true iff there is a removal event for #opId.
   * */
  bool registered(OpId opId) const;

  /**
   * Registed a removal event.
   * */
  void insert(const RemovalEvent &e) { events.push_back(e); }

  /** String summary of the RemovalEvents */
  void append(std::ostream &) const;
  std::string str() const;

  std::vector<RemovalEvent> events;
};

std::ostream &operator<<(std::ostream &, const RemovalEvent &);
std::ostream &operator<<(std::ostream &, const RemovalEvents &);

} // namespace multiout
} // namespace common
} // namespace poprithms

#endif
