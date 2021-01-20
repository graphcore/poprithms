// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_RESULT_HPP
#define POPRITHMS_MEMORY_INPLACE_RESULT_HPP
#include <algorithm>

#include <poprithms/memory/inplace/constraint.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

enum OpeningStatus {
  Valid = 0,   ///< Opening the Mux is valid
  Cycle,       ///< Opening the Mux is invalid, as the additional
               ///< constraints required will result in a cycle
  AlreadyOpen, ///< Opening is invalid, because the Mux is already open
  NotParallelWriteable, ///< The opening is invalid, as it results in writing
                        ///< a Tensor which is not parallel writeable.
};

/** A summary of the result of attempting to open a Mux.
 *
 * This class consists of
 * 1) OpeningStatus (see above), and
 * 2) The constraints required if (1) is OpeningStatus::Valid, and
 * 3) The new schedule if one is required by the constraints in (2).
 * */
class OpeningResult {
public:
  static OpeningResult validWithUnchangedSchedule(Constraints &&);
  static OpeningResult validWithChangedSchedule(Constraints &&, OpIds &&);
  static OpeningResult cycle();
  static OpeningResult alreadyOpen();
  static OpeningResult notParallelWriteable();

  OpeningStatus status() const { return status_; }
  bool isValid() const { return status() == OpeningStatus::Valid; }

  const Constraints &constraints() const;

  const OpIds &schedule() const;

  bool scheduleChange() const { return scheduleChange_; }

  void append(std::ostream &ost) const;

private:
  OpeningStatus status_;
  Constraints constraints_;
  OpIds schedule_;
  bool scheduleChange_;

  /** Constructor for a successful opening. */
  OpeningResult(OpeningStatus st, Constraints &&cs, OpIds &&sc, bool hs)
      : status_(st), constraints_(std::move(cs)), schedule_(std::move(sc)),
        scheduleChange_(hs) {}
};

using OpeningResults  = std::vector<OpeningResult>;
using OpeningStatuses = std::vector<OpeningStatus>;

std::ostream &operator<<(std::ostream &, const OpeningResult &);
std::ostream &operator<<(std::ostream &, const OpeningResults &);
std::ostream &operator<<(std::ostream &, const OpeningStatus &);
std::ostream &operator<<(std::ostream &, const OpeningStatuses &);
std::ostream &operator<<(std::ostream &, const Constraints &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
