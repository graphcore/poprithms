// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_INPLACE_RESULT_HPP
#define POPRITHMS_MEMORY_INPLACE_RESULT_HPP
#include <algorithm>

#include <poprithms/memory/inplace/usings.hpp>

namespace poprithms {
namespace memory {
namespace inplace {

enum InplaceStatus {
  Valid = 0, ///< The inplace transformation is valid
  Cycle,     ///< The inplace transformation is not valid, as the additional
             ///< constraints result in a cycle
  AlreadyInplace, ///< The inplace transformation is not valid, because the
                  ///< target is already inplace
  NotParallelWriteable, ///< The inplace transformation is not valid, as
                        ///< it results in writing a Tensor which is not
                        ///< parallel writeable.
};

/** A summary of the result of attempting to inplace an Op. Consists of
 * 1) The InplaceStatus (see above), and
 * 2) The constraints required if (1) is InplaceStatus::Valid.
 * 3) The new schedule if one is required by the constraints in (2).
 * */
class InplaceResult {
public:
  static InplaceResult validWithUnchangedSchedule(Constraints &&);
  static InplaceResult validWithChangedSchedule(Constraints &&, OpIds &&);
  static InplaceResult cycle();
  static InplaceResult alreadyInplace();
  static InplaceResult notParallelWriteable();

  InplaceStatus status() const { return status_; }
  bool isValid() const { return status() == InplaceStatus::Valid; }

  const Constraints &constraints() const;

  const OpIds &schedule() const;

  bool scheduleChange() const { return scheduleChange_; }

  void append(std::ostream &ost) const;

private:
  InplaceStatus status_;
  Constraints constraints_;
  OpIds schedule_;
  bool scheduleChange_;

  /** Constructor for a successful inplacing attempt. */
  InplaceResult(InplaceStatus st, Constraints &&cs, OpIds &&sc, bool hs)
      : status_(st), constraints_(std::move(cs)), schedule_(std::move(sc)),
        scheduleChange_(hs) {}
};

using InplaceResults  = std::vector<InplaceResult>;
using InplaceStatuses = std::vector<InplaceStatus>;

std::ostream &operator<<(std::ostream &, const InplaceResult &);
std::ostream &operator<<(std::ostream &, const InplaceResults &);
std::ostream &operator<<(std::ostream &, const InplaceStatus &);
std::ostream &operator<<(std::ostream &, const InplaceStatuses &);
std::ostream &operator<<(std::ostream &, const Constraints &);

} // namespace inplace
} // namespace memory
} // namespace poprithms

#endif
