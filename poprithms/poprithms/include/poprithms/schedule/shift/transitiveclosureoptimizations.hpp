// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_SCHEDULE_SHIFT_TRANSITIVECLOSUREOPTIMIZATIONS
#define POPRITHMS_SCHEDULE_SHIFT_TRANSITIVECLOSUREOPTIMIZATIONS

#include <algorithm>
#include <array>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

namespace poprithms {
namespace schedule {
namespace shift {

// All the currently supported optimizations which can be run to make
// graph scheduling faster. See the method comments for information on what
// they each do.
//
// These optimizations reduce the search space, while ensuring*  that the
// globally optimal schedule remains in the search space. That is, they
// eliminate regions of the search space with "bad" schedules.
//
// *We have some proofs of global optimality for some of the optimizations,
// but others don't yet have rigorous proofs.
enum class TransitiveClosureOptim {
  SlideLinks = 0,
  LinkTightDrops,
  LinkCloseTightPairs,
  ConstrainWeightSeparatedGroups,
  ConstrainParallelChains,
  CombineAllocsWithCommonOps,
  DisconnectAllocsWithOneOp,
  DisconnectAllocsWithZeroWeight,
  DisconnectInbetweenerAllocs,
  DisconnectFixedDurationAllocs,
  ConnectContiguousAllocs,
  N
};

class TransitiveClosureOptimizations {

public:
  /**
   * Return a string corresponding to the enum value #optim
   * */
  static std::string str(TransitiveClosureOptim optim);

  /**
   * Create a TransitiveClosureOptimizations with all optimizations off. To
   * create a TransitiveClosureOptimization with (say) only optimizations
   * 'foo' and 'bar' enabled, you can use
   *
   * <code>
   *    auto tcos = allOff().withFoo(true).withBar(false);
   * </code>
   * */
  static TransitiveClosureOptimizations allOff() { return all(false); }

  /**
   * Return true iff all optimizations are off.
   * */
  bool allOptimizationsOff() const;

  /**
   * Create a TransitiveClosureOptimizations with all optimizations on. To
   * create a TransitiveClosureOptimization with (say) only optimizations
   * 'foo' and 'bar' disabled, you can use
   *
   * <code>
   *    auto tcos = allOn().withFoo(false).withBar(false);
   * </code>
   * */
  static TransitiveClosureOptimizations allOn();

  /**
   * Return true iff all optimizations are on.
   * */
  bool allOptimizationsOn() const;

  /**
   * Recall:
   * A pair of Ops (a,b) is defined to be a "tight pair" if
   *   1) b is the only output of a,
   *   2) a is the only input of b.
   *
   * A pair of Ops (a,b) forms a 'linked pair' if there is hard constraint
   * that b appears directly after a (with no Op inbetween).
   *
   * In terms of reducing the search space of possible schedules, ops which
   * are 'linked' are better than ops which only have an ordinary constraint.
   *
   * If it can be determined that there is a local minimum for the switch
   * scheduler in which 2 unlinked Ops are contiguous, then they can be linked
   * , so as to reduce the search space of possible schedules.
   *
   * LinkTightDrops. If (a,b) is a tight pair, and b is guaranteed to increase
   * liveness less than a, then upgrade (a,b) to a linked pair.
   * */
  TransitiveClosureOptimizations &withLinkTightDrops(bool);
  bool linkTightDrops() const;

  /**
   * LinkCloseTightPairs. If (a,b) is a tight pair, and there is no Op c in
   * the unconstrained dual of a which can have an increase in liveness equal
   * to or inbetween those of a and b, then upgrade (a,b) to a linked pair.
   * Example
   *
   *  +---a--->-b
   *  |
   *  c->-d-->--e
   *
   * d and e are in the unconstrained dual of a. If the effect on liveness of
   * neither d nor e is between the effect of a and b, the a and b will always
   * be scheduled contiguously in an optimal schedule.
   * */
  TransitiveClosureOptimizations &withLinkCloseTightPairs(bool);
  bool linkCloseTightPairs() const;

  /**
   * ConstrainWeightSeparatedGroups. If a and b have common inputs, and it is
   * guaranteed that the increases in livenesses in PostUnconstrained(a,b) are
   * all less than or equal to those in PostUnconstrained(b,a), then insert a
   * constraint a->b, and some additional related constraints.
   *
   * Recall that PostUnconstrained(x,y) is all Ops which are after x and
   * unconstrained w.r.t. y.
   *
   * a -> A --+
   *          +--> C
   * b -> B --+
   *
   * The set A above is PostUnconstrained(a, b), and B is PostUnconstrained(b,
   * a). So this optimization insert constraints a->b and some others (some of
   * A to some of B) if (a,A) are "better" than (b,B).
   * */
  TransitiveClosureOptimizations &withConstrainWeightSeparatedGroups(bool);
  bool constrainWeightSeparatedGroups() const;

  /**
   * ConstrainParallelChains. If a and b have common inputs, and both belong
   * to tight chains with common outputs, and if (1) a's chain is not shorter
   * than b's and (2) the cumulative increase in liveness along a's chain is
   * never greater than along b's, then insert constraints from a's chain to
   * b's chain, to form a ladder of constraints
   * */
  TransitiveClosureOptimizations &withConstrainParallelChains(bool);
  bool constrainParallelChains() const;

  /**
   * \sa AllocSimplifier::combineAllocsWithCommonOps
   * */
  TransitiveClosureOptimizations &withCombineAllocsWithCommonOps(bool);
  bool combineAllocsWithCommonOps() const;

  /**
   * \sa AllocSimplifier::disconnectAllocsWithOneOp
   * */
  TransitiveClosureOptimizations &withDisconnectAllocsWithOneOp(bool);
  bool disconnectAllocsWithOneOp() const;

  /**
   * \sa AllocSimplifier::disconnectAllocsWithZeroWeight
   * */
  TransitiveClosureOptimizations &withDisconnectAllocsWithZeroWeight(bool);
  bool disconnectAllocsWithZeroWeight() const;

  /**
   * \sa AllocSimplifier::disconnectInbetweenerAllocs
   * */
  TransitiveClosureOptimizations &withDisconnectInbetweenerAllocs(bool);
  bool disconnectInbetweenerAllocs() const;

  /**
   * \sa AllocSimplifier::disconnectFixedDurationAllocs
   * */
  TransitiveClosureOptimizations &withDisconnectFixedDurationAllocs(bool);
  bool disconnectFixedDurationAllocs() const;

  /**
   * \sa AllocSimplifier::connectContiguousAllocs
   * */
  TransitiveClosureOptimizations &withConnectContiguousAllocs(bool);
  bool connectContiguousAllocs() const;

  TransitiveClosureOptimizations &withMaxIterations(int);
  int maxIterations() const { return maxNumberOfIterations; }

  /**
   * SlideLinks is always enabled if any other is enabled. This transformation
   * generates constraints from links, which are added to a transitive
   * closure.
   * */
  bool slideLinks() const { return !allOptimizationsOff(); }

  bool operator==(const TransitiveClosureOptimizations &rhs) const;
  bool operator!=(const TransitiveClosureOptimizations &rhs) const;
  bool operator<(const TransitiveClosureOptimizations &rhs) const;

  std::vector<TransitiveClosureOptim> enabled() const;

  void append(std::ostream &) const;

private:
  // Each of the optional optimizations inherits from this base class:
  struct Option {
    Option() = delete;
    Option(bool on__) : on_(on__) {}
    bool on() const { return on_; }
    bool off() const { return !on(); }
    Option &update(bool on__) {
      on_ = on__;
      return *this;
    }
    void append(std::ostream &oss) const {
      oss << name() << " : " << (on() ? "Yes" : "No");
    }

    virtual TransitiveClosureOptim getEnum() const = 0;

  private:
    bool on_{false};
    std::string name() const { return str(getEnum()); }

    virtual void noWeakVTables();
  };

  struct LinkTightDrops : public Option {
    LinkTightDrops(bool on) : Option(on) {}
    TransitiveClosureOptim getEnum() const final;
  } linkTightDrops_;

  struct LinkCloseTightPairs : public Option {
    LinkCloseTightPairs(bool on) : Option(on) {}
    TransitiveClosureOptim getEnum() const final;
  } linkCloseTightPairs_;

  struct ConstrainWeightSeparatedGroups : public Option {
    ConstrainWeightSeparatedGroups(bool on) : Option(on) {}

    TransitiveClosureOptim getEnum() const final;
  } constrainWeightSeparatedGroups_;

  struct ConstrainParallelChains : public Option {
    ConstrainParallelChains(bool on) : Option(on) {}

    TransitiveClosureOptim getEnum() const final;
  } constrainParallelChains_;

  struct CombineAllocsWithCommonOps : public Option {
    CombineAllocsWithCommonOps(bool on) : Option(on) {}

    TransitiveClosureOptim getEnum() const final;
  } combineAllocsWithCommonOps_;

  struct DisconnectAllocsWithOneOp : public Option {
    DisconnectAllocsWithOneOp(bool on) : Option(on) {}

    TransitiveClosureOptim getEnum() const final;

  } disconnectAllocsWithOneOp_;

  struct DisconnectAllocsWithZeroWeight : public Option {
    DisconnectAllocsWithZeroWeight(bool on) : Option(on) {}

    TransitiveClosureOptim getEnum() const final;
  } disconnectAllocsWithZeroWeight_;

  struct DisconnectInbetweenerAllocs : public Option {
    DisconnectInbetweenerAllocs(bool on) : Option(on) {}

    TransitiveClosureOptim getEnum() const final;
  } disconnectInbetweenerAllocs_;

  struct DisconnectFixedDurationAllocs : public Option {
    DisconnectFixedDurationAllocs(bool on) : Option(on) {}

    TransitiveClosureOptim getEnum() const final;
  } disconnectFixedDurationAllocs_;

  struct ConnectContiguousAllocs : public Option {
    ConnectContiguousAllocs(bool on) : Option(on) {}

    TransitiveClosureOptim getEnum() const final;
  } connectContiguousAllocs_;

  int maxNumberOfIterations;

  std::vector<const Option *> getOptions() const {
    return {&linkTightDrops_,
            &linkCloseTightPairs_,
            &constrainWeightSeparatedGroups_,
            &constrainParallelChains_,
            &combineAllocsWithCommonOps_,
            &disconnectAllocsWithOneOp_,
            &disconnectAllocsWithZeroWeight_,
            &disconnectInbetweenerAllocs_,
            &disconnectFixedDurationAllocs_,
            &connectContiguousAllocs_};
  }

  explicit TransitiveClosureOptimizations(LinkTightDrops a,
                                          LinkCloseTightPairs b,
                                          ConstrainWeightSeparatedGroups c,
                                          ConstrainParallelChains d,
                                          CombineAllocsWithCommonOps e,
                                          DisconnectAllocsWithOneOp f,
                                          DisconnectAllocsWithZeroWeight g,
                                          DisconnectInbetweenerAllocs h,
                                          DisconnectFixedDurationAllocs i,
                                          ConnectContiguousAllocs j,
                                          int mits)
      : linkTightDrops_(a), linkCloseTightPairs_(b),
        constrainWeightSeparatedGroups_(c), constrainParallelChains_(d),
        combineAllocsWithCommonOps_(e), disconnectAllocsWithOneOp_(f),
        disconnectAllocsWithZeroWeight_(g), disconnectInbetweenerAllocs_(h),
        disconnectFixedDurationAllocs_(i), connectContiguousAllocs_(j),
        maxNumberOfIterations(mits) {}

  static TransitiveClosureOptimizations all(bool);
};

std::ostream &operator<<(std::ostream &os,
                         const TransitiveClosureOptimizations &);

std::ostream &operator<<(std::ostream &os, const TransitiveClosureOptim &);
std::ostream &operator<<(std::ostream &os,
                         const std::vector<TransitiveClosureOptim> &);

} // namespace shift
} // namespace schedule
} // namespace poprithms

#endif
