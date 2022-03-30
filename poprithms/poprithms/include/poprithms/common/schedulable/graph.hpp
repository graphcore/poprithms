// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_SCHEDULABLE_GRAPH_HPP
#define POPRITHMS_COMMON_SCHEDULABLE_GRAPH_HPP

#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <poprithms/common/multiout/fwdedgemap.hpp>
#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/common/multiout/optionaltensorid.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/common/schedulable/additionalfwdedges.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/schedule/vanilla/vanilla.hpp>

namespace poprithms {
namespace common {
namespace schedulable {

class Op;

using common::multiout::FwdEdgeMap;
using common::multiout::InIndices;
using common::multiout::OpId;
using common::multiout::OpIds;
using common::multiout::OptionalTensorId;
using common::multiout::OptionalTensorIds;
using common::multiout::TensorId;
using common::multiout::TensorIds;

/**
 * The common::multiout::Graph from which this Graph inherits, does not have
 * control dependencies. It only has data dependencies implicitly defined by
 * the Tensors produced and consumed by Ops. This class introduces control
 * dependencies between Ops, which needn't be data dependencies.
 *
 * The second feature which this Graph introduces is a partitioning into
 * sub-graphs of the Ops. Each Op has a single SubGraphId attribute. Control
 * dependencies can only be introduced between Ops with the same SubGraphId.
 * */
class Graph : public common::multiout::Graph {

public:
  Graph()              = default;
  Graph(Graph &&)      = default;
  Graph(const Graph &) = default;
  Graph &operator=(Graph &&) = default;
  Graph &operator=(const Graph &) = default;
  virtual ~Graph() override       = default;

  /**
   * Insert a topological constraint between 2 Ops, #before and #after, which
   * ensures that #before is scheduled before #after is. #before and #after
   * must be Ops with the same SubGraphId.
   * */
  void constraint(OpId before, OpId after);

  /**
   * Insert a topological constraint between the producers of 2 Tensors,
   * #before and #after.
   * */
  void constraint(const TensorId &before, const TensorId &after) {
    constraint(before.opId(), after.opId());
  }

  /**
   * Insert multiple constraints, 1 between every consecutive pair of
   * Ops/Tensors.
   * */
  template <class Arg0, class... Args>
  void constraint(Arg0 a, Arg0 b, Args... c) {
    constraint(a, b);
    constraint(b, c...);
  }

  /**
   * Insert multiple constraints, 1 constraint from #before to every Op in
   * #afters.
   * */
  void constraint(OpId before, const OpIds &afters);

  /**
   * Insert multiple constraints, 1 constraint from every Op in #befores to
   * the Op #after.
   * */
  void constraint(const OpIds &befores, OpId after);

  /**
   * A method for inserting constraints between groups of Ops.
   *
   * \param bins Operations in different elements of \a bins will be
   *             scheduled in increasing bin index. For example, if Op #a is
   *             in bins[0] and Op #b is in bins[2], then #a will be scheduled
   *             before #b.
   * */
  void binConstraint(const std::vector<OpIds> &bins);

  /**
   * Get all Ops in the Graph which have SubGraphId #subGraphId.
   * */
  OpIds opIds(SubGraphId subGraphId) const;

  /**
   * Ids of all live ops.
   * */
  OpIds opIdsAllSubGraphs() const { return multiout::Graph::opIds(); }

  /**
   * Get all Tensors in the Graph which have SubGraphId #subGraphId.
   * */
  TensorIds tensorIds(SubGraphId subGraphId) const;

  /**
   * A link is a strong type of constraint. It not only ensures that an Op
   * #before appears before some Op #after, but it also ensures that there
   * are no other Ops between then. That is, #before and #after are guaranteed
   * to appear contiguously in the schedule, if there is a link between them.
   * */
  // TODO(T39536) link constrains are quite powerful, and used in popart.
  void link(OpId before, OpId after);

  /**
   * Insert multiple links, one between every pair of Ops in the argument
   * list.
   * */
  template <class Arg0, class... Args> void link(Arg0 a, Arg0 b, Args... c) {
    link(a, b);
    link(b, c...);
  }

  /**
   * Insert multiple links, one between every contiguous pair of Ops in
   * #opIds.
   * */
  void link(const OpIds &opIds);

  /**
   * Links can be expressed entirely in terms of simple constraints. This
   * method reduces all links to constraints. Note that there is no
   * guarantee than an Op added to this Graph after this method is called
   * does not appear between 2 Ops which had a link between them before this
   * method is called.
   * */
  // TODO(T39536)
  void simplifyLinks();

  /**
   * return the SubGraphId of the Op #opId.
   * */
  SubGraphId subGraphId(OpId opId) const;

  /**
   * return the SubGraphId of the creator of #tId.
   * */
  SubGraphId subGraphId(const TensorId &tId) const;

  /**
   * return the SubGraphIds of all the creators of the Tensors in #tIds, in
   * order.
   * */
  SubGraphIds subGraphIds(const TensorIds &tIds) const;

  /**
   * Return a 'cheap' scheduling of this Graph
   *
   * \sa schedule::vanilla
   * */
  OpIds
  vanillaSchedule(const AdditionalFwdEdges & = NoAdditionalFwdEdges()) const;

  /**
   * A sub-schedule of a set of ops. This is equivalent to
   * 1) get the schedule for the complete graph, then
   * 2) pull out the entries in #opIds, retaining their relative positions.
   * */
  OpIds vanillaSubSchedule(
      const std::set<OpId> &,
      const AdditionalFwdEdges & = NoAdditionalFwdEdges()) const;

  /**
   * Return a random scheduling of this Graph
   * */
  OpIds
  randomSchedule(uint32_t seed,
                 const AdditionalFwdEdges & = NoAdditionalFwdEdges()) const;

  /**
   * Return a schedule of this Graph, but partitioned by SubGraphId.
   * */
  std::vector<OpIds>
  vanillaSchedules(const AdditionalFwdEdges & = NoAdditionalFwdEdges()) const;

  std::vector<OpIds>
  randomSchedules(uint32_t,
                  const AdditionalFwdEdges & = NoAdditionalFwdEdges()) const;

  /**
   * Return a schedule of all Ops in a single SubGraphId.
   * */
  OpIds vanillaSubGraphSchedule(
      SubGraphId,
      const AdditionalFwdEdges & = NoAdditionalFwdEdges()) const;

  OpIds randomSubGraphSchedule(
      SubGraphId,
      uint32_t,
      const AdditionalFwdEdges & = NoAdditionalFwdEdges()) const;

  uint64_t nSubGraphs() const { return subGraphStates.size(); }

  /**
   * return true if there is exactly one way to schedule this Graph
   * */
  bool hasUniqueSchedule(
      SubGraphId,
      const AdditionalFwdEdges & = NoAdditionalFwdEdges()) const;

  /**
   * In some situations, redundant constraints can be removed from a Graph
   * without increasing the number of possible schedules. This can be useful
   * to accelerate operations which are O(Edges).
   * */
  // TODO(T39701) implement this using TransitiveClosures.
  void simplifyConstraints() const;

  /**
   * Convert all of the constraints in this Graph to a forward edge map.
   *
   * \return A FwdEdgeMap X, where X[opId] contains all Ops which must be
   *         scheduled after opId.
   * */
  FwdEdgeMap getForwardEdgeMap_u64(
      const AdditionalFwdEdges & = NoAdditionalFwdEdges()) const;

  FwdEdgeMap getSubGraphForwardEdgeMap_u64(
      SubGraphId,
      const AdditionalFwdEdges & = NoAdditionalFwdEdges()) const;

  /**
   * Generate a new SubGraphId, with name #graphName.
   * */
  SubGraphId createSubGraphId(const std::string &graphName);

  /**
   * Get the name of #subGraphId.
   * */
  std::string subGraphName(SubGraphId subGraphId) const;

  /**
   * It can be useful to ensure that the Ops added to this Graph are
   * guaranteed to be scheduled in the order that they are added. This
   * simulates the experience of 'eager mode' graph execution, where
   * operations are executed exactly in the order they appear, in a python
   * script for example.
   *
   * This 'eager order' is disabled by default, and can be switched on and
   * off at any time during Graph construction.
   *
   * All Ops inserted while in eager mode are guaranteed to be scheduled in
   * the order they are added.
   *
   * \param subGraphId enable/disable eager order for Ops with this
   *        SubGraphId.
   *
   * \param enable if true, then eager order is enabled.
   * */
  void toggleEager(SubGraphId subGraphId, bool enable);

  /**
   * return true of the #subGraphId is currently in eager mode.
   * */
  bool eagerIsEnabled(SubGraphId subGraphId) const;

  /**
   * Insert constraints to ensure that Op #opId is scheduled after all other
   * Ops currently in the Graph with its SubGraphId.
   * */
  void ensureLastOfCurrentOps(OpId opId);

  /**
   * Get a consensus on the SubGraphId from all Tensors in #tIds. If not all
   * Tensors in #tIds have the same SubGraphId, or if #tIds is empty, then
   * an error is thrown.
   * */
  SubGraphId subGraphIdFromTensorIds(const TensorIds &tIds) const;
  SubGraphId subGraphIdFromTensorIds(const std::vector<TensorIds> &) const;

  /**
   * Assert that the tensors in #tIds are in the sub-graph #sgId. If they are
   * not, a descriptive error is thrown.
   * */
  void assertSubGraphId(const TensorIds &tIds, SubGraphId sgId) const;

  /**
   * return all Ops with #subGraphId which can be scheduled last. That is,
   * all Ops which have no out Ops.
   * */
  OpIds
  mayBeFinals(SubGraphId subGraphId,
              const AdditionalFwdEdges & = NoAdditionalFwdEdges()) const;

  /**
   * All of the Ops which must be scheduled before #opId due to an explict
   * non-data dependency constraint.
   * */
  OpIds controlDependencyInOps(OpId opId) const;

  /**
   * All of the Ops which must be scheduled before #opId due to a data
   * dependency.
   * */
  OpIds dataDependencyInOps(OpId opId) const;

  /**
   * All of the Ops which must be scheduled after #opId due to an explict
   * non-data dependency constraint.
   * */
  OpIds controlDependencyOutOps(OpId opId) const;

  /**
   * All of the Ops which must be scheduled after #opId due to a data
   * dependency.
   * */
  OpIds dataDependencyOutOps(OpId opId) const;

private:
  /**
   * All Ops which must be scheduled before #opId, either because of a data
   * dependency, or a non-data "control" dependency.
   * */
  OpIds allInOps(OpId opId) const;

  /**
   * All Ops which must be scheduled after #opId, either because of a data
   * dependency, or a non-data "control" dependency.
   * */
  OpIds allOutOps(OpId opId) const;

public:
  std::vector<poprithms::util::StringColumn>
  getSchedulableColumns(const OpIds &) const;

  std::vector<poprithms::util::StringColumn> getSchedulableColumns() const {
    return getSchedulableColumns(multiout::Graph::opIds());
  }

  /**
   * Verify that this Graph is in a valid state, including all state inherited
   * from base classes.
   * */
  void assertSchedulableGraphCorrectness() const;

  const Op &schedulableOp(OpId opId) const { return op(opId); }

  /**
   * How control dependencies should be propagated when graphs are
   * modified might depend on the specific transformation. This placeholder
   * method will need extension in the future.*/
  enum class ControlDependencyPropagationType {
    ///< If 'from' (see method below) has a control dependency to be before x,
    ///< then all data and control dependency inputs to 'from' will remain
    ///< before x after opId is removed.
    ConserveLocally
  };
  void propagateControlDependencies(OpId from,
                                    ControlDependencyPropagationType);

  /** Get a vector of SubGraphIds from a vector of integers. */
  SubGraphIds asSubGraphIds(const std::vector<uint64_t> &) const;

  /** Get a vector of integers from a vector of SubGraphIds. */
  std::vector<uint64_t> asUnsigned64s(const SubGraphIds &) const;

protected:
  OpId insertSchedulableOp(std::unique_ptr<Op>);

  // TODO(T49671): reconsider the naming of this method.
  bool schedulableTypeSpecificEqualTo(const Graph &rhs) const {
    // All of the state of this Graph is captures in this single field
    return rhs.subGraphStates == subGraphStates;
  }

  // inheriting classes must implement this method, to safely remove derived
  // attributes of ops.
  virtual void schedulableTypeSpecificRemoveOp(
      OpId opToRemove,
      const OptionalTensorIds &outputSubstitutes) = 0;

  // replacements must be in the same sub-graph.
  virtual void schedulableTypeSpecificVerifyValidOutputSubstitute(
      const TensorId &before,
      const TensorId &after) const = 0;

private:
  // This method will remove the control dependencies of opToRemove, ensuring
  // that there are no dangling ends when it is removed. No control
  // dependncies are 'transferred', these must be done before the call to
  // #removeOp, using for example propagateControlDependencies.
  void multiOutTypeSpecificRemoveOp(
      OpId opToRemove,
      const OptionalTensorIds &outputSubstitutes) final;

  // This method will remove
  void multiOutTypeSpecificVerifyValidOutputSubstitute(
      const TensorId &before,
      const TensorId &after) const final;

  // Ops which inherit from this class should use insertSchedulableOp, and
  // not insertMultioutOp. By having this 'using' here, we make this method
  // private, preventing its use in derived classes.
  using common::multiout::Graph::insertMultioutOp;

  /**
   * Insert a "null" Op which serves no purpose other than to separate bins of
   * Ops. This is used in the insertBinBoundary method.
   * */
  virtual OpId insertBinBoundary(SubGraphId) = 0;

  Op &op(OpId);
  const Op &op(OpId) const;

  // TODO(T39700) there's a quicker path if relatively few Ops have been
  // deleted, using a vector instead of an unordered_map. That is why I've
  // called this the "sparse" method, maybe we can implement a "dense" method
  // later.
  //
  // Note that this method assumes that opIds is a "complete" sub-graph,
  // that is all dependencies are present. There is no check that this is
  // the case.
  FwdEdgeMap getSparseForwardEdgeMap_u64(
      const OpIds &,
      const AdditionalFwdEdges & = NoAdditionalFwdEdges()) const;

  /**
   * Derived classes can optionally add extra scheduling constraints. These
   * are in addition to the data and control dependencies. This method returns
   * a map where, for key:values, for every v in values, there is a constraint
   * key->v. All keys and values must be in the #opIds passed in as the method
   * argument.
   * */
  virtual std::map<OpId, OpIds>
  schedulableDerivedSpecificConstraints(const OpIds &opIds) const = 0;

  // Separate Ops by SubGraphId
  std::vector<OpIds> subGraphPartitioned(const OpIds &) const;

  /**
   * The Graph class is global, in the same way as a poplar::Graph is.
   * The concept of a sub-graph/program can be partially captured by
   * annotating Ops with SubGraphIds. SubGraphIds all have (user provided)
   * strings associated with them to help debugging and make logging clearer.
   */
  class SubGraphState {
  public:
    // struct for the tracking of eager scheduling mode:
    enum class Eager { Disabled = 0, Enabled };

    // Construct a SubGraphState from a name. It has eager scheduling mode
    // disabled.
    SubGraphState(const std::string &name)
        : name_(name), eager_(Eager::Disabled), hasLast_(false), last_(-1),
          ops_({}) {}

    const std::string &name() const { return name_; }
    bool eagerEnabled() const { return eager_ == Eager::Enabled; }

    // If this SubGraph was in eager mode when the last Op was added, this is
    // the Op that was added:
    OpId knownLast() const { return last_; }

    bool hasKnownLast() const { return hasLast_; }

    // All the Ops which are in this SubGraph.
    OpIds ops() const { return OpIds(ops_.cbegin(), ops_.cend()); }

    // Change eager mode.
    void toggleEager(bool);

    bool operator==(const SubGraphState &rhs) const { return t() == rhs.t(); }
    bool operator<(const SubGraphState &rhs) const { return t() < rhs.t(); }

    void setLast(OpId opId) {
      eager_   = Eager::Enabled;
      hasLast_ = true;
      last_    = opId;
    }

    void insertBack(OpId opId) { ops_.insert(/* hint = */ ops_.end(), opId); }

    void removeOp(OpId opId);

  private:
    std::tuple<std::string, Eager, bool, OpId> t() const {
      return {name_, eager_, hasLast_, last_};
    }
    std::string name_;
    Eager eager_;
    bool hasLast_;
    OpId last_;
    // using a set, to make removing easier, when eventually we get to it.
    std::set<OpId> ops_;
  };
  std::vector<SubGraphState> subGraphStates;
};

} // namespace schedulable
} // namespace common
} // namespace poprithms

#endif
