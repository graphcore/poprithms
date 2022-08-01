// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COLORING_PROPAGATOR_HPP
#define POPRITHMS_COLORING_PROPAGATOR_HPP

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <poprithms/error/error.hpp>

namespace poprithms {
namespace coloring {

/**
 * Interface for propagating a value along the edges of a directed graph.
 *
 * \tparam Node The node type in the directed graph.
 * \tparam Color The 'value' is parameterized by by this. Each Node has one
 *               Color.
 *
 * The user implements an interface for a generic directed graph, and can
 * then set node colors and propagate them to neighbors in various ways.
 *
 * One use case is for partitioning a graph into pipeline stages. A user might
 * know which stages some operations must be in, but then want some automated
 * way of filling in the stages for the unset ops. For this example, op=Node
 * and stage=Color.
 * */
template <typename Node, typename Color> class IPropagator {

private:
  using Nodes  = std::vector<Node>;
  using Colors = std::vector<Color>;

public:
  virtual ~IPropagator() = default;

  /**
   * Set the color of #node to #color. If #node already has a color that is
   * different to #color, then an error is thrown.
   * */
  void setColor(Node node, Color color) {
    auto found = mNodeToColor.find(node);

    // Case where a color has already been set for #node:
    if (found != mNodeToColor.cend()) {
      if (found->second != color) {
        std::ostringstream oss;
        oss << "Attempt to set color of the node " << node << " to " << color
            << ", but the color is already set to " << found->second
            << ". Node string : " << nodeString(node);
        throw poprithms::error::error("coloring", oss.str());
      } else {
        return;
      }
    }

    // Case where it's the first time a color is set for #node:
    {
      // node:color
      mNodeToColor.insert({node, color});

      // color:nodes
      auto found2 = mColorToNodes.find(color);
      if (found2 == mColorToNodes.cend()) {
        mColorToNodes.insert({color, {node}});
      } else {
        found2->second.push_back(node);
      }
    }
  }

  /**
   * Set the color of #node to #color. Proagate #color backwards from #node to
   * all nodes which do not have a color. Backward edges are defined by the
   * virtual method #ins.
   * */
  void setAndPropagateBackward(Node node, Color color) {
    setColor(node, color);
    propagateBackward(node);
  }

  /**
   * Set the color of #node to #color. Proagate #color forward from #node to
   * nodes which do not have a color. Forward edges are defined by the method
   * #outs.
   * */
  void setAndPropagateForward(Node node, Color color) {
    setColor(node, color);
    propagateForward(node);
  }

  /**
   * The color of #node. If the color has not previously been set, an error is
   * thrown.
   * */
  Color color(Node node) const {
    auto found = mNodeToColor.find(node);
    if (found == mNodeToColor.cend()) {
      std::ostringstream oss;
      oss << "No color for the node " << node
          << " is set. Node string : " << nodeString(node);
      throw poprithms::error::error("coloring", oss.str());
    }
    return found->second;
  }

  /**
   * Propagate the color of #node forward to all nodes which do not have a
   * color. Forward edges are defined by the virtual method, #outs.
   * */
  void propagateForward(Node node) {
    return propagate(node, [this](Node node) { return outs(node); });
  }

  /**
   * Propagate the color of #node backwards to all nodes which do not have a
   * color. Backwards edges are defined by the virtual method, #ins.
   * */
  void propagateBackward(Node node) {
    return propagate(node, [this](Node node) { return ins(node); });
  }

  /**
   * Propagate the color of #node to all nodes in its connected component.
   * Connections between nodes are defined by the virtual methods #ins and
   * #outs. Nodes which already have colors set are not considered to be
   * neighbors.
   * */
  void propagateForwardAndBackward(Node node) {
    auto x = [this](Node node) {
      auto a = ins(node);
      auto b = outs(node);
      a.insert(a.end(), b.cbegin(), b.cend());
      return a;
    };

    return propagate(node, x);
  }

  /**
   * Propagate the color #v starting from all nodes with color #v.
   * */
  void propagateAllForwardAndBackward(Color v) {
    for (auto node : allWithColor(v)) {
      propagateForwardAndBackward(node);
    }
  }

  void propagateAllBackward(Color v) {
    for (auto node : allWithColor(v)) {
      propagateBackward(node);
    }
  }

  void propagateAllForward(Color v) {
    for (auto node : allWithColor(v)) {
      propagateForward(node);
    }
  }

  void propagateAllForwardAndBackward(const Colors &vals) {
    for (auto v : vals) {
      propagateForwardAndBackward(v);
    }
  }

  bool hasColor(Node node) const {
    return mNodeToColor.find(node) != mNodeToColor.cend();
  }

  /**
   * \return All nodes which have been set to have color #color.
   * */
  const Nodes &allWithColor(Color color) const {
    auto found = mColorToNodes.find(color);
    if (found != mColorToNodes.cend()) {
      return found->second;
    }
    return empty_;
  }

  /**
   * Set nodes without any color to have color #to. Which nodes?
   *
   * Starting from all nodes with color #from:
   *   1) get all outs.
   *   2) filter (1) to only those without colors.
   *   3) filter (2) to only those which satisfy #condition.
   *
   * Propagate the color #to forward from all nodes in (3).
   * */
  template <class Condition>
  void flushForward(Color from, Color to, const Condition &condition) {
    for (auto src : allWithColor(from)) {
      for (auto dst0 : outs(src)) {
        if (!hasColor(dst0) && condition(dst0)) {
          setAndPropagateForward(dst0, to);
        }
      }
    }
  }

  const std::map<Node, Color> &colorMap() const { return mNodeToColor; }

  /**
   * The interface that must be implemented by the inheriting class. It
   * defines the forward and backward edges between nodes, and a method which
   * returns a string from nodes, which allows for better context in error
   * messages.
   * */
  virtual Nodes outs(Node) const             = 0;
  virtual Nodes ins(Node) const              = 0;
  virtual std::string nodeString(Node) const = 0;

private:
  std::map<Node, Color> mNodeToColor;
  std::map<Color, Nodes> mColorToNodes;
  Nodes empty_{};

  void visitNode(Node nxt, Color color, std::vector<Node> &stack_) {
    if (!hasColor(nxt)) {
      setColor(nxt, color);
      stack_.push_back(nxt);
    }
  }

  template <typename F> void propagate(Node node, F &&f) {
    auto color_ = color(node);
    Nodes stack_{node};
    while (!stack_.empty()) {
      auto nxt = stack_.back();
      stack_.pop_back();
      for (auto nodeNxt : f(nxt)) {
        visitNode(nodeNxt, color_, stack_);
      }
    }
  }
};

} // namespace coloring
} // namespace poprithms

#endif
