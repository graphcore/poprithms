// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_MEMORY_UNWIND_GRAPH_HPP
#define POPRITHMS_MEMORY_UNWIND_GRAPH_HPP

#include <array>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/graph.hpp>
#include <poprithms/common/multiout/tensorid.hpp>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/memory/nest/region.hpp>
#include <poprithms/memory/unwind/path.hpp>
#include <poprithms/memory/unwind/subgraphid.hpp>
#include <poprithms/memory/unwind/sumlike.hpp>
#include <poprithms/memory/unwind/valuedtensorid.hpp>
#include <poprithms/util/copybyclone.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace memory {
namespace unwind {

using common::multiout::TensorIds;
using memory::nest::Region;
using ndarray::Dimension;
using ndarray::Dimensions;
using Lower = ndarray::Shape::Lower;
using Upper = ndarray::Shape::Upper;
using chain::Chain;
using memory::nest::Region;
using ndarray::Shape;
using ndarray::Shapes;
using ndarray::Stride;
using ndarray::Strides;
using util::Permutation;

class Op;

/**
 * This is a Graph designed with the bare essentials for descibing algorithms
 * for determining good layouts of Graph inputs, based on desirable layouts of
 * internal Tensors. It doesn't define what \a layout means, this is
 * application specific, but it does describe the relationships between
 * Tensors of their layouts.
 *
 * This is a challenging problem faced by frameworks built on top of
 * Pop(lar/Libs). Poplibs provides APIs for setting Tensor layouts for certain
 * operations such as matmuls and convolutions. If a Graph input Tensor does
 * not go directly into a one of these special operations which has an API for
 * setting layouts, it is not obvious how the Tensor should be mapped to
 * tiles. The responsibility is on the user of Poplibs, i.e. the machine
 * learning framework, to map input Tensors to IPU tiles.
 *
 * This Graph class is for \a describing the unwinding problem. Solving the
 * problem is the responsibility of the \a Solution class.
 *
 * This documentation is complemented by the markdown file,
 * notes/unwinding/Unwinding.md, which contains clear diagrams and more
 * indepth examples.
 * */

/**
 *
 * First example (basics of unwinding)
 * -----------------------------------
 * Suppose the computation is
 *
 *   out = matmul(X.dimShuffle(perm), concat(Y, Z.reverse(dims)))
 *
 *  Diagramatically,
 *
 *       X             Y    Z
 *       |             |    |
 *  dimShuffle(perm)   |    |
 *       |             |    |
 *       |             |  reverse(dims)
 *       |             |    |
 *       |             +-+--+
 *       |               |
 *       |             concat
 *       |               |
 *       +-------+-------+
 *               |
 *             matmul.
 *
 * The graph for this computation has 3 inputs, X, Y, and Z. These 3 inputs
 * need to have their inputs set.
 *
 * Poplibs' matmul is optimised for inputs which have very specific layouts.
 * To help the user of Poplibs create Tensors with these layout, there are
 * special API functions. For example there is
 * #createMatmulLHS(shape_lhs, shape_rhs), which returns a Tensor of shape
 * shape_lhs, with a tile mapping specialised for this particular matmul.
 *
 * Looking at the diagram above, we know that X.dimShuffle(perm) should have
 * this specialised layout, as it enters a matmul on the lhs. Let's call a
 * Tensor with this specialized layout \a LHS. The user of Popl(ar/ibs) needs
 * to create the Tensor X though, what layout should X have? It should be
 *
 * layout(X) = layout(LHS.dimShuffle(perm.inverse())),
 *
 * because then,
 *
 * layout(X.dimShuffle(perm))
 *    = layout(LHS.dimShuffle(perm.inverse()).dimShuffle(perm))
 *    = layout(LHS),
 *
 * which is exactly what we want for the lhs input to the matmul.
 *
 * The general approach is to start from points where the desired layout is
 * known, such as LHS, and then "unwind" back to the inputs. This is what we
 * did to find the layout of X.
 *
 * Poplibs also provides an API to set the layout for the RHS input of a
 * matmul. Using the same unwinding approach as for LHS, we see that
 *
 * layout(Y) = layout(RHS.slice(.)), and
 *
 * layout(Z) = layout(RHS.slice(.).reverse(dims)).
 *
 * That's the basic idea of unwinding. Starting from internal points where
 * the desired layout is known, backtrack or "unwind" through the graph to the
 * inputs. We will next look at some slightly more complex examples.
 *
 * */

/**
 *
 * Second example (an unbroadcast add)
 * -----------------------------------------
 * In the first example, the layouts of inputs were unwound directly from
 * Tensors with known optimal layouts (LHS and RHS). Let us now extend that
 * example, to a case where that is not possible for one of the input Tensors:
 *
 *   out0 = matmul(X.dimShuffle(perm), concat(Y, Z.reverse(dims))),
 *   out1 = Q + X.
 *
 *     Q  +------X             Y    Z
 *     |  |      |             |    |
 *     |  |      |             |    |
 *     |  |      |             |  reverse(dims)
 *     |  |      |             |    |
 *     |  |      |             +-+--+
 *     |  |      |               |
 *     |  | dimShuffle          concat
 *     |  |      |               |
 *     +-++      +-------+-------+
 *       |               |
 *      add           matmul
 *       |               |
 *     [out1]          [out0]
 *
 * This graph has 4 inputs: the 3 which we met in the first example (X, Y and
 * Z), whose layouts can be set directly by unwinding the matmul inputs,
 * and a new input: Q.
 *
 * Let's assume for now that Q has the same shape as X, so there is no
 * implicit numpy broadcasting. We'll consider the case of implicit
 * broadcasting in the next example.
 *
 * We cannot unwind to Q from a Tensor with a known input, as there is no
 * Poplibs API for creating the LHS input to the add operator.
 *
 * We use a slightly different heuristic in this case. It is generally a good
 * idea, when executing elementwise operations with multiple inputs such as
 * add on an IPU, to have all the inputs have the same tile mapping. This is
 * good, because the full elementwise operation can be executed without
 * needing any inter-tile communication. So in this case, a good choice is
 *
 * layout(Q) = layout(X).
 *
 * We must therefore set layout(X) before layout(Q).
 *
 * This principle, of having the same layout for all inputs to an add, can be
 * applied to any variadic elementwise operation (add, sum, mul, etc.) -- copy
 * the layout across as many variadic elementwise inputs as possible, to
 * minimise exchange.
 *
 * */

/**
 * Third example (broadcast add)
 * -----------------------------
 * In this example, we consider the case where Q is smaller than X.
 * Specifically, we're in a situation where we've determined the optimal
 * layout for X, of shape (M, N), and we need to determine the layout of Q of
 * shape, say, (N,), where
 *
 *    out1 = Q + X.
 *
 * In the previous unbroadcast example, where Q and X were the same shape, Q
 * inherited X's layout exactly, so as to minimise the cost of inter-tile
 * exchange. In the case where Q must be broadcast up, there is still a good
 * layout for Q in terms of X, and Poplibs provides an API for this:
 * #createBias
 *
 * More information on this case, and how it represented in this Graph class,
 * can be found in the comment for the class method, \a sumLike.
 *
 * */

/**
 *
 * Fourth example (call copies)
 * ----------------------------
 *
 * The principal we used in the second example, where an unbroadcast add was
 * considered, was to minimise inter-tile exchange. This same principal can be
 * used for copies, which are a special kind of binary elementwise operation.
 * Specifically, it is always beneficial to have the Tensor being copied into
 * a call operation to have the same layout as the Tensor to which it is
 * copied.
 *
 * Suppose the graph is
 *
 * Call(a, b) = matmul(a.reverse(), b.dimShuffle(perm)), and
 *
 * out = Call(A, B) + Call(C, D).
 *
 * Diagramatically,
 *
 *
 *  + - - - - Call(a,b) - - - - - +
 *  |                             |
 *  |                             |
 *  |      a           b          |
 *  |      |           |          |           A   B     C   D
 *  |      |           |          |           |   |     |   |
 *  |    reverse    dimShuffle    |           +--++     ++--+
 *  |      |           |          |              |       |
 *  |      +-----+-----+          |            Call     Call
 *  |                             |              +---+---+
 *  |            |                |                  |
 *  |          matmul             |                 out
 *  |                             |
 *  | - - - - - - - - - - - - - - +
 *
 *
 * In total there are 8 Tensors here which the user needs to set layouts
 * for:
 * - The 2 inputs to the Call operator, a and b,
 * - The 4 inputs to the main graph, A, B, C, and D, and
 * - The 2 outputs of the Call, which we have not named in the diagram.
 *
 * We will discuss the ordering in which the layouts are chosen later, but for
 * now assume that a and b have their layouts set first, using the matmul, as
 * discussed in the first example.
 *
 * None of A, B, C, and D can be unwound to directly from any Tensor with
 * known layout in the main Graph's scope. But they can all be unwound to from
 * the points at which they are copied into Call. Setting
 *
 *   layout(A) = layout(a)
 *   layout(C) = layout(a)
 *   layout(B) = layout(b)
 *   layout(D) = layout(b)
 *
 * is beneficial as it minimises the cost of the copies into Call. Note that
 * the benefit of reduced copy cost is independent of the benefit obtained by
 * having a and b have the correct layouts for a matmul, and will be modelled
 * as independent components in our cost model.
 *
 * Finally, there are the 2 outputs of the call. Note that these do not have
 * to have the same layout as the Call's matmul's output -- the user has
 * complete freedom to set their layouts. However, in this example, the best
 * layout that the user can choose is indeed that of matmul's output. Again,
 * this is to minimise the cost of the copy.
 *
 * */

/**
 *
 * Recall that Poplibs provides APIs to create Tensors with good layouts
 * for inputs to matmuls. What about the matmul output? This is not a Tensor
 * whose layout can be set by a user, as is the case for almost all Tensors
 * created by Poplibs operations. However there are some differences between
 * these "off limits" operations, which provide 3 useful categories for this
 * project.
 *
 * First type: unwindable.
 * = = = = = = = = = = = =
 * These are operations, such as as the view-changing Ops (dimShuffle, slice,
 * etc.) and unary elementwise Ops, for which the mapping of tile layouts
 * between inputs and outputs is transparent in both directions, and local.
 * That is, the tile mapping of any input (output) element can be determined
 * directly from a single output (input) element's tile mapping.
 *
 * Second type: barrier.
 * = = = = = = = = = = =
 * These are operations, such as batch normalization and max pooling, where
 * the mapping of tile layouts between inputs and outputs is completely
 * "backwards-opaque", and "forwards-non-local". Let's break these 2
 * "adjectives" down!
 *
 * Backwards-opaque: The user cannot determine the tile mapping of any
 *                   input element from any set of the output elements.
 *
 * Forwards-non-local: The tile mapping of an output element depends on the
 *                     tile mappings of all the input elements. This means it
 *                     isn't possible to know the layout of any output
 *                     elements until all of the input elements' tile mappings
 *                     are known. This imposes constraints on the order in
 *                     which layouts are set.
 *
 *
 * Third type: fixed-point.
 * = = = = = = = = = = = = =
 * This third category is quite similar to a barrier, except it is not
 * forwards-non-local. In fact, the layout of the output is completely
 * independent of any of the inputs, and so layouts can be derived from the
 * output of a fixed-point operation before any of the inputs' layouts are
 * known.
 *
 * We will model matmuls as fixed-point operations. See the discussion in
 * T32143 for why we think this is possible and a good idea.
 *
 * Not that the fixed-point type is an abstraction which is not implemented in
 * class, as it is identical to a source Tensor.
 *
 *
 * Dependencies of layouts
 * -----------------------
 *
 * Consider this example:
 *
 * out = maxpool(matmul(X, Y.reverse(dims)) + Z.
 *
 * Diagramatically,
 *
 *      X      Y        Z
 *      |      |        |
 *      |    reverse    |
 *      |      |        |
 *      +--+---+        |
 *         |            |
 *       matmul         |
 *         |            |
 *      maxpool         |
 *         |            |
 *         +--------+---+
 *                  |
 *                 add
 *                  |
 *                 out
 *
 * We've already seen how to set layout(X) and layout(Y) by unwinding from
 * a matmul. We've also seen that layout(Z) can be determined from the
 * layouts of the other inputs to add, assuming for now
 *    shape(Z) = shape(maxpool-out).
 *
 * The observation I'd like to make here is that layout(Z) can only be
 * determined after the maxpool's layout is completely set. So the order can
 * look like:
 *
 * 1)  layout(X) = layout(LHS)  using same definition of LHS as first example.
 * 2)  layout(Y) = layout(RHS.reverse(dims))
 * 3)  layout(matmul-out) : determined by calling Poplibs' matmul.
 * 4)  layout(maxpool-out) : determined by calling Poplibs' maxpool.
 * 5)  layout(Z) = layout(maxpool-out)
 *
 * We said above that we will treat matmul as a fixed-point operation, and not
 * as a barrier (see T32143). We could therefore also have the order like
 *
 * 1)  layout(matmul-out) : determined by calling Poplibs' matmul.
 * 2)  layout(maxpool-out) : determined by calling Poplibs' maxpool.
 * 3)  layout(Z) = layout(maxpool-out)
 * 4)  layout(X) = layout(LHS)
 * 5)  layout(Y) = layout(RHS.reverse(dims)).
 *
 * For implementation and compile time reasons, it's better to use the first
 * order, because the second would mean calling matmul twice: once in a
 * dummy Graph to get the layout of the output, and again later to insert
 * codelets into the final poplar Graph. Don't worry about this point for now
 * though, it is an implementation detail which belongs in a different
 * abstraction level.
 *
 * */

/**
 * Unifying the examples
 * ----------------------
 *
 * So far we have presented examples of graphs with familiar computational
 * operations in them, and described what their inputs layouts should be.
 * We'll now turn to the question of how to succinctly represent this is in a
 * custom graph class.
 *
 * We've discussed 3 ways in which layouts can be determined, and the
 * motivation for each.
 *
 * 1) from operations which have special Poplibs APIs to create their inputs,
 *    such as matmul
 * 2) from variadic elementwise operations such as add
 * 3) from copies into and out of call operations.
 *
 * Fortunately, they're all essentially the same and will be treated as such
 * in this project. We will now describe our cost model and API. We start by
 * presenting the different types of operators:
 *
 * Sinks
 * =====
 * The Tensors which need to have their layouts set by the user must appear in
 * this Graph as outputs of Sink Ops. They are called "Sinks" because they are
 * the ends of unwinding paths. All machine learning graph inputs should be
 * created with Sink operators, as all Graph inputs must have their layouts
 * (tile mappings) set.
 *
 * Sources
 * =======
 * Sources Tensors do not correspond to any Tensors in the actual compute
 * graph. They are Tensors in this Graph which represent target layouts, which
 * may or may not be copied.
 *
 * Source Tensors have layouts which are considered fixed, and are never
 * derived from other Tensors' layouts. An example is the LHS Tensor presented
 * in the first example. This is quite a subtle point: the LHS Tensor is not
 * in the compute Graph, it is just a suggested layout for the input to the
 * matmul.
 *
 * Barriers
 * ========
 * Ops for which every element of every output Tensor depends on all input
 * elements (forwards-non-local). Moreover, the layouts of inputs cannot be
 * inferred from output layouts (backwards-opaque).
 *
 * SumLike
 * =======
 * A variadic elementwise operator, such as sum, add, mul, pow, etc.
 *
 *
 * The score
 * =========
 * Any 2 Tensors in the Graph can be tied together in an \a ValuedPair. A
 * ValuedPair consists of
 *  1) 2 Tensors, of the same Shape.
 *  2) a value (a double) of attraction, describing how good it is for the 2
 *     Tensors to have the same layout.
 *
 * The score for a Solution is then
 *
 *   sum_(all valuedPair pairs p)
 *   {
 *      p.value *
 *      (number of corresponding elements p.first and p.second
 *                                       which have the same mapping)
 *   }
 *
 * This is the entire cost model. All considerations discussed previously:
 * (1) Operations like matmul with APIs for creating input layouts and (2)
 * copies into and out of calls, and (3) common layouts for all inputs to
 * variadic elementwise operator, can be captured in this model.
 *
 * More information can be the class comments.
 *
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
   * Insert a source of unwinding into this Graph. As discussed in the class
   * introduction, a Source is a Tensor whose layout cannot be derived from
   * other Tensor layouts, and is known immediately on insertion into this
   * Graph. It needn't correspond to any Tensor in a compute Graph.
   *
   * \param s    The Shape of the Source Tensor.
   *
   * \param sgid The subgraph to which the Source Tensor is added. In general
   *             this is not important, as it doesn not represent a Tensor in
   *             the computation Graph.
   * */
  TensorId source(const Shape &s, SubGraphId sgid);
  TensorId source(const Shape &s, SubGraphId, const std::string &name);

  /** Insert a Source Tensor into subgraph 0. */
  TensorId source0(const Shape &s) { return source(s, SubGraphId(0)); }

  /**
   * Insert a target of unwinding, or a Sink, into this Graph. As discussed
   * in the class introduction, inputs to a compute Graph whose layout needs
   * to be determined should be created as Sinks in this Graph.
   *
   * \param s The Shape of the Sink Tensor.
   *
   * \param sgid The subgraph into which this Sink Tensor is inserted. If the
   *             computation graph does not contain any operations which
   *             require subgraphs, such as calls, loops, and conditionals,
   *             then the SubGraph should be the same for all Sinks, and the
   *             convenience method sink0 can be used to represent this "main"
   *             scope.
   * */
  TensorId sink(const Shape &s, SubGraphId sgid);
  TensorId sink(const Shape &s, SubGraphId, const std::string &name);

  /**
   * A convenience method for inserting a Sink into subgraph 0.
   * */
  TensorId sink0(const Shape &s) { return sink(s, SubGraphId(0)); }

  /**
   * A barrier Op blocks the backwards flow of layout information. It
   * is not possible to determine the layouts of inputs based on outputs
   * (backwards-opaque), and it is only possible to determine the layouts of
   * outputs based on inputs when **all** inputs layouts are known (forwards
   * non-local).
   *
   * The layout of any element in the output is assumed to depend on all
   * elements in all input Tensors. This has implications for the order in
   * which layouts can be set.
   *
   * An example from Poplibs might be maxpool, where it is not possible for a
   * user to know the layout of the output until the layout of the entire
   * input is known.
   *
   * Note that sometimes a Barrier might not be best Op to represent an
   * operation when the output layout is independent of the input layouts. An
   * example is Poplibs' matmul (see the discussion in T32143). In particular,
   * it is advantageous to create a operation as a Source instead of a Barrier
   * when an input to the operation or any Tensor preceding it in the DAG
   * might benefit from having a layout derived from the output. Example:
   *
   *
   *       X . . . . +
   *       |         .
   *       |         .
   *    barrier   valued pair connecting X and Y.
   *       |         .
   *       v         .
   *       |         .
   *       Y . . . . +
   *
   * In the above case, it is not possible to have X and Y have the same
   * layout and obtain the associated value in the final score, because a
   * barrier assumes layout(Y) = f(layout(X)) for some unkowable function f.
   * */
  OpId barrier(const TensorIds &inputs, const Shapes &outputShapes);

  /**
   * Insert a ValuedPair. A ValuedPair signfiies that having the same layouts
   * for Tensors \a  a and \a b is beneficial, and each element which has the
   * same layout will contribute \a value to the final score of a Solution.
   * For example, if Tensors a and b are of shape (3,) and have layouts given
   * by integers, [ 0 5 4 ] and [ 0 7 4] respectively, and \a value is 7, then
   * the objective function will have a contribution of 2*7 = 14, because
   * there are 2 corresponding elements which have the same layout (at indices
   * 0 and 2).
   *
   * See Unwinding.md for for a better visual description.
   * */
  void insertValuedPair(const TensorId &a, const TensorId &b, double value);

  /**
   * Unwindable operator which subsamples a Tensor in a specified Region.
   * \sa Region
   * */
  TensorId settSample(const TensorId &, const Region &);

  /**
   * Unwindable operator which reverses a Tensor along certain dimensions.
   */
  TensorId reverse(const TensorId &, const Dimensions &);

  /**
   * Unwindable operator which reshapes a Tensor, keeping the number of
   * elements unchanged.
   * */
  TensorId reshape(const TensorId &, const Shape &);

  /**
   * Unwindable operator which squeezes all dimensions of size 1 out of the
   * input's Shape.
   * */
  TensorId squeeze(const TensorId &id);

  /**
   * Unwinable operator which permutes the dimensions a Tensor.
   * */
  TensorId dimShuffle(const TensorId &, const Permutation &);

  /**
   * Unwindable operators which concatenates multiple Tensors together along
   * a certain dimension.
   * */
  TensorId concat(const TensorIds &, uint64_t);

  /**
   * Unwindable operator which slices a Tensor in a region defined by lower
   * and upper bounds.
   * */
  TensorId slice(const TensorId &, const Lower &, const Upper &);

  /**
   * Unwindable operator which slices a Tensor in dimension \a d between \a l
   * and \a u.
   * */
  TensorId slice(const TensorId &, Dimension d, uint64_t l, uint64_t u);

  /** Unwindable operators which slices a Tensor in dimension 0, between \a l
   * and \a u
   * */
  TensorId slice(const TensorId &id, uint64_t l, uint64_t u);

  /** Unwindable operator which subsamples a Tensor along a single dimension,
   * \a d every \a s'th stride.
   * */
  TensorId subSample(const TensorId &, Stride s, Dimension d);

  /**
   * Unwindable operator which reshapes a Tensor to be of rank 1.
   * */
  TensorId flatten(const TensorId &);

  /**
   * Unwindable operator which samples a Tensor with different strides in each
   * dimension.
   * */
  TensorId subSample(const TensorId &, const Strides &);

  /**
   * Variadic elementwise operator, with attractions inserted between certain
   * input Tensors. The output can unwind through the input at \a
   * unwindableIndex. That is, the layout of the output matches the layout of
   * the input at \a unwindableIndex.
   *
   * The attraction between inputs is of value \a val. This attraction is
   * direct between inputs of the same Shape, for inputs of different Shapes
   * an intermediate sumLikeReduce Op is needed.
   *
   * Example 1 (unbroadcast add)
   *
   *   A of Shape (5,4)
   *   B of Shape (5,4)
   *   C = sumLike({A, B}, 0, 10.).
   *
   *      A       B       ValuedPairs
   *      |       |       ===============
   *      |       |       (A, B, 10.)
   *      +---+---+
   *          |           Unwinding
   *       sumLike        =========
   *          |           A <-> C
   *          C
   *
   *
   * Example 2 (broadcast add)
   *
   *   A of Shape (5,4)
   *   B of Shape (5,1)
   *   C of Shape (4)
   *   D = sumLike({A,B,C}, 0, 10.)
   *
   * In this example, the inputs do not have the same Shapes. Reduction Ops,
   * which will correspond to Poplibs addBias, are inserted to reduce to the
   * correct Shapes.
   *
   *                        A
   *                        |
   *              +---------+-----------+
   *              |         |           |
   *     sumLikeReduce      |        sumLikeReduce
   *        |               |                 |
   * Shape (5,1)      B     |      C        Shape (4)
   *  target E        |     |      |        target F
   *                  +--sumLike---+
   *                        |
   *                        D
   *
   * ValuedPairs      Unwinding
   * ===============      =========
   * (B, E, 10.)          A <-> D.
   * (C, F, 10.)
   *
   *
   *
   * */
  SumLikeOut sumLike(const TensorIds &, InIndex unwindableIndex, double val);

  /**
   * Call subgraph \a inner from subgraph \a outer, copying the Tensors in
   * copyInSources in subgraph \a outer into subgraph \a inner, before the
   * call and then copying the Tensors \a copyOutSources out at the end of the
   * call. The returned Tensors are the copies of \a copyOutSources created in
   * scope \a outer.
   *
   * This method is really just a helper method, which inserts Sink Tensors
   * for the outputs of the call, in \a outer, and inserts ValuedPairs for all
   * the copies into \a inner, and all the copies out of \a inner. The value
   * associated to all of these copies is \a value.
   *
   * */
  TensorIds call(SubGraphId outer,
                 SubGraphId inner,
                 const TensorIds &copyInSources,
                 const TensorIds &copyInDestinations,
                 const TensorIds &copyOutSources,
                 double value);

  /**
   * A call with more fine-grained control over the values of input and
   * output copies.
   * */
  TensorIds call(SubGraphId outer,
                 SubGraphId inner,
                 const TensorIds &copyInSources,
                 const TensorIds &copyInDestinations,
                 const TensorIds &copyOutSources,
                 const std::vector<double> &inCopyValues,
                 const std::vector<double> &outCopyValues);

  /**
   * Unwindable operator which maps a Tensor's layout directly to another
   * Tensor.
   * */
  TensorId identity(const TensorId &);

  /** All Sinks in this Graph */
  TensorIds sinks() const;

  /** All Sources in this Graph */
  TensorIds sources() const;

  /** All Barriers in this Graph */
  TensorIds barriers() const;

  /** All Sources and Barriers in this Graph */
  TensorIds sourcesAndBarriers() const;

  /**
   * All Tensors which are attracted to \a tId, and their value of
   * attraction.
   * */
  ValuedTensorIds valuedPartners(const TensorId &tId) const;

  /** All ValuedPairs in this Graph. */
  ValuedPairs valuedPairs() const;

  /** Append a string describing this Graph to \a ost */
  void append(std::ostream &ost) const;

  /**
   * Map Regions #inRegs, which enter the Op #opId at the input index
   * #inIndex, to the Regions in the output Tensor at #outIndex which use
   * #inRegs.
   *
   * An example:
   *   Suppose the Op #opId flattens a Tensor of Shape (4,4). Suppose that
   *   inRegs is the slice [1:3, :], described as:
   *      ....
   *      1111
   *      1111
   *      ....
   *
   *    This Regions maps to slice [4:12] in the output:
   *      ....11111111....
   *
   *   This method would therefore return this flat slice. Using the Region
   *   class constructors, this means we would we map
   *   <code>
   *        Region::fromBounds({4,4}, {1,3}, {0,4});
   *   </code>
   *   to
   *   <code>
   *        Region::fromBounds({16}, {4}, {12});
   *   </code>
   * */
  DisjointRegions outRegions(const DisjointRegions &inRegs,
                             InIndex inIndex,
                             OpId opId,
                             OutIndex outIndex) const;

  /** Map Regions in the output Tensor at #outIndex to the Regions in the
   *  input Tensor at #inIndex which they use, via the Op #opId. This is the
   *  inverse of the method \a outRegions. In particular it is guaranteed that
   *  inRegions(outRegions(inRegs, in, opId, out), in, opId, out) = inRegs.
   * */
  DisjointRegions inRegions(const DisjointRegions &out,
                            InIndex inIndex,
                            OpId opId,
                            OutIndex outIndex) const;

  /**
   * \return The TensorId of the input to Op \a opId at input index \a
   *         inIndex.
   * */
  TensorId inTensorId(InIndex inIndex, OpId opId) const;

  bool isSink(const TensorId &) const;
  bool isSource(const TensorId &) const;
  bool isBarrier(const TensorId &) const;

  SubGraphId subGraphId(const TensorId &) const;
  SubGraphIds subGraphIds(const TensorIds &) const;

  bool isUnwindable(OpId, InIndex, OutIndex) const;

  /**
   * Extend the Chain c by passing it through The Op \a opId backwards, from
   * OutIndex \a outIndex to InIndex \a inIndex.
   *
   * For example, if the Op is a Reshape, and inIndex = outIndex = 0,
   * and the reshape goes from (3,8) to (6,4), then the Chain \a c passed in
   * must end with Shape (6,4), and will will have a link added to it which
   * reshapes from (6,4) to (3,8).
   * */
  void
  extendBwd(Chain &c, OpId opId, InIndex inIndex, OutIndex outIndex) const;

  /**
   * Construct a Path from \a src to \a dst, passing through \a links.
   * */
  Path
  getPath(const TensorId &src, const Links &links, const TensorId &dst) const;

  /**
   * A Path from Tensor \a src to Tensor \a dst along an empty Chain. \a src
   * and \a dst must of course have the same Shape, as the chain is empty .
   * */
  Path fullEmpty(const TensorId &src, const TensorId &dst) const;

  /**
   * Return an extension of \a path, extended by passing its output through
   * \a opId.
   * */
  Path extendedPath(const Path &oath, InIndex, OpId opId, OutIndex) const;

  /**
   * Extend the Chain \a ch by passing its output through the Link \a l
   * */
  void extend(Chain &, const Link &l) const;

  /**
   * Extend the Chain \a ch by passing its output through all the Links in \a
   * links
   * */
  void extend(Chain &, const Links &) const;

  /**
   * Return an extension of \a chain, extended by passing its output through
   * \a opId.
   * */
  Chain extended(const Chain &chain, InIndex, OpId opId, OutIndex) const;

  /**
   * \return The name associated to subgraph \a sgid.
   * */
  std::string name(SubGraphId sgid) const;

  /**
   * Set the name of the i'th subgraph to \a n
   * */
  void setSubGraphName(SubGraphId i, const std::string &n) { sgNames[i] = n; }

  /**
   * If \a ids is empty, or not all Tensors in \a ids have the same
   * SubGraphId, then an error is thrown. Otherwise, the SubGraphId which is
   * common th all Tensors is returned.
   *
   * This method can be useful when determining what subgraph to add a Source
   * Tensor to, based on a set of Tensors which should be in the same
   *subgraph.
   **/
  SubGraphId subGraphIdFromTensorIds(const TensorIds &ids) const;

private:
  /**
   * Specialized Barrier in sumReduce.
   * */
  TensorId sumLikeReduce(const TensorId &id, const Shape &out);

  template <class T, class... Args>
  OpId
  createOpWithInputs(const TensorIds &inIds, const Shapes &out, Args... args);

  template <class T, class... Args>
  OpId createInputlessOp(SubGraphId, const Shapes &out, Args... args);

  OpId insertOp(std::unique_ptr<Op>);

  /**
   * Define what is means for this Graph to be the same as \a rhs.
   *
   * This method is called from the multiout base class, after already
   * checking that rhs is a unwind Graph. See operator== in the multiout Graph
   * class for details. This method just comares the attributes specific to
   * the unwind Graph for equivalence.
   * */
  bool multiOutTypeSpecificEqualTo(
      const common::multiout::Graph &rhs) const final {
    return sgNames == static_cast<const Graph &>(rhs).sgNames;
  }

  Op &op(OpId);
  const Op &op(OpId) const;

  /**
   * The Graph class is global, in the same way as a poplar::Graph is.
   * Subgraphs (main, calls, loops, ifs) can be captured by annotating Ops
   * with SubGraphIds. Subgraphs can have strings associated with them to help
   * debugging and to make logging clearer.
   */
  std::unordered_map<SubGraphId, std::string> sgNames;
};

std::ostream &operator<<(std::ostream &, const Graph &);

} // namespace unwind
} // namespace memory
} // namespace poprithms

#endif
