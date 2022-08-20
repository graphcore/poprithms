// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_PROGRAM_PIPELINE_IMUTATOR_HPP
#define POPRITHMS_PROGRAM_PIPELINE_IMUTATOR_HPP

#include <map>
#include <unordered_set>

#include <poprithms/common/multiout/consumptionid.hpp>
#include <poprithms/common/multiout/ioindices.hpp>
#include <poprithms/common/multiout/traversal.hpp>
#include <poprithms/common/schedulable/subgraphid.hpp>
#include <poprithms/ndarray/deviceid.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/ndarray/tensorinfo.hpp>

namespace poprithms {
namespace program {
namespace pipeline {

using PipelineStage  = poprithms::util::TypedInteger<'P', int>;
using PipelineStages = std::vector<PipelineStage>;

using poprithms::common::multiout::ConsumptionIds;
using poprithms::common::multiout::OpId;
using poprithms::common::multiout::OpIds;
using poprithms::common::multiout::OutIndex;
using poprithms::common::multiout::OutIndices;
using poprithms::common::multiout::TensorId;
using poprithms::common::multiout::TensorIds;
using poprithms::common::schedulable::SubGraphId;
using poprithms::common::schedulable::SubGraphIds;
using poprithms::ndarray::DeviceId;
using poprithms::ndarray::DeviceIds;
using poprithms::ndarray::DType;
using poprithms::ndarray::Shape;
using poprithms::ndarray::Shapes;

/**
 * Interface to an object which creates sub-graphs and ops, as needed for the
 * pipelining algorithm implemented in this project.
 *
 * The most 'opinionated' method in this class is probably the method
 * #refFrom_. This method allows a tensor to be used globally. An alternative
 * approach (see T67161) would use a more standard approach which avoids
 * global access to tensors.
 * */
class IMutator {

public:
  virtual ~IMutator() = default;

  /**
   * Call the sub-graph #callee from the sub-graph #caller. There are no
   * inputs and no outputs, like for a poplar call program.
   * */
  virtual OpId call(SubGraphId caller, SubGraphId callee) const = 0;

  /**
   * Call the sub-graph #callee from the sub-graph #caller, a total of
   * #tripCount times. There are no inputs or outputs, along the lines of a
   * poplar repeat program.
   * */
  virtual OpId
  repeat(SubGraphId caller, SubGraphId callee, uint64_t tripCount) const = 0;

  /**
   * Create a sub-graph with name #n.
   * */
  virtual SubGraphId createSubGraph(const std::string &n) const = 0;

  /**
   * Create a sub-graph with name #n, where ops are always scheduled in the
   * order they are created in the sub-graph. This is like a poplar Sequence.
   * */
  virtual SubGraphId createInOrderSubGraph(const std::string &n) const = 0;

  /**
   * Create a clone of the op #opId. The clone will be identiical to #opId
   * except that it has inputs #ins in sub-graph #sg and the outputs are on
   * the devices #outDevIds.
   * */
  virtual OpId clone(OpId opId,
                     const TensorIds &ins,
                     SubGraphId sg,
                     const DeviceIds &outDevIds) const = 0;

  /**
   * Create a reference to the tensor #tId in the sub-graph #sg.
   *
   * If all tensors are global (as is the case for poplar Tensors) then this
   * method can just return the input tensor id #tId.
   * */
  virtual TensorId refFrom_(const TensorId &tId, SubGraphId sg) const = 0;

  /**
   * Copy the tensor #tId to the device #devId.
   * */
  virtual TensorId copy(const TensorId &tId, DeviceId devId) const = 0;

  /**
   * Copy the tensor #src to the tensor #dst.
   * */
  virtual TensorId copy_(const TensorId &src, const TensorId &dst) const = 0;

  /**
   * Create a variable of type #dt and shape #s in the sub-graph #sgId. The
   * variable will be on device #devId.
   * */
  virtual TensorId variable(DType dt,
                            const Shape &s,
                            DeviceId devId,
                            SubGraphId sgId) const = 0;

  /**
   * Create a variable which is like #t0 is all respects, except that it has
   * shape #s.
   * */
  virtual TensorId variableLike(const TensorId &t0, const Shape &s) const = 0;

  /**
   * Create a variable which is like #t0 is all respects, except that it on
   * device #dId and in sub-graph #sgId.
   * */
  virtual TensorId
  variableLike(const TensorId &t0, DeviceId dId, SubGraphId sgId) const = 0;

  /**
   * Take a slice of #t0 at index #index in dimension 0. If this tensor has
   * shape (S0, S1,... SZ) then the returned tensor has shape (S1,... SZ).
   * */
  virtual TensorId dynamicAt(const TensorId &t0,
                             const TensorId &index) const = 0;

  /**
   * Update a slice of #sliceable with values #slice. The slice is at index
   * #index in dimension 0. #slice has a rank that is 1 lower than that of
   * #sliceable.
   * */
  virtual TensorId updateAt_(const TensorId &sliceable,
                             const TensorId &slice,
                             const TensorId &index) const = 0;

  /**
   * Add the value #v to the tensor #tId.
   * */
  virtual TensorId add(const TensorId &tId, uint64_t v) const = 0;

  /**
   * Subtract the value #v from the tensor #tId.
   * */
  virtual TensorId sub(const TensorId &tId, uint64_t v) const = 0;

  /**
   * Add the value #v to the tensor #tId, inplace.
   * */
  virtual TensorId add_(const TensorId &tId, uint64_t v) const = 0;

  /**
   * Set the value of the tensor #tId to 0.
   * */
  virtual TensorId zero_(const TensorId &tId) const = 0;

  /**
   * Return #tId modulus #mod.
   * */
  virtual TensorId modulo(const TensorId &tId, uint64_t mod) const = 0;

  /**
   * Initialize the accumulator #accl. If the accumulation is done by
   * summation, for example, this should set the value of #accl to 0.
   *
   * The tensor #unpipelined is the tensor in the unpipelined graph which was
   * marked for accumulation. The type of accumulation can be determined by
   * storing the objective in this mutator.
   * */
  virtual TensorId initAccumulator_(const TensorId &unpipelined,
                                    const TensorId &accl) const = 0;

  /**
   * Accumulate the tensor #toUpdate by combining #partial into it. This might
   * be an add inplace, but any operation is permitted.
   *
   * The tensor #iteration is a scalar fixed-point tensor which is the number
   * of accumulations performed so far. This can be useful for example if the
   * accumulation is running mean:
   *
   * toUpdate <- iteration/(iteration+1)*toUpdate +
   *                                  + partial/(iteration+1).
   *
   * The tensor #unpipelined is the tensor in the unpipelined graph which was
   * marked for accumulation. The type of accumulation can be determined by
   * storing the objective in this mutator.
   * */
  virtual TensorId accumulate(const TensorId &unpipelined,
                              const TensorId &partial,
                              const TensorId &toUpdate,
                              const TensorId &iteration) const = 0;

  virtual void setName(OpId, const std::string &) const = 0;
  virtual std::string name(OpId) const                  = 0;

  OpId
  call(SubGraphId caller, SubGraphId callee, const std::string &n) const {
    auto x = call(caller, callee);
    setName(x, n);
    return x;
  }

private:
  virtual void noWeakVTables();
};

} // namespace pipeline
} // namespace program
} // namespace poprithms

#endif
