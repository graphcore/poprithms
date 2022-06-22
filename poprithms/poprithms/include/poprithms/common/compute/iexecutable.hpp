// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_IEXECUTABLE_HPP
#define POPRITHMS_COMMON_COMPUTE_IEXECUTABLE_HPP

#include <poprithms/common/compute/graph.hpp>

namespace poprithms {
namespace common {
namespace compute {

/**
 * Interface for a class which can run/execute a Graph, on devices. This class
 * stores the constant Graph which it is compiled/lowered from. It has public
 * methods for setting host tensor values, running sub-graphs of the Graph,
 * and getting host tensor values.
 * */
class IExecutable {

public:
  IExecutable() = delete;
  IExecutable(Graph &&);

  IExecutable(IExecutable &&)      = default;
  IExecutable(const IExecutable &) = default;

  virtual ~IExecutable();

  /**
   * Return the host tensor of #tId. There is no copying of tensor data here:
   * HostTensors are wrappers around shared pointers, and this method just
   * creates a copy of that underlying shared pointer.
   * */
  HostTensor getHostValue(const TensorId &tId) const;

  /**
   * Set the value of the host tensor #tId to #v. There IS a copying of data
   * here, the values in #v are copied to the tensor stored for #tId.
   * */
  void setHostValue(const TensorId &id, const HostTensor &v) {
    getHostValue(id).update_(v);
  }

  /**
   * Set the values of the host tensor #tId to #vs.
   * */
  template <typename T>
  void setHostValue(const TensorId &tId, std::vector<T> &&vs) {
    setHostValue(tId, HostTensor::tensor<T>(shape(tId), std::move(vs)));
  }

  /**
   * Set the values of the host tensor #tId to #vs.
   * */
  template <typename T>
  void setHostValue(const TensorId &tId, const std::vector<T> &vs) {
    setHostValue<T>(tId, std::vector<T>(vs));
  }

  /**
   * Set the values of multiple host tensors. #m is a map whose keys are
   * tensor ids, and whose values are host tensors.
   * */
  template <typename Map> void setHostValues(Map &&m) {
    for (const auto &x : m) {
      setHostValue(x.first, x.second);
    }
  }

  /**
   * Host tensors can either manage the lifetime of their underlying data, or
   * they can be wrappers around raw pointers. This method is for tensors
   * which are wrappers around raw pointers. It updates the pointer being
   * wrapped by #tId to #v -- no numerical data is copied.
   *
   * \sa Graph::setUserManagedHost.
   * */
  template <typename T>
  void setHostValuePointer(const TensorId &tId, T *v) const {
    getHostValue(tId).updateRef<T>(v);
  }

  /**
   * Run the sub-graph #sgId.
   * */
  void run(SubGraphId sgId);

  /**
   * The graph which this IExecutable was created with.
   * */
  const Graph &graph() const { return graph_; }

  /**
   * A few convenience methods which forward directly to Graph methods.
   * */
  uint64_t nOps() const { return graph().nOps(); }
  uint64_t nOutTensors(OpId id) const { return graph().nOutTensors(id); }
  Shape shape(const TensorId &tId) const { return graph().shape(tId); }

  /**
   * Set the remote values of all replicas of the remote tensor, #rId.
   *
   * \param tVals a vector of rank-2 tensors, one for each of the replicas of
   *              the remote tensor #rId.
   * */
  void setRemoteValue(const TensorId &rId, const HostTensors &tVals);

  /**
   * Set the value of the #r'th replica of the remote tensor #rId to value
   * #tVal. #tVal must be a rank-2 tensor.
   * */
  void
  setRemoteValue(const TensorId &tId, const HostTensor &tVal, uint64_t r);

  /**
   * Get the value of the #r'th replica of the remote tensor, #rId. This call
   * will copy the data of this replica to a new host tensor.
   * */
  HostTensor getRemoteValue(const TensorId &rId, uint64_t r) const;

private:
  // Get the host tensor (shared pointer) of #hId.
  virtual HostTensor
  executableSpecificGetHostValue(const TensorId &hId) const = 0;

  // Get the value (copy) of replica #replica of remote tensor, #tId.
  virtual HostTensor
  executableSpecificGetRemoteValue(const TensorId &tId,
                                   uint64_t replica) const = 0;

  virtual void executableSpecificSetRemoteValue(const TensorId &,
                                                const HostTensor &,
                                                uint64_t replica) = 0;

private:
  virtual void executableSpecificRun(SubGraphId) = 0;

  // The Graph that this executable is created from. It is const, to ensure
  // that it is not be modified -- all optimizations on the Graph must be run
  // before constructing an executable.
  const Graph graph_;
};
} // namespace compute
} // namespace common
} // namespace poprithms

#endif
