// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMMON_COMPUTE_RTENSOR_IMPL_HPP
#define POPRITHMS_COMMON_COMPUTE_RTENSOR_IMPL_HPP

#include <algorithm>
#include <iostream>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>

#include <poprithms/common/compute/graph.hpp>
#include <poprithms/common/compute/ops/binaryelementwise.hpp>
#include <poprithms/common/compute/ops/dynamic.hpp>
#include <poprithms/common/compute/ops/encode.hpp>
#include <poprithms/common/compute/ops/interdevicecopy.hpp>
#include <poprithms/common/compute/ops/matmul.hpp>
#include <poprithms/common/compute/ops/reduce.hpp>
#include <poprithms/common/compute/ops/reffrom.hpp>
#include <poprithms/common/compute/ops/unaryelementwise.hpp>
#include <poprithms/common/compute/ops/viewchange.hpp>
#include <poprithms/common/compute/rtensor.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/groupedmatmulpack.hpp>

namespace poprithms {
namespace common {
namespace compute {

template <typename T>
T RTensor<T>::constant(const poprithms::compute::host::Tensor &t) const {
  return subGraph().constant(t, deviceId());
}

template <typename T>
T RTensor<T>::constant(SubGraphId sgId, double v) const {
  return RSubGraph<T>(sgId, graph()).constant(dtype(), v, deviceId());
}

template <typename T> RSubGraph<T> RTensor<T>::subGraph() const {
  return RSubGraph<T>(subGraphId(), graph());
}

template <typename T> T RTensor<T>::constant(DType d, double v) const {
  return constant(poprithms::compute::host::Tensor::safeScalar(d, v));
}

template <typename T> T RTensor<T>::div_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Div_>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::div(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Div>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::min_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Min_>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::min(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Min>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::max_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Max_>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::max(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Max>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::pow_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Pow_>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::pow(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Pow>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::rem_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Remainder_>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::rem(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Remainder>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::copyFrom_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<CopyFrom_>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::greaterThan(const RTensor<T> &rhs) const {
  return createBooleanWithNumpyShape<GreaterThan>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::equalTo(const RTensor<T> &rhs) const {
  return createBooleanWithNumpyShape<EqualTo>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::sub_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Sub_>({id(), rhs.id()});
}
template <typename T> T RTensor<T>::sub(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Sub>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::fill_(const HostTensor &vScalar) const {
  return createUnaryWithSameInfo<Fill_>(vScalar);
}

template <typename T> T RTensor<T>::log_() const {
  return createUnaryWithSameInfo<Log_>();
}
template <typename T> T RTensor<T>::log() const {
  return createUnaryWithSameInfo<Log>();
}
template <typename T>
T RTensor<T>::encodeOneHot01_(const RTensor<T> &indices) const {
  return createTensor<EncodeOneHot01_>({id(), indices.id()}, {info()});
}

template <typename T>
T RTensor<T>::encodeOneHotOffOn_(const RTensor<T> &indices,
                                 const RTensor<T> &off,
                                 const RTensor<T> &on) const {
  return createTensor<EncodeOneHotOffOn_>(
      {id(), indices.id(), off.id(), on.id()}, {info()});
}

template <typename T> T RTensor<T>::exp_() const {
  return createUnaryWithSameInfo<Exp_>();
}
template <typename T> T RTensor<T>::exp() const {
  return createUnaryWithSameInfo<Exp>();
}

template <typename T> T RTensor<T>::sqrt_() const {
  return createUnaryWithSameInfo<Sqrt_>();
}
template <typename T> T RTensor<T>::sqrt() const {
  return createUnaryWithSameInfo<Sqrt>();
}

template <typename T> T RTensor<T>::signum_() const {
  return createUnaryWithSameInfo<Signum_>();
}
template <typename T> T RTensor<T>::signum() const {
  return createUnaryWithSameInfo<Signum>();
}

template <typename T> T RTensor<T>::inv_() const {
  return createUnaryWithSameInfo<Inv_>();
}
template <typename T> T RTensor<T>::inv() const {
  return createUnaryWithSameInfo<Inv>();
}

template <typename T> T RTensor<T>::neg_() const {
  return createUnaryWithSameInfo<Neg_>();
}
template <typename T> T RTensor<T>::neg() const {
  return createUnaryWithSameInfo<Neg>();
}

template <typename T> T RTensor<T>::cos_() const {
  return createUnaryWithSameInfo<Cos_>();
}
template <typename T> T RTensor<T>::cos() const {
  return createUnaryWithSameInfo<Cos>();
}

template <typename T> T RTensor<T>::abs_() const {
  return createUnaryWithSameInfo<Abs_>();
}
template <typename T> T RTensor<T>::abs() const {
  return createUnaryWithSameInfo<Abs>();
}

template <typename T> T RTensor<T>::sin_() const {
  return createUnaryWithSameInfo<Sin_>();
}
template <typename T> T RTensor<T>::sin() const {
  return createUnaryWithSameInfo<Sin>();
}

template <typename T> T RTensor<T>::dstInCaller(const CallEvent &ce) const {
  TensorId dst = graph().dstInCaller(id_, ce);
  return {dst, &graph()};
}

template <typename T> T RTensor<T>::srcInCaller(const CallEvent &cse) const {
  return {graph().srcInCaller(id_, cse), &graph()};
}

template <typename T> T RTensor<T>::dstInCaller(OpId call) const {
  return dstInCaller(graph().callEvent(call));
}

template <typename T>
RTensor<T>::RTensor(const TensorId &tId, Graph *m) : id_(tId), pGraph_(m) {}

template <typename T> T RTensor<T>::refTo_(SubGraphId destination) const {
  auto x = graph().template tRefFrom<RefFrom>(id(), destination);
  return {x, &graph()};
}

template <class T>
template <class TOp, class... Args>
OpId RTensor<T>::createComputeOp(const TensorIds &inIds_,
                                 const TensorInfos &outInfos,
                                 Args &&...args) const {

  if (inIds_.empty()) {
    std::ostringstream oss;
    oss << "Failed to use Tensor::createComputeOp without any inputs. "
        << "Inputs are required to determine the sub-graph of the output. "
        << "This case must be handled using Graph::createComputeOp "
        << "directly. ";
    err(oss.str());
  }
  auto sgId = graph().subGraphId(inIds_.at(0));
  return graph().template createComputeOp<TOp>(
      inIds_, sgId, outInfos, std::forward<Args>(args)...);
}

template <typename T> T RTensor<T>::reshape_(const Shape &s) const {
  return createUnaryViewChange<Reshape_>(s);
}

template <typename T> T RTensor<T>::expand_(const Shape &s) const {
  return createUnaryViewChange<Expand_>(s);
}

template <typename T> TensorInfo RTensor<T>::info() const {
  return graph().tensorInfo(id());
}

template <typename T> T RTensor<T>::dimShuffle_(const Permutation &p) const {
  return createUnaryViewChange<DimShuffle_>(p.apply(shape().get()), p);
}

template <typename T> T RTensor<T>::reverse_(const Dimensions &dims_) const {

  // Canonicalize the reverse dimensions (ascending order, repetitions
  // reduced modulo 2).
  const auto revDimsCanonical =
      shape().getCanonicalReverseIndices(dims_.get());

  return createUnaryViewChange<Reverse_>(
      shape().get(), Dimensions(std::move(revDimsCanonical)));
}

template <typename T> T RTensor<T>::reduce(const CommutativeOp cop) const {
  return reduce(Shape({}), cop);
}

template <typename T>
T RTensor<T>::reduce(const Shape &outShape, const CommutativeOp cop) const {
  auto out_ = reduce(shape().reductionDimensions(outShape), cop);
  return out_.reshape_(outShape);
}

template <typename T>
T RTensor<T>::reduce(Dimension d, const CommutativeOp cop) const {
  Dimensions dims({d});
  return reduce(dims, cop);
}

template <typename T>
T RTensor<T>::reduce(const Dimensions &d, CommutativeOp cop) const {

  if (d.empty()) {
    // we cannot just return this tensor, because there must at least be a
    // copy here.
  }
  auto outShape = shape().get();
  for (uint64_t i = 0; i < d.get().size(); ++i) {
    if (d.at(i).get() >= shape().rank_u64()) {
      err("Invalid dimension in reduceSum");
    }
    outShape[d.at(i).get()] = 1;
  }

  switch (cop) {
  case (CommutativeOp::Sum): {
    return createUnaryWithNewShape<ReduceSum>(outShape, d);
  }

  case (CommutativeOp::Min): {
    return createUnaryWithNewShape<ReduceMin>(outShape, d);
  }

  case (CommutativeOp::Max): {
    return createUnaryWithNewShape<ReduceMax>(outShape, d);
  }

  case (CommutativeOp::Product): {
    return createUnaryWithNewShape<ReduceProduct>(outShape, d);
  }
  default:
    err("Unrecognised reduction type");
  }
}

template <typename T> T RTensor<T>::mul_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Mul_>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::mul(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Mul>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::add_(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Add_>({id(), rhs.id()});
}

template <typename T> T RTensor<T>::add(const RTensor<T> &rhs) const {
  return createWithNumpyShape<Add>({id(), rhs.id()});
}

template <typename T>
T RTensor<T>::padWithBroadcastConstZero_(const Lower &l,
                                         const Upper &u) const {

  auto z = constant(0.);

  // progressively pad the tensor, one dimension at a time. The padding on all
  // edges is all an alias of the constant 'z' above.

  auto t = T(id(), &graph());
  for (uint64_t d = 0; d < rank_u64(); ++d) {

    // concatate the tensors (lower_padding, t, upper_padding). If the padding
    // is zero, then leave it off.
    std::vector<T> toConcat;
    if (l.at(d) > 0) {
      toConcat.push_back(z.expand_(t.shape().resizeSingleDim(l.at(d), d)));
    }

    toConcat.push_back(t);

    if (u.at(d) > 0) {
      toConcat.push_back(z.expand_(t.shape().resizeSingleDim(u.at(d), d)));
    }

    t = concat_(toConcat, d);
  }

  return t;
}

template <typename T>
T RTensor<T>::slice_(const Lower &l, const Upper &u) const {

  const auto outShape = shape().slice(l, u);

  // Check for a slice which doesn't slice anything out. The second condition
  // here is to confirm that the bounds are valid (lower is 0s).
  if (outShape == shape() && Shape(u) == shape()) {
    return {id(), &graph()};
  }

  return createUnaryWithNewShape<Slice_>(outShape, l, u);
}

template <typename T>
T RTensor<T>::slice_(const Dimensions &dims,
                     const std::vector<uint64_t> &starts_,
                     const std::vector<uint64_t> &ends_) const {
  auto lu = shape().getFullSliceBounds(dims, starts_, ends_);
  return slice_(lu.first, lu.second);
}

template <typename T>
T RTensor<T>::slice(const Dimensions &a,
                    const std::vector<uint64_t> &b,
                    const std::vector<uint64_t> &c) const {
  return slice_(a, b, c).copy();
}

template <typename T> T RTensor<T>::copy(DeviceId target) const {

  const auto targetType = graph().device(target).deviceType();
  if (targetType != deviceType()) {
    std::ostringstream oss;
    oss << "Tensor::copy(target=" << target
        << ") is invalid, as this tensor, " << id()
        << " has a different device type. "
        << "This method can only copy Ipu->Ipu. " << deviceType() << "->"
        << targetType;
    err(oss.str());
  }

  auto targetTensor = variable(target);
  return targetTensor.copyFrom_(*this);
}

template <typename T>
T RTensor<T>::slice(const Lower &l, const Upper &u) const {
  return slice_(l, u).copy();
}

template <typename T>
T RTensor<T>::slice(Dimension d, int64_t l, int64_t u) const {
  return slice_(d, l, u).copy();
}

template <typename T>
T RTensor<T>::slice_(Dimension d, int64_t l, int64_t u) const {
  if (l < 0 || u < 0) {
    std::ostringstream oss;
    oss << "Invalid call, Tensor::slice_ (Dimension = " << d.get()
        << ", l = " << l << ", u = " << u << "). "
        << "The lower (l) and upper (u) bounds must both be non-negative. ";
    err(oss.str());
  }

  const auto fullSliceBounds = shape().getFullSliceBounds(
      d, static_cast<uint64_t>(l), static_cast<uint64_t>(u));

  return slice_(std::get<0>(fullSliceBounds), std::get<1>(fullSliceBounds));
}

template <typename T> T RTensor<T>::reverse(const Dimensions &ds) const {
  return reverse_(ds).copy();
}

template <typename T> T RTensor<T>::reverse(uint64_t d) const {
  return reverse_(d).copy();
}
template <typename T> T RTensor<T>::to(DType t) const {
  return createTensor<Cast>({id()}, {info().withDType(t)});
}

template <typename T> T RTensor<T>::dimShuffleFinalTwo() const {
  return dimShuffle(Permutation::reverseFinalTwo(rank_u64()));
}

template <typename T>
T RTensor<T>::dimRoll(uint64_t from, uint64_t to) const {
  return dimShuffle(Permutation::dimRoll(rank_u64(), {from, to}));
}

template <typename T>
T RTensor<T>::dimRoll_(uint64_t from, uint64_t to) const {
  return dimShuffle_(Permutation::dimRoll(rank_u64(), {from, to}));
}

template <typename T> T RTensor<T>::dimShuffleFinalTwo_() const {
  return dimShuffle_(Permutation::reverseFinalTwo(rank_u64()));
}

template <typename T>
T RTensor<T>::squeeze(const std::vector<uint64_t> &dims) const {
  return reshape(shape().squeeze(dims));
}

template <typename T>
T RTensor<T>::squeeze_(const std::vector<uint64_t> &dims) const {
  return reshape_(shape().squeeze(dims));
}

template <typename T> T RTensor<T>::variable(const Shape &s0) const {
  return subGraph().variable(dtype(), s0, deviceId());
}

template <typename T>
T RTensor<T>::variable(DeviceId dId, const Shape &s0) const {
  return subGraph().variable(dtype(), s0, dId);
}

template <typename T>
T RTensor<T>::variable(DeviceId dId, SubGraphId sgId) const {
  return RSubGraph<T>(sgId, graph()).variable(dtype(), shape(), dId);
}

template <typename T> T RTensor<T>::variable(DType t) const {
  return subGraph().variable(t, shape(), deviceId());
}

template <typename T> T RTensor<T>::variable(DType t, const Shape &s) const {
  return subGraph().variable(t, s, deviceId());
}

template <typename T> T RTensor<T>::variable(SubGraphId sgId) const {
  return RSubGraph<T>(sgId, graph()).variable(dtype(), shape(), deviceId());
}

template <typename T> T RTensor<T>::variable() const {
  return subGraph().variable(dtype(), shape(), deviceId());
}

template <typename T> T RTensor<T>::variable(DeviceId did) const {
  return subGraph().variable(dtype(), shape(), did);
}

template <typename T>
T RTensor<T>::concat_(const std::vector<T> &ts, uint64_t axis) {

  if (ts.size() == 0) {
    err("cannot concatenate empty vector of Tensors");
  }

  // If there is just 1 tensor being concatenated, return it.
  if (ts.size() == 1) {
    return T(ts[0].id(), &ts[0].graph());
  }

  auto &m  = ts[0].graph();
  auto ids = TSlickConverter::getIds(ts);
  auto out = ts[0].template createTensor<Concat_>(
      ids,
      {m.tensorInfo(ts[0]).withShape(Shape::concat(m.shapes(ids), axis))},
      axis);

  return out;
}

template <typename T>
T RTensor<T>::updateFromHost_(
    const RTensor<T> &source,
    const CopyBetweenHostAndIpuOptions &copyOptions) const {
  return createTensor<CopyFromHostToIpu_>(
      {source.id(), id()}, {{info()}}, copyOptions);
}

template <typename T>
T RTensor<T>::updateFromIpu_(
    const RTensor<T> &source,
    const CopyBetweenHostAndIpuOptions &copyOptions) const {
  return createTensor<CopyFromIpuToHost_>(
      {source.id(), id()}, {{info()}}, copyOptions);
}

template <typename T>
T RTensor<T>::hostToIpu(
    DeviceId ipuDestination,
    const CopyBetweenHostAndIpuOptions &copyOptions) const {
  if (rank_u64() < 2) {
    err("Source of host->ipu copy must be at least rank 2");
  }

  // Create an ipu tensor:
  const auto target = variable(ipuDestination, shape().fromDim(2));

  // Copy to ipu tensor:
  return target.updateFromHost_(*this, copyOptions);
}

template <typename T>
T RTensor<T>::ipuToHost(
    CircularBufferCount circularBufferCount,
    const CopyBetweenHostAndIpuOptions &copyOptions) const {

  // Create a host tensor:
  const auto target = variable(graph().host(),
                               shape()
                                   .prepend(graph().replicationFactor_u64())
                                   .prepend(circularBufferCount.get()));

  // Copy to host tensor:
  return target.updateFromIpu_(*this, copyOptions);
}

namespace {

template <typename T> class MatmulTensorMoldingHelper {
public:
  static Shape shape(const T &t) { return t.shape(); }
  static int64_t dim(const T &t, uint64_t d) { return t.dim(d); }
  static T unsqueeze(const T &t, uint64_t d) { return t.unsqueeze_(d); }
  static T reshape(const T &t, const Shape &s) { return t.reshape_(s); }
  static T expand(const T &t, const Shape &s) { return t.expand_(s); }
};

template <typename T> T getT(RTensor<T> t) { return T(t.id(), &t.graph()); }
} // namespace

template <typename T>
T RTensor<T>::matmul(const RTensor<T> &rhs,
                     DType outType,
                     const MatMulOptions &matMulOptions) const {

  // reshapes and expands.
  auto matMulPack =
      poprithms::ndarray::GroupedMatMulPack<MatmulTensorMoldingHelper<T>, T>(
          getT(*this), getT(rhs));

  // output is rank-3.
  const Shape outShape{
      matMulPack.nGroups(), matMulPack.M_i64(), matMulPack.N_i64()};

  const TensorInfo outInfo(outShape, deviceId(), outType);

  const auto out3d = createTensor<MatMul>(
      {matMulPack.lhs3d(), matMulPack.rhs3d()}, {outInfo}, matMulOptions);

  // reshape to correct grouped matmul output shape.
  return out3d.reshape_(matMulPack.outShape());
}

template <typename T>
std::vector<T> RTensor<T>::tensors(const TensorIds &ids, Graph &g) {
  std::vector<T> ts;
  ts.reserve(ids.size());
  for (const auto &id : ids) {
    ts.push_back({id, &g});
  }
  return ts;
}

template <typename T>
TensorIds RTensor<T>::tensorIds(const std::vector<T> &ts) {
  TensorIds tIds;
  tIds.reserve(ts.size());
  for (auto t : ts) {
    tIds.push_back(t.id());
  }
  return tIds;
}

template <typename T>
T RTensor<T>::dynamicMultiSlice(const RTensor<T> &offset,
                                const Dimensions &dims,
                                const Shape &sizes) const {

  auto slicedShape = DynamicMultiWithDimensions_::getSlicedShape(
      offset.shape(), shape(), dims, sizes);

  const auto outInfo = info().withShape(slicedShape);

  auto slice = variable(slicedShape);

  return slice.dynamicMultiSlice_(*this, offset, dims);
}

template <typename T>
T RTensor<T>::dynamicMultiSlice_(const RTensor<T> &sliceable,
                                 const RTensor<T> &offset,
                                 const Dimensions &dims) const {

  return createTensor<DynamicMultiSlice_>(
      {sliceable, *this, offset}, {info()}, dims);
}

template <typename T>
T RTensor<T>::dynamicMultiUpdate_(const RTensor<T> &update,
                                  const RTensor<T> &offset,
                                  const Dimensions &dims) const {
  return createTensor<DynamicMultiUpdate_>(
      {id(), update.id(), offset.id()}, {info()}, dims);
}

template <typename T>
T RTensor<T>::dynamicMultiUpdateMax_(const RTensor<T> &source,
                                     const RTensor<T> &offset) const {
  return createTensor<DynamicMultiUpdateMax_>(
      {id(), source.id(), offset.id()}, {info()});
}

template <typename T> T RTensor<T>::setToLowest_() const {
  return fill_(HostTensor::lowestScalar(dtype()));
}
template <typename T> T RTensor<T>::reduceSumAcrossReplicas_() const {
  return createUnaryWithSameInfo<ReduceSumAcrossReplicas_>();
}
template <typename T> T RTensor<T>::reduceSumAcrossReplicas() const {
  return createUnaryWithSameInfo<ReduceSumAcrossReplicas>();
}

template <typename T>
T RTensor<T>::update_(const RTensor<T> &update,
                      const Offsets &offsets,
                      const Dimensions &dims) const {
  auto starts = offsets.get();
  auto ends_  = update.shape().addToDims(offsets.get_i64()).get_u64();
  auto s_     = slice_(dims, starts, ends_);
  s_.copyFrom_(update);
  return T(id(), &graph());
}

template <typename T>
T RTensor<T>::update_(const RTensor<T> &update,
                      const Offsets &offsets) const {

  if (offsets.size() != rank_u64()) {
    std::ostringstream oss;
    oss << "Expected full rank update in "
        << " Tensor::update_ without explicit Dimensions. ";
    oss << "This Tensor is " << id() << " and the 'update' Tensor is "
        << update.id() << ". offsets is of size " << offsets.size();
    err(oss.str());
  }
  std::vector<uint64_t> dims_(rank_u64());
  std::iota(dims_.begin(), dims_.end(), 0ULL);
  return update_(update, offsets, Dimensions(dims_));
}

template <typename T>
T RTensor<T>::softmax(uint64_t d, StableSoftmax stable) const {

  T t(id(), &graph());

  // subtract the maximum in each reduction plane.
  if (stable == StableSoftmax::Yes) {
    t = t - t.reduceMax(Dimension(d));
  }
  t = t.exp();
  return t / t.reduceSum(Dimension(d));
}

template <typename T>
typename RTensor<T>::NllGrad RTensor<T>::nllGrad(const RTensor<T> &labels,
                                                 StableSoftmax ss) const {

  if (rank_u64() != 2) {
    std::ostringstream oss;
    oss << "Can only call nnlGrad on rank-2 tensors (N, C). "
        << "This tensor has shape " << shape();
    err(oss.str());
  }

  if (labels.rank_u64() != 1 || labels.dim(0) != dim(0)) {
    std::ostringstream oss;
    oss << "Expected labels to have shape (" << dim(0)
        << ") for this tensor of shape " << shape()
        << ". But labels has shape " << labels.shape() << ".";
    err(oss.str());
  }

  // where N = dim(0) and C = dim(1):

  // shape: (N, 1)
  auto probs = softmax(1, ss);

  // shape: (N,C)
  auto encoded = variable().encodeOneHot01_(labels);

  // shape: (N,C)
  auto dIn = probs - encoded;

  // shape: (N,)
  auto logProbs = (probs * encoded).reduceSum(Dimension(1)).log();

  // shape: ()
  auto loss = logProbs.reduceSum().neg().squeeze_();

  return NllGrad(loss, dIn);
}

template <typename T> void RTensor<T>::append(std::ostream &os) const {
  os << "id=" << id() << ",subGraphId=" << subGraphId()
     << ",shape=" << shape() << ",dtype=" << dtype();
}

template <typename T>
T RTensor<T>::remoteToIpu(const RTensor<T> &indices) const {

  const Shape sliceShape =
      CopyBetweenRemoteAndIpu_::shapeOfIpuSlice(indices.shape(), shape());

  const auto sliceInfo = TensorInfo(sliceShape, indices.deviceId(), dtype());

  const auto slice = subGraph().variable(sliceInfo);

  return slice.updateIpuFromRemote_(T(id(), &graph()), indices);
}

template <typename T>
T RTensor<T>::updateIpuFromRemote_(const RTensor<T> &remoteTensor,
                                   const RTensor<T> &indices) const {

  // This is an ipu tensor. Output is an alias of it.
  return createTensor<CopyFromRemoteToIpu_>(
      TensorIds({remoteTensor.id(), id(), indices.id()}), {info()});
}

template <typename T>
T RTensor<T>::updateRemoteFromIpu_(const RTensor<T> &ipuTensor,
                                   const RTensor<T> &indices) const {

  // This is a remote tensor. Output is an alias of it.
  return createTensor<CopyFromIpuToRemote_>(
      TensorIds({id(), ipuTensor.id(), indices.id()}), {info()});
}

template <typename T>
T RTensor<T>::ipuToRemote(const RTensor<T> &indices,
                          uint64_t nRepeats,
                          const RemoteOptions &opts) const {
  const auto remoteShape =
      CopyBetweenRemoteAndIpu_::shapeOfRemoteSliceable(shape(), nRepeats);

  const auto remote =
      subGraph().remoteVariable(dtype(), remoteShape, deviceId(), opts);
  return remote.updateRemoteFromIpu_({id(), &graph()}, indices);
}

template <typename T>
T RTensor<T>::ipuToRemote(const RemoteOptions &opts) const {
  if (rank_u64() != 2 || dim(0) != 1) {
    std::ostringstream oss;
    oss << "Expected rank-2 tensor with dim(0)=1. "
        << "But this ipu tensor has shape " << shape();
    err(oss.str());
  }
  auto indices = constant(DType::Unsigned32, 0).reshape_({1});
  return ipuToRemote(indices, 1, opts);
}

template <typename T> T RTensor<T>::remoteToIpu() const {
  if (rank_u64() != 2 || dim(0) != 1) {
    std::ostringstream oss;
    oss << "Expected rank-2 tensor with dim(0)=1. "
        << "But this remote tensor has shape " << shape();
    err(oss.str());
  }
  auto ipu_ = graph().remote(deviceId()).ipu();
  auto indices =
      subGraph().constant(DType::Unsigned32, 0, ipu_).reshape_({1});
  return remoteToIpu(indices);
}

} // namespace compute
} // namespace common
} // namespace poprithms

#endif
