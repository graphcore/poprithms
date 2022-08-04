// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
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
#include <poprithms/common/compute/tensor.hpp>
#include <poprithms/error/error.hpp>
#include <poprithms/ndarray/groupedmatmulpack.hpp>

namespace poprithms {
namespace common {
namespace compute {

const Op &Tensor::op(OpId opId) const { return graph().computeOp(opId); }

Tensor::Tensor(const TensorId &tId, Graph *pGraph)
    : id_(tId), pGraph_(pGraph) {}

Tensor Tensor::constant(const poprithms::compute::host::Tensor &t) const {
  return SubGraph(*this).constant(t, deviceId());
}

Tensor Tensor::constant(SubGraphId sgId, double v) const {
  return SubGraph(sgId, graph()).constant(dtype(), v, deviceId());
}

Tensor Tensor::constant(DType d, double v) const {
  return constant(poprithms::compute::host::Tensor::safeScalar(d, v));
}

Tensor Tensor::div_(const Tensor &rhs) const {
  return createWithNumpyShape<Div_>({id(), rhs.id()});
}

Tensor Tensor::div(const Tensor &rhs) const {
  return createWithNumpyShape<Div>({id(), rhs.id()});
}

Tensor Tensor::min_(const Tensor &rhs) const {
  return createWithNumpyShape<Min_>({id(), rhs.id()});
}

Tensor Tensor::min(const Tensor &rhs) const {
  return createWithNumpyShape<Min>({id(), rhs.id()});
}

Tensor Tensor::max_(const Tensor &rhs) const {
  return createWithNumpyShape<Max_>({id(), rhs.id()});
}

Tensor Tensor::max(const Tensor &rhs) const {
  return createWithNumpyShape<Max>({id(), rhs.id()});
}

Tensor Tensor::pow_(const Tensor &rhs) const {
  return createWithNumpyShape<Pow_>({id(), rhs.id()});
}

Tensor Tensor::pow(const Tensor &rhs) const {
  return createWithNumpyShape<Pow>({id(), rhs.id()});
}

Tensor Tensor::rem_(const Tensor &rhs) const {
  return createWithNumpyShape<Remainder_>({id(), rhs.id()});
}

Tensor Tensor::rem(const Tensor &rhs) const {
  return createWithNumpyShape<Remainder>({id(), rhs.id()});
}

Tensor Tensor::copyFrom_(const Tensor &rhs) const {
  return createWithNumpyShape<CopyFrom_>({id(), rhs.id()});
}

Tensor Tensor::greaterThan(const Tensor &rhs) const {
  return createBooleanWithNumpyShape<GreaterThan>({id(), rhs.id()});
}

Tensor Tensor::equalTo(const Tensor &rhs) const {
  return createBooleanWithNumpyShape<EqualTo>({id(), rhs.id()});
}

Tensor Tensor::sub_(const Tensor &rhs) const {
  return createWithNumpyShape<Sub_>({id(), rhs.id()});
}
Tensor Tensor::sub(const Tensor &rhs) const {
  return createWithNumpyShape<Sub>({id(), rhs.id()});
}

Tensor Tensor::fill_(const HostTensor &vScalar) const {
  return createUnaryWithSameInfo<Fill_>(vScalar);
}

Tensor Tensor::log_() const { return createUnaryWithSameInfo<Log_>(); }
Tensor Tensor::log() const { return createUnaryWithSameInfo<Log>(); }

Tensor Tensor::encodeOneHot01_(const Tensor &indices) const {
  return createTensor<EncodeOneHot01_>({id(), indices.id()}, {info()});
}

Tensor Tensor::encodeOneHotOffOn_(const Tensor &indices,
                                  const Tensor &off,
                                  const Tensor &on) const {
  return createTensor<EncodeOneHotOffOn_>(
      {id(), indices.id(), off.id(), on.id()}, {info()});
}

Tensor Tensor::exp_() const { return createUnaryWithSameInfo<Exp_>(); }
Tensor Tensor::exp() const { return createUnaryWithSameInfo<Exp>(); }

Tensor Tensor::sqrt_() const { return createUnaryWithSameInfo<Sqrt_>(); }
Tensor Tensor::sqrt() const { return createUnaryWithSameInfo<Sqrt>(); }

Tensor Tensor::signum_() const { return createUnaryWithSameInfo<Signum_>(); }
Tensor Tensor::signum() const { return createUnaryWithSameInfo<Signum>(); }

Tensor Tensor::inv_() const { return createUnaryWithSameInfo<Inv_>(); }
Tensor Tensor::inv() const { return createUnaryWithSameInfo<Inv>(); }

Tensor Tensor::neg_() const { return createUnaryWithSameInfo<Neg_>(); }
Tensor Tensor::neg() const { return createUnaryWithSameInfo<Neg>(); }

Tensor Tensor::cos_() const { return createUnaryWithSameInfo<Cos_>(); }
Tensor Tensor::cos() const { return createUnaryWithSameInfo<Cos>(); }

Tensor Tensor::abs_() const { return createUnaryWithSameInfo<Abs_>(); }
Tensor Tensor::abs() const { return createUnaryWithSameInfo<Abs>(); }

Tensor Tensor::sin_() const { return createUnaryWithSameInfo<Sin_>(); }
Tensor Tensor::sin() const { return createUnaryWithSameInfo<Sin>(); }

Tensor Tensor::dstInCaller(const CallEvent &ce) const {
  TensorId dst = graph().dstInCaller(id_, ce);
  return {dst, &graph()};
}

Tensor Tensor::srcInCaller(const CallEvent &cse) const {
  return {graph().srcInCaller(id_, cse), &graph()};
}

Tensor Tensor::dstInCaller(OpId call) const {
  return dstInCaller(graph().callEvent(call));
}

Tensor Tensor::refTo_(SubGraphId destination) const {
  auto x = graph().template tRefFrom<RefFrom>(id(), destination);
  return {x, &graph()};
}

template <class TOp, class... Args>
OpId Tensor::createComputeOp(const TensorIds &inIds_,
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

Tensor Tensor::reshape_(const Shape &s) const {
  return createUnaryViewChange<Reshape_>(s);
}

Tensor Tensor::expand_(const Shape &s) const {
  return createUnaryViewChange<Expand_>(s);
}

TensorInfo Tensor::info() const { return graph().tensorInfo(id()); }

Tensor Tensor::dimShuffle_(const Permutation &p) const {
  return createUnaryViewChange<DimShuffle_>(p.apply(shape().get()), p);
}

Tensor Tensor::reverse_(const Dimensions &dims_) const {

  // Canonicalize the reverse dimensions (ascending order, repetitions
  // reduced modulo 2).
  const auto revDimsCanonical =
      shape().getCanonicalReverseIndices(dims_.get());

  return createUnaryViewChange<Reverse_>(
      shape().get(), Dimensions(std::move(revDimsCanonical)));
}

Tensor Tensor::reduce(const CommutativeOp cop) const {
  return reduce(Shape({}), cop);
}

Tensor Tensor::reduce(const Shape &outShape, const CommutativeOp cop) const {
  auto out_ = reduce(shape().reductionDimensions(outShape), cop);
  return out_.reshape_(outShape);
}

Tensor Tensor::reduce(Dimension d, const CommutativeOp cop) const {
  Dimensions dims({d});
  return reduce(dims, cop);
}

Tensor Tensor::reduce(const Dimensions &d, CommutativeOp cop) const {

  if (d.empty()) {
    // we cannot just return this tensor, because there must at least be a
    // copy here.
  }
  auto outShape = shape().get();
  for (uint64_t i = 0; i < d.get().size(); ++i) {
    if (d.at(i).get() >= shape().rank_u64()) {
      err("Invalid dimension in reduce");
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

Tensor Tensor::mul_(const Tensor &rhs) const {
  return createWithNumpyShape<Mul_>({id(), rhs.id()});
}

Tensor Tensor::mul(const Tensor &rhs) const {
  return createWithNumpyShape<Mul>({id(), rhs.id()});
}

Tensor Tensor::add_(const Tensor &rhs) const {
  return createWithNumpyShape<Add_>({id(), rhs.id()});
}

Tensor Tensor::add(const Tensor &rhs) const {
  return createWithNumpyShape<Add>({id(), rhs.id()});
}

Tensor Tensor::padWithBroadcastConstZero_(const Lower &l,
                                          const Upper &u) const {

  auto z = constant(0.);

  // progressively pad the tensor, one dimension at a time. The padding on all
  // edges is all an alias of the constant 'z' above.

  auto t = Tensor(id(), &graph());
  for (uint64_t d = 0; d < rank_u64(); ++d) {

    // concatate the tensors (lower_padding, t, upper_padding). If the padding
    // is zero, then leave it off.
    Tensors toConcat;
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

Tensor Tensor::slice_(const Lower &l, const Upper &u) const {

  const auto outShape = shape().slice(l, u);

  // Check for a slice which doesn't slice anything out. The second condition
  // here is to confirm that the bounds are valid (lower is 0s).
  if (outShape == shape() && Shape(u) == shape()) {
    return {id(), &graph()};
  }

  return createUnaryWithNewShape<Slice_>(outShape, l, u);
}

Tensor Tensor::slice_(const Dimensions &dims,
                      const std::vector<uint64_t> &starts_,
                      const std::vector<uint64_t> &ends_) const {
  auto lu = shape().getFullSliceBounds(dims, starts_, ends_);
  return slice_(lu.first, lu.second);
}

Tensor Tensor::slice(const Dimensions &a,
                     const std::vector<uint64_t> &b,
                     const std::vector<uint64_t> &c) const {
  return slice_(a, b, c).copy();
}

Tensor Tensor::copy(DeviceId target) const {

  const auto targetType = graph().device(target).deviceType();
  if (targetType != deviceType()) {
    std::ostringstream oss;
    oss << "Tensor::copy(target=" << target
        << ") is invalid, as this tensor, " << id()
        << " has a different device type. "
        << "This method can only copy Ipu->Ipu. " << deviceType() << "->"
        << targetType << " is not allowed.";
    err(oss.str());
  }

  auto targetTensor = variable(target);
  return targetTensor.copyFrom_(*this);
}

Tensor Tensor::slice(const Lower &l, const Upper &u) const {
  return slice_(l, u).copy();
}

Tensor Tensor::slice(Dimension d, int64_t l, int64_t u) const {
  return slice_(d, l, u).copy();
}

Tensor Tensor::slice_(Dimension d, int64_t l, int64_t u) const {
  if (l < 0 || u < 0) {
    std::ostringstream oss;
    oss << "Invalid call, Tensor::slice_(Dimension = " << d.get()
        << ", l = " << l << ", u = " << u << "). "
        << "The lower (l) and upper (u) bounds must both be non-negative. ";
    err(oss.str());
  }

  const auto fullSliceBounds = shape().getFullSliceBounds(
      d, static_cast<uint64_t>(l), static_cast<uint64_t>(u));

  return slice_(std::get<0>(fullSliceBounds), std::get<1>(fullSliceBounds));
}

Tensor Tensor::reverse(const Dimensions &ds) const {
  return reverse_(ds).copy();
}

Tensor Tensor::reverse(uint64_t d) const { return reverse_(d).copy(); }
Tensor Tensor::to(DType t) const {
  return createTensor<Cast>({id()}, {info().withDType(t)});
}

Tensor Tensor::dimShuffleFinalTwo() const {
  return dimShuffle(Permutation::reverseFinalTwo(rank_u64()));
}

Tensor Tensor::dimRoll(uint64_t from, uint64_t to) const {
  return dimShuffle(Permutation::dimRoll(rank_u64(), {from, to}));
}

Tensor Tensor::dimRoll_(uint64_t from, uint64_t to) const {
  return dimShuffle_(Permutation::dimRoll(rank_u64(), {from, to}));
}

Tensor Tensor::dimShuffleFinalTwo_() const {
  return dimShuffle_(Permutation::reverseFinalTwo(rank_u64()));
}

Tensor Tensor::squeeze(const std::vector<uint64_t> &dims) const {
  return reshape(shape().squeeze(dims));
}

Tensor Tensor::squeeze_(const std::vector<uint64_t> &dims) const {
  return reshape_(shape().squeeze(dims));
}

Tensor Tensor::variable(const Shape &s0) const {
  return SubGraph(*this).variable(dtype(), s0, deviceId());
}

Tensor Tensor::variable(DeviceId dId, const Shape &s0) const {
  return SubGraph(*this).variable(dtype(), s0, dId);
}

Tensor Tensor::variable(DeviceId dId, SubGraphId sgId) const {
  return SubGraph(sgId, graph()).variable(dtype(), shape(), dId);
}

Tensor Tensor::variable(DType t) const {
  return SubGraph(*this).variable(t, shape(), deviceId());
}

Tensor Tensor::variable(DType t, const Shape &s) const {
  return SubGraph(*this).variable(t, s, deviceId());
}

Tensor Tensor::variable(SubGraphId sgId) const {
  return SubGraph(sgId, graph()).variable(dtype(), shape(), deviceId());
}

Tensor Tensor::variable() const {
  return SubGraph(*this).variable(dtype(), shape(), deviceId());
}

Tensor Tensor::variable(DeviceId did) const {
  return SubGraph(*this).variable(dtype(), shape(), did);
}

Tensor Tensor::concat_(const Tensors &ts, uint64_t axis) {

  if (ts.size() == 0) {
    err("Cannot concatenate empty vector of tensors.");
  }

  // If there is just 1 tensor being concatenated, return it.
  if (ts.size() == 1) {
    return Tensor(ts[0].id(), &ts[0].graph());
  }

  auto &m  = ts[0].graph();
  auto ids = SlickConverter::getIds(ts);
  auto out = ts[0].template createTensor<Concat_>(
      ids,
      {m.tensorInfo(ts[0]).withShape(Shape::concat(m.shapes(ids), axis))},
      axis);

  return out;
}

Tensor Tensor::updateFromHost_(
    const Tensor &source,
    const CopyBetweenHostAndIpuOptions &copyOptions) const {
  return createTensor<CopyFromHostToIpu_>(
      {source.id(), id()}, {{info()}}, copyOptions);
}

Tensor Tensor::updateFromIpu_(
    const Tensor &source,
    const CopyBetweenHostAndIpuOptions &copyOptions) const {
  return createTensor<CopyFromIpuToHost_>(
      {source.id(), id()}, {{info()}}, copyOptions);
}

Tensor
Tensor::hostToIpu(DeviceId ipuDestination,
                  const CopyBetweenHostAndIpuOptions &copyOptions) const {
  if (rank_u64() < 2) {
    err("Source of host->ipu copy must be at least rank 2.");
  }

  // Create an ipu tensor:
  const auto target = variable(ipuDestination, shape().fromDim(2));

  // Copy to ipu tensor:
  return target.updateFromHost_(*this, copyOptions);
}

Tensor
Tensor::ipuToHost(CircularBufferCount circularBufferCount,
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

class MatmulTensorMoldingHelper {
public:
  using T = Tensor;
  static Shape shape(const T &t) { return t.shape(); }
  static int64_t dim(const T &t, uint64_t d) { return t.dim(d); }
  static T unsqueeze(const T &t, uint64_t d) { return t.unsqueeze_(d); }
  static T reshape(const T &t, const Shape &s) { return t.reshape_(s); }
  static T expand(const T &t, const Shape &s) { return t.expand_(s); }
};

// Tensor getTensor(Tensor t) { return Tensor(t.id(), &t.graph()); }
} // namespace

Tensor Tensor::matmul(const Tensor &rhs,
                      DType outType,
                      const MatMulOptions &matMulOptions) const {

  // reshapes and expands.
  auto matMulPack =
      poprithms::ndarray::GroupedMatMulPack<MatmulTensorMoldingHelper,
                                            Tensor>(*this, rhs);

  // output is rank-3.
  const Shape outShape{
      matMulPack.nGroups(), matMulPack.M_i64(), matMulPack.N_i64()};

  const TensorInfo outInfo(outShape, deviceId(), outType);

  const auto out3d = createTensor<MatMul>(
      {matMulPack.lhs3d(), matMulPack.rhs3d()}, {outInfo}, matMulOptions);

  // reshape to correct grouped matmul output shape.
  return out3d.reshape_(matMulPack.outShape());
}

Tensors Tensor::tensors(const TensorIds &ids, Graph &g) {
  Tensors ts;
  ts.reserve(ids.size());
  for (const auto &id : ids) {
    ts.push_back({id, &g});
  }
  return ts;
}

TensorIds Tensor::tensorIds(const Tensors &ts) {
  TensorIds tIds;
  tIds.reserve(ts.size());
  for (auto t : ts) {
    tIds.push_back(t.id());
  }
  return tIds;
}

Tensor Tensor::dynamicMultiSlice(const Tensor &offset,
                                 const Dimensions &dims,
                                 const Shape &sizes) const {

  auto slicedShape = DynamicMultiWithDimensions_::getSlicedShape(
      offset.shape(), shape(), dims, sizes);

  const auto outInfo = info().withShape(slicedShape);

  auto slice = variable(slicedShape);

  return slice.dynamicMultiSlice_(*this, offset, dims);
}

Tensor Tensor::dynamicMultiSlice_(const Tensor &sliceable,
                                  const Tensor &offset,
                                  const Dimensions &dims) const {

  return createTensor<DynamicMultiSlice_>(
      {sliceable, *this, offset}, {info()}, dims);
}

Tensor Tensor::dynamicMultiUpdate_(const Tensor &update,
                                   const Tensor &offset,
                                   const Dimensions &dims) const {
  return createTensor<DynamicMultiUpdate_>(
      {id(), update.id(), offset.id()}, {info()}, dims);
}

Tensor Tensor::dynamicMultiUpdateMax_(const Tensor &source,
                                      const Tensor &offset) const {
  return createTensor<DynamicMultiUpdateMax_>(
      {id(), source.id(), offset.id()}, {info()});
}

Tensor Tensor::setToLowest_() const {
  return fill_(HostTensor::lowestScalar(dtype()));
}
Tensor Tensor::reduceSumAcrossReplicas_() const {
  return createUnaryWithSameInfo<ReduceSumAcrossReplicas_>();
}
Tensor Tensor::reduceSumAcrossReplicas() const {
  return createUnaryWithSameInfo<ReduceSumAcrossReplicas>();
}

Tensor Tensor::update_(const Tensor &update,
                       const Offsets &offsets,
                       const Dimensions &dims) const {
  auto starts = offsets.get();
  auto ends_  = update.shape().addToDims(offsets.get_i64()).get_u64();
  auto s_     = slice_(dims, starts, ends_);
  s_.copyFrom_(update);
  return Tensor(id(), &graph());
}

Tensor Tensor::update_(const Tensor &update, const Offsets &offsets) const {

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

Tensor Tensor::softmax(uint64_t d, StableSoftmax stable) const {

  Tensor t(id(), &graph());

  // subtract the maximum in each reduction plane.
  if (stable == StableSoftmax::Yes) {
    t = t - t.reduceMax(Dimension(d));
  }
  t = t.exp();
  return t / t.reduceSum(Dimension(d));
}

NllGrad Tensor::nllGrad(const Tensor &labels, StableSoftmax ss) const {

  if (rank_u64() != 2) {
    std::ostringstream oss;
    oss << "Can only call nnlGrad on rank-2 tensors (N, C). "
        << "This tensor has shape " << shape() << '.';
    err(oss.str());
  }

  if (labels.rank_u64() != 1 || labels.dim(0) != dim(0)) {
    std::ostringstream oss;
    oss << "Expected labels to have shape (" << dim(0)
        << ") for this tensor of shape " << shape()
        << ". But labels has shape " << labels.shape() << '.';
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

void Tensor::append(std::ostream &os) const {
  os << "id=" << id() << ",subGraphId=" << subGraphId()
     << ",shape=" << shape() << ",dtype=" << dtype();
}

Tensor Tensor::remoteToIpu(const Tensor &indices) const {

  const Shape sliceShape =
      CopyBetweenRemoteAndIpu_::shapeOfIpuSlice(indices.shape(), shape());

  const auto sliceInfo = TensorInfo(sliceShape, indices.deviceId(), dtype());

  const auto slice = SubGraph(*this).variable(sliceInfo);

  return slice.updateIpuFromRemote_(Tensor(id(), &graph()), indices);
}

Tensor Tensor::updateIpuFromRemote_(const Tensor &remoteTensor,
                                    const Tensor &indices) const {

  // This is an ipu tensor. Output is an alias of it.
  return createTensor<CopyFromRemoteToIpu_>(
      TensorIds({remoteTensor.id(), id(), indices.id()}), {info()});
}

Tensor Tensor::updateRemoteFromIpu_(const Tensor &ipuTensor,
                                    const Tensor &indices) const {

  // This is a remote tensor. Output is an alias of it.
  return createTensor<CopyFromIpuToRemote_>(
      TensorIds({id(), ipuTensor.id(), indices.id()}), {info()});
}

Tensor Tensor::ipuToRemote(const Tensor &indices,
                           uint64_t nRepeats,
                           const RemoteOptions &opts) const {
  const auto remoteShape =
      CopyBetweenRemoteAndIpu_::shapeOfRemoteSliceable(shape(), nRepeats);

  const auto remote =
      SubGraph(*this).remoteVariable(dtype(), remoteShape, deviceId(), opts);
  return remote.updateRemoteFromIpu_({id(), &graph()}, indices);
}

Tensor Tensor::ipuToRemote(const RemoteOptions &opts) const {
  if (rank_u64() != 2 || dim(0) != 1) {
    std::ostringstream oss;
    oss << "Expected rank-2 tensor with dim(0)=1. "
        << "But this ipu tensor has shape " << shape() << '.';
    err(oss.str());
  }
  auto indices = constant(DType::Unsigned32, 0).reshape_({1});
  return ipuToRemote(indices, 1, opts);
}

Tensor Tensor::remoteToIpu() const {
  if (rank_u64() != 2 || dim(0) != 1) {
    std::ostringstream oss;
    oss << "Expected rank-2 tensor with dim(0)=1. "
        << "But this remote tensor has shape " << shape() << '.';
    err(oss.str());
  }
  auto ipu_ = graph().remote(deviceId()).ipu();
  auto indices =
      SubGraph(*this).constant(DType::Unsigned32, 0, ipu_).reshape_({1});
  return remoteToIpu(indices);
}

const Op &Tensor::op() const { return graph().computeOp(opId()); }
SubGraphId Tensor::subGraphId() const { return graph().subGraphId(id()); }
uint64_t Tensor::nelms_u64() const { return graph().nelms_u64(id()); }
DeviceId Tensor::deviceId() const { return graph().deviceId(id()); }
DeviceType Tensor::deviceType() const {
  return graph().deviceType(deviceId());
}
DType Tensor::dtype() const { return graph().dtype(id()); }

Tensor Tensor::unfold_(Dimension d, uint64_t size, uint64_t step) const {
  using H = ndarray::TUnfoldHelper<Tensor>;
  return ndarray::Unfolder<Tensor, H>::unfold(
      Tensor(id(), &graph()), d.get(), size, step);
}

Graph &Tensor::graph() const { return *pGraph_; }

Tensor Tensor::name(const std::string &n) const {
  graph().setName(id(), n);
  return Tensor(id(), &graph());
}

Tensor Tensor::rootRef() const {
  return {op().rootRef(outIndex()), &graph()};
}

Tensors Tensor::refsExcludingSelf() const {
  return tensors(op().refsExcludingSelf(outIndex()), graph());
}

void Tensor::setName(const std::string &nm) const {
  graph().setName(id(), nm);
}

Tensor Tensor::inTensor(InIndex i) const {
  return {graph().inTensorId(opId(), i), &graph()};
}

Tensors Tensor::dstsInCallee(const CallEvent &ce) const {
  return tensors(graph().dstsInCallee(id(), ce), graph());
}

Tensor Tensor::srcInCallee(uint64_t calleeIndex) const {
  return {
      graph().srcInCallee({opId(), subGraphId(), calleeIndex}, outIndex()),
      &graph()};
}

void Tensor::before(const Tensor &after) const {
  graph().constraint(id(), after.id());
}

template <class TOp, class... Args>
Tensor Tensor::createWithNumpyShape(const TensorIds &ins,
                                    Args &&...args) const {
  return createTensor<TOp>(
      ins,
      {info().withShape(Shape::numpyVariadic(graph().shapes(ins)))},
      std::forward<Args>(args)...);
}

template <class TOp, class... Args>
Tensor Tensor::createBooleanWithNumpyShape(const TensorIds &ins,
                                           Args &&...args) const {
  return createTensor<TOp>(
      ins,
      {info()
           .withShape(Shape::numpyVariadic(graph().shapes(ins)))
           .withDType(DType::Boolean)},
      std::forward<Args>(args)...);
}

std::string Tensor::str() const {
  std::ostringstream oss;
  append(oss);
  return oss.str();
}

} // namespace compute
} // namespace common
} // namespace poprithms
