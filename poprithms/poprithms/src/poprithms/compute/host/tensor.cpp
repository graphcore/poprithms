// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "./include/allocdata.hpp"
#include "./include/basedata.hpp"
#include "./include/externdecl.hpp"
#include "./include/ieeehalf.hpp"
#include "./include/pointerdata.hpp"
#include "./include/typeswitch.hpp"
#include "./include/viewdata.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <random>
#include <sstream>

#include <poprithms/compute/host/error.hpp>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/util/printiter.hpp>
#include <poprithms/util/stringutil.hpp>

namespace poprithms {
namespace compute {
namespace host {

namespace {
template <typename T>
std::ostream &operator<<(std::ostream &ost, const std::vector<T> &vs) {
  util::append(ost, vs);
  return ost;
}
} // namespace

std::vector<char> Tensor::getNativeCharVector() const {
  return tData().getNativeCharVector();
}

Shapes Tensor::getShapes(const Tensors &tensors) {
  Shapes shapes_;
  shapes_.reserve(tensors.size());
  for (const auto &t : tensors) {
    shapes_.push_back(t.shape());
  }
  return shapes_;
}

Tensor Tensor::flattenTo2d(uint64_t axis) const {
  return reshape({shape().flattenTo2d(axis)});
}
Tensor Tensor::flattenTo2d_(uint64_t axis) const {
  return reshape_({shape().flattenTo2d(axis)});
}

Tensor Tensor::squeeze(const std::vector<uint64_t> &dims) const {
  return reshape(shape().squeeze(dims));
}
Tensor Tensor::squeeze_(const std::vector<uint64_t> &dims) const {
  return reshape_(shape().squeeze(dims));
}

Tensor Tensor::unsqueeze(const std::vector<uint64_t> &dims) const {
  return reshape(shape().unsqueeze(dims));
}
Tensor Tensor::unsqueeze_(const std::vector<uint64_t> &dims) const {
  return reshape_(shape().unsqueeze(dims));
}

template <typename T> Tensor Tensor::tCopyData(const Shape &s, const T *d) {
  if (d == nullptr && s.nelms() != 0) {
    std::ostringstream oss;
    oss << "Invalid call to Tensor::tCopyData(shape = " << s
        << " d = nullptr). ";
    throw error(oss.str());
  }
  const auto nElms = s.nelms_u64();
  std::vector<T> vData_(nElms);
  std::memcpy(vData_.data(), d, sizeof(T) * nElms);
  return Tensor(s,
                ndarray::get<T>(),
                std::make_shared<AllocData<T>>(std::move(vData_)));
}

class Tensor::Caster {
public:
  template <typename T> static Tensor go(const Shape &s, const void *vp) {
    return Tensor::tCopyData<T>(s, reinterpret_cast<const T *>(vp));
  }
};

template <>
Tensor Tensor::Caster::go<IeeeHalf>(const Shape &s, const void *vp) {
  const auto asu16 = static_cast<const uint16_t *>(vp);
  return Tensor::copyFloat16(s, asu16);
}

Tensor Tensor::copy(DType t, const Shape &s, const void *vp) {
  return typeSwitch<Caster, Tensor>(t, s, vp);
}

Tensor Tensor::to(DType t) const {
  switch (t) {
  case DType::Boolean:
    return toBoolean();
  case DType::Int8:
    return toInt8();
  case DType::Unsigned8:
    return toUnsigned8();
  case DType::Int16:
    return toInt16();
  case DType::Unsigned16:
    return toUnsigned16();
  case DType::Int32:
    return toInt32();
  case DType::Unsigned32:
    return toUnsigned32();
  case DType::Int64:
    return toInt64();
  case DType::Unsigned64:
    return toUnsigned64();
  case DType::Float16:
    return toFloat16();
  case DType::Float32:
    return toFloat32();
  case DType::Float64:
    return toFloat64();
  case DType::N: {
    throw error("\"N\" is not a valid DType to cast to");
  }
  default:
    throw error("unrecognised DType");
  }
}

class Tensor::ScalarCaster {
public:
  template <typename T> static Tensor go(const double v) {
    const T v_ = static_cast<T>(v);
    return tScalar<T>(v_);
  }
};

Tensor Tensor::scalar(DType type, const double v) {
  return typeSwitch<ScalarCaster, Tensor>(type, v);
}

Tensor concat(const Tensors &ts, uint64_t axis) {
  return Tensor::concat(ts, axis);
}

Tensor concat_(const Tensors &ts, uint64_t axis) {
  return Tensor::concat_(ts, axis);
}

void Tensor::confirmNonEmptyConcat(uint64_t nToCat) {
  if (nToCat == 0) {
    std::ostringstream oss;
    oss << "Failed in Tensor::confirmNonEmptyConcat(nToCat = " << nToCat
        << "). Only a non-empty vector of Tensors "
        << "can be concatenated.";
    throw error(oss.str());
  }
}

std::ostream &operator<<(std::ostream &ost, const Tensor &t) {
  t.append(ost);
  return ost;
}

void Tensor::confirmType(DType t) const {
  if (dtype() != t) {
    std::ostringstream oss;
    oss << "Failed in " << *this << ".confirmType(" << t << ')'
        << ", as this Tensor is of type " << dtype() << '.';
    throw error(oss.str());
  }
}

void Tensor::append(std::ostream &ost) const {
  ost << "shape=" << shape() << ",tData=(" << tData() << ",values=\n";
  tData().appendValues(ost, shape());
}

const Tensor &OptionalTensor::value() const {
  if (!has_value()) {
    throw error(
        "Invalid call to OptionalTensor::value(). has_value() is false.");
  }
  return t;
}

bool Tensor::identicalTo(const Tensor &rhs) const {
  return shape() == rhs.shape() && dtype() == rhs.dtype() &&
         &tData() == &rhs.tData();
}

Tensor Tensor::expand(const Shape &to) const {
  return Tensor(to, dtype(), tData().expand(shape(), to));
}

Tensor Tensor::expand_(const Shape &to) const {
  return Tensor(to, dtype(), tData().expand_(shape(), to));
}

namespace {
struct CoRowMaj {
  CoRowMaj(Tensor &&arg0_, Tensor &&arg1_, Shape &&shape_)
      : arg0(arg0_), arg1(arg1_), shape(shape_) {}
  Tensor arg0;
  Tensor arg1;
  Shape shape;
};

// Return, versions of a and b which are
//         1) row major
//         2) of the same shape (numpy broadcast their Shapes together).
//
CoRowMaj getRowMajorPair(const Tensor &a, const Tensor &b) {
  auto outShape = a.shape().numpyBinary(b.shape());
  auto arg0     = a.shape() == outShape ? a : a.expand(outShape);
  auto arg1     = b.shape() == outShape ? b : b.expand(outShape);
  arg0          = arg0.implIsOrigin() ? arg0 : arg0.copy();
  arg1          = arg1.implIsOrigin() ? arg1 : arg1.copy();
  return CoRowMaj(std::move(arg0), std::move(arg1), std::move(outShape));
}

Tensor getArg1InplaceTarget(const Tensor &a, const Shape &arg0Shape) {
  auto arg1 = a.shape() == arg0Shape ? a : a.expand(arg0Shape);
  arg1      = arg1.implIsOrigin() ? arg1 : arg1.copy();
  return arg1;
}
} // namespace

bool Tensor::implIsView() const { return !tData().isOriginData(); }

Tensor Tensor::add(const Tensor &rhs) const {
  auto co = getRowMajorPair(*this, rhs);
  return {co.shape, dtype(), co.arg0.tData().add(co.arg1.tData())};
}
Tensor Tensor::add_(const Tensor &rhs) const {
  tData().add_(getArg1InplaceTarget(rhs, shape()).tData());
  return *this;
}

Tensor Tensor::mul(const Tensor &rhs) const {
  auto co = getRowMajorPair(*this, rhs);
  return {co.shape, dtype(), co.arg0.tData().mul(co.arg1.tData())};
}
Tensor Tensor::mul_(const Tensor &rhs) const {
  tData().mul_(getArg1InplaceTarget(rhs, shape()).tData());
  return *this;
}

Tensor Tensor::pow(const Tensor &rhs) const {
  auto co = getRowMajorPair(*this, rhs);
  return {co.shape, dtype(), co.arg0.tData().pow(co.arg1.tData())};
}
Tensor Tensor::pow_(const Tensor &rhs) const {
  tData().pow_(getArg1InplaceTarget(rhs, shape()).tData());
  return *this;
}

Tensor Tensor::subtract(const Tensor &rhs) const {
  auto co = getRowMajorPair(*this, rhs);
  return {co.shape, dtype(), co.arg0.tData().subtract(co.arg1.tData())};
}
Tensor Tensor::subtract_(const Tensor &rhs) const {
  tData().subtract_(getArg1InplaceTarget(rhs, shape()).tData());
  return *this;
}

Tensor Tensor::divide(const Tensor &rhs) const {
  auto co = getRowMajorPair(*this, rhs);
  return {co.shape, dtype(), co.arg0.tData().divide(co.arg1.tData())};
}
Tensor Tensor::divide_(const Tensor &rhs) const {
  tData().divide_(getArg1InplaceTarget(rhs, shape()).tData());
  return *this;
}

Tensor Tensor::mod(const Tensor &rhs) const {
  auto co = getRowMajorPair(*this, rhs);
  return {co.shape, dtype(), co.arg0.tData().mod(co.arg1.tData())};
}
Tensor Tensor::mod_(const Tensor &rhs) const {
  tData().mod_(getArg1InplaceTarget(rhs, shape()).tData());
  return *this;
}

Tensor Tensor::operator<=(const Tensor &rhs) const {
  auto co = getRowMajorPair(*this, rhs);
  return {co.shape,
          DType::Boolean,
          co.arg0.tData().lessThanOrEqualTo(co.arg1.tData())};
}

Tensor Tensor::relu() const {
  auto positive = (*this > scalar(dtype(), 0.)).to(dtype());
  return mul(positive);
}

Tensor Tensor::relu_() const {
  auto positive = (*this > scalar(dtype(), 0.)).to(dtype());
  return mul_(positive);
}

Tensor Tensor::operator<(const Tensor &rhs) const {
  auto co = getRowMajorPair(*this, rhs);
  return {
      co.shape, DType::Boolean, co.arg0.tData().lessThan(co.arg1.tData())};
}

Tensor Tensor::operator>=(const Tensor &rhs) const {
  auto co = getRowMajorPair(*this, rhs);
  return {co.shape,
          DType::Boolean,
          co.arg0.tData().greaterThanOrEqualTo(co.arg1.tData())};
}

Tensor Tensor::operator>(const Tensor &rhs) const {
  auto co = getRowMajorPair(*this, rhs);
  return {
      co.shape, DType::Boolean, co.arg0.tData().greaterThan(co.arg1.tData())};
}

Tensor Tensor::operator==(const Tensor &rhs) const {
  auto co = getRowMajorPair(*this, rhs);
  return {co.shape, DType::Boolean, co.arg0.tData().equalTo(co.arg1.tData())};
}

Tensor Tensor::copy() const {
  return {shape(), dtype(), tData().toOriginData()};
}
Tensor Tensor::abs() const { return {shape(), dtype(), tData().abs()}; }
Tensor Tensor::abs_() const {
  tData().abs_();
  return *this;
}

Tensor Tensor::sqrt() const { return {shape(), dtype(), tData().sqrt()}; }
Tensor Tensor::sqrt_() const {
  tData().sqrt_();
  return *this;
}

Tensor Tensor::mod(int64_t modulo) const {
  return mod(Tensor::int64(modulo).to(dtype()));
}
Tensor Tensor::mod_(int64_t modulo) const {
  return mod_(Tensor::int64(modulo).to(dtype()));
}

Tensor Tensor::ceil() const { return {shape(), dtype(), tData().ceil()}; }
Tensor Tensor::ceil_() const {
  tData().ceil_();
  return *this;
}

Tensor Tensor::floor() const { return {shape(), dtype(), tData().floor()}; }
Tensor Tensor::floor_() const {
  tData().floor_();
  return *this;
}

void Tensor::confirmValidReshape(const Shape &s) const {
  if (s.nelms_u64() != nelms_u64()) {
    std::ostringstream oss;
    oss << "Failed in " << *this << ".confirmValidReshape(" << s
        << ") as number of elements not preserved. "
        << " Cannot go from Shape " << shape() << " with "
        << shape().nelms_u64() << " elements, to Shape " << s << " with "
        << s.nelms_u64() << '.';
  }
}

bool Tensor::allZero() const { return tData().allZero(); }

bool Tensor::allNonZero() const { return tData().allNonZero(); }

Tensor Tensor::reshape_(const Shape &s) const {
  confirmValidReshape(s);
  return {s, dtype(), tData_};
}
Tensor Tensor::reshape(const Shape &s) const {
  confirmValidReshape(s);
  return {s, dtype(), tData().toOriginData()};
}

Tensor Tensor::dimShuffle(const Permutation &p) const {
  return {shape().dimShuffle(p), dtype(), tData().dimShuffle(shape(), p)};
}

Tensor Tensor::dimShuffle_(const Permutation &p) const {
  return {shape().dimShuffle(p), dtype(), tData().dimShuffle_(shape(), p)};
}

Tensor Tensor::dimShuffle() const {
  return dimShuffle(Permutation::reverse(rank_u64()));
}

Tensor Tensor::dimShuffle_() const {
  return dimShuffle_(Permutation::reverse(rank_u64()));
}

namespace {
std::vector<uint64_t> getCanonicalDims(const std::vector<uint64_t> &dims,
                                       const Shape &s0) {

  // Number of times (mod(2)) that a dimension appears in dims.
  std::vector<bool> dims_(s0.rank_u64(), false);
  for (auto d : dims) {
    if (d >= s0.rank_u64()) {
      std::ostringstream oss;
      oss << "Error in getCanonicalDims(dims=";
      util::append(oss, dims);
      oss << ", Shape s0=" << s0 << "). "
          << "s0 is of rank " << s0.rank_u64()
          << ", which is not greater than " << d << " : invalid dimension. ";

      throw error(oss.str());
    }
    dims_[d] = !dims_[d];
  }

  std::vector<uint64_t> canon;
  for (uint64_t d = 0; d < s0.rank_u64(); ++d) {
    if (dims_[d]) {
      canon.push_back(d);
    }
  }
  return canon;
}
} // namespace

Tensor Tensor::reverse(const std::vector<uint64_t> &dims) const {
  return {shape(),
          dtype(),
          tData().reverse(shape(), getCanonicalDims(dims, shape()))};
}

Tensor Tensor::reverse_(const std::vector<uint64_t> &dims) const {
  return {shape(),
          dtype(),
          tData().reverse_(shape(), getCanonicalDims(dims, shape()))};
}

Tensor Tensor::reverse(uint64_t d) const {
  return reverse(std::vector<uint64_t>{d});
}
Tensor Tensor::reverse_(uint64_t d) const {
  return reverse_(std::vector<uint64_t>{d});
}

Tensor Tensor::subSample(const std::vector<uint64_t> &strides) const {
  return {shape().subSample(strides),
          dtype(),
          tData().subSample(shape(), strides)};
}
Tensor Tensor::subSample_(const std::vector<uint64_t> &strides) const {
  return {shape().subSample(strides),
          dtype(),
          tData().subSample_(shape(), strides)};
}

namespace {
std::vector<uint64_t>
getStrides(uint64_t stride, uint64_t dimension, const Shape &s0) {
  std::vector<uint64_t> strides(s0.rank_u64(), 1);
  if (dimension >= s0.rank_u64()) {
    std::ostringstream oss;
    oss << "Invalid dimension (" << dimension
        << ") in getStrides, where shape (" << s0 << ") is only of rank "
        << s0.rank_u64() << ". Expected dimension to be less than rank. ";
    throw error(oss.str());
  }
  strides[dimension] = stride;
  return strides;
}
} // namespace

Tensor Tensor::subSample(Stride stride, Dimension dimension) const {
  return subSample(getStrides(stride.val, dimension.val, shape()));
}

Tensor Tensor::subSample_(Stride stride, Dimension dimension) const {
  return subSample_(getStrides(stride.val, dimension.val, shape()));
}

Tensor Tensor::slice(const Lower &l, const Upper &u) const {
  return {shape().slice(l, u), dtype(), tData().slice(shape(), l, u)};
}

Tensor Tensor::slice_(const Lower &l, const Upper &u) const {
  return {shape().slice(l, u), dtype(), tData().slice_(shape(), l, u)};
}

Tensor Tensor::gather(uint64_t dimension,
                      const std::vector<int64_t> &where) const {
  return {shape().resizeSingleDim(where.size(), dimension),
          dtype(),
          tData().gather(shape(), dimension, where)};
}
Tensor Tensor::gather_(uint64_t dimension,
                       const std::vector<int64_t> &where) const {
  return {shape().resizeSingleDim(where.size(), dimension),
          dtype(),
          tData().gather_(shape(), dimension, where)};
}

Tensor Tensor::slice(const Starts &starts,
                     const Ends &ends,
                     const Steps &steps,
                     const Dims &dims) const {
  const auto normalized =
      shape().getNormalizedSliceParams(starts, ends, steps, dims);
  return {
      shape().slice(normalized), dtype(), tData().slice(shape(), normalized)};
}

Tensor Tensor::slice_(const Starts &starts,
                      const Ends &ends,
                      const Steps &steps,
                      const Dims &dims) const {
  const auto normalized =
      shape().getNormalizedSliceParams(starts, ends, steps, dims);
  return {
      shape().slice(normalized), dtype(), tData().slice(shape(), normalized)};
}

bool Tensor::containsAliases() const { return tData().containsAliases(); }

// get the BaseData for each Tensor in tIns.
std::vector<const BaseData *> Tensor::getBaseDataPtrs(const Tensors &tIns) {
  std::vector<const BaseData *> tDatas;
  tDatas.reserve(tIns.size());
  for (const auto &tIn : tIns) {
    tDatas.push_back(&tIn.tData());
  }
  return tDatas;
}

// namespace

Tensor Tensor::concat(const Tensors &tIns, uint64_t axis) {

  confirmNonEmptyConcat(tIns.size());
  const auto shapes      = getShapes(tIns);
  const auto tDatas      = Tensor::getBaseDataPtrs(tIns);
  const auto tDataConcat = BaseData::concat(tDatas, shapes, axis);

  return {Shape::concat(shapes, axis), tIns[0].dtype(), tDataConcat};
}

Tensor Tensor::concat_(const Tensors &tIns, uint64_t axis) {

  confirmNonEmptyConcat(tIns.size());
  const auto shapes      = getShapes(tIns);
  const auto tDatas      = Tensor::getBaseDataPtrs(tIns);
  const auto tDataConcat = BaseData::concat_(tDatas, shapes, axis);

  return {Shape::concat(shapes, axis), tIns[0].dtype(), tDataConcat};
}

// return true if absolute(a - b) <= (atol + rtol * absolute(b)) for all a in
// this Tensor, b in rhs (this is exactly the numpy definition).
bool Tensor::allClose(const Tensor &b, double relTol, double absTol) const {
  const auto diff    = this->toFloat64() - b.toFloat64();
  const auto absDiff = diff.abs();
  const auto absB    = b.toFloat64().abs();
  const auto threshold =
      Tensor::float64(absTol) + Tensor::float64(relTol) * absB;

  auto reslt = (absDiff <= threshold).allNonZero();
  return reslt;
}

void Tensor::assertAllClose(const Tensor &b,
                            double relTol,
                            double absTol) const {
  if (!allClose(b, relTol, absTol)) {

    std::ostringstream oss;
    oss << "Failed in assertAllClose(.). "
        << "This Tensor is \n"
        << *this << ", \nand b is " << b << ". Failed with relTol=" << relTol
        << " and absTol=" << absTol;
    throw error(oss.str());
  }
}

Tensor operator+(const Tensor &a, const Tensor &b) { return a.add(b); }
Tensor operator-(const Tensor &a, const Tensor &b) { return a.subtract(b); }
Tensor operator*(const Tensor &a, const Tensor &b) { return a.mul(b); }
Tensor operator/(const Tensor &a, const Tensor &b) { return a.divide(b); }
Tensor operator%(const Tensor &a, const Tensor &b) { return a.mod(b); }

// numerical type specific templates:

template <typename T> Tensor Tensor::tScalar(T f) {
  return Tensor({}, ndarray::get<T>(), std::make_shared<AllocData<T>>(f));
}

template <typename T> Tensor Tensor::tRefData(const Shape &s, T *d) {
  return Tensor(s,
                ndarray::get<T>(),
                std::make_shared<PointerData<T>>(d, s.nelms_u64()));
}

namespace {
std::string
getBadSizeString(const Shape &s, const std::string &tString, uint64_t n) {
  std::ostringstream oss;
  oss << "Error in " << tString << ", where Shape is " << s
      << ", with number of elements " << s.nelms_u64()
      << ", and the data vector has length " << n << '.'
      << " There should be exactly 1 element in the data vector "
      << "per element in the Shape. ";
  return oss.str();
}

} // namespace

template <typename T>
Tensor Tensor::tMoveVector(const Shape &s, std::vector<T> &&vs) {
  if (vs.size() != s.nelms_u64()) {
    throw error(getBadSizeString(s,
                                 std::string("tCopyVector<") +
                                     ndarray::lcase<T>() + ">",
                                 vs.size()));
  }
  return Tensor(
      s, ndarray::get<T>(), std::make_shared<AllocData<T>>(std::move(vs)));
}

template <typename T>
Tensor Tensor::tCopyVector(const Shape &s, const std::vector<T> &vs) {
  if (vs.size() != s.nelms_u64()) {
    throw error(getBadSizeString(s,
                                 std::string("tCopyVector<") +
                                     ndarray::lcase<T>() + ">",
                                 vs.size()));
  }

  auto vsCopy = vs;
  return Tensor(
      s, DType::Float16, std::make_shared<AllocData<T>>(std::move(vsCopy)));
}

template <typename T> Tensor Tensor::tArange(T x0, T x1, T step) {
  std::vector<T> vals;
  if ((x1 - x0) * step > T(0)) {
    vals.reserve(1 + static_cast<uint64_t>((x1 - x0) / step));
  }

  auto current = x0;
  while ((x1 - current) * step > T(0)) {
    vals.push_back(current);
    current = x0 + static_cast<T>(vals.size()) * step;
  }

  const Shape s{static_cast<int64_t>(vals.size())};
  return Tensor(
      s, ndarray::get<T>(), std::make_shared<AllocData<T>>(std::move(vals)));
}

template <typename T>
Tensor Tensor::tRandomUniform(T low, T upp, const Shape &s, uint32_t seed) {
  std::mt19937 gen(seed);
  std::vector<T> data__(s.nelms_u64());
  const auto denom  = static_cast<T>(gen.max() - gen.min());
  const auto factor = (upp - low) / denom;
  for (auto &x : data__) {
    x = low + factor * static_cast<T>(gen() - gen.min());
  }
  return Tensor(s,
                ndarray::get<T>(),
                std::make_shared<AllocData<T>>(std::move(data__)));
}

// Type specific implementations:

// Float64
Tensor Tensor::float64(const Shape &s, std::vector<double> &&vs) {
  return tMoveVector<double>(s, std::move(vs));
}
Tensor Tensor::float64(const Shape &s, const std::vector<double> &vs) {
  return tCopyVector<double>(s, vs);
}
Tensor
Tensor::uniformFloat64(double l, double u, const Shape &s, uint32_t seed) {
  return tRandomUniform<double>(l, u, s, seed);
}

std::vector<double> Tensor::getFloat64Vector() const {
  return tData().getFloat64Vector();
}

Tensor Tensor::float64(double f) { return tScalar<double>(f); }
Tensor Tensor::toFloat64() const {
  return Tensor(shape(), DType::Float64, tData().toFloat64());
}
Tensor Tensor::refFloat64(const Shape &s, double *p) {
  return tRefData(s, p);
}
Tensor Tensor::copyFloat64(const Shape &s, const double *v) {
  return Tensor::tCopyData<double>(s, v);
}
Tensor Tensor::arangeFloat64(double x0, double x1, double step) {
  return tArange<double>(x0, x1, step);
}

// Float32
Tensor Tensor::float32(const Shape &s, std::vector<float> &&vs) {
  return tMoveVector<float>(s, std::move(vs));
}
Tensor
Tensor::uniformFloat32(float l, float u, const Shape &s, uint32_t seed) {
  return tRandomUniform<float>(l, u, s, seed);
}
Tensor Tensor::float32(const Shape &s, const std::vector<float> &vs) {
  return tCopyVector<float>(s, vs);
}
Tensor Tensor::arangeFloat32(float x0, float x1, float step) {
  return tArange<float>(x0, x1, step);
}
std::vector<float> Tensor::getFloat32Vector() const {
  return tData().getFloat32Vector();
}
Tensor Tensor::copyFloat32(const Shape &s, const float *v) {
  return Tensor::tCopyData<float>(s, v);
}
Tensor Tensor::toFloat32() const {
  return Tensor(shape(), DType::Float32, tData().toFloat32());
}
Tensor Tensor::refFloat32(const Shape &s, float *p) { return tRefData(s, p); }
Tensor Tensor::float32(float f) { return tScalar<float>(f); }

// Float16
Tensor Tensor::float16(float f) { return tScalar<IeeeHalf>(f); }
Tensor Tensor::toFloat16() const {
  return Tensor(shape(), DType::Float16, tData().toFloat16());
}
Tensor Tensor::copyFloat16(const Shape &s, const uint16_t *v) {
  std::vector<IeeeHalf> halfs;
  halfs.resize(s.nelms_u64());
  for (uint64_t i = 0; i < halfs.size(); ++i) {
    halfs[i] = IeeeHalf::fromBits(v[i]);
  }
  return {s,
          DType::Float16,
          std::make_shared<AllocData<IeeeHalf>>(std::move(halfs))};
}
Tensor
Tensor::uniformFloat16(float l, float u, const Shape &s, uint32_t seed) {
  return uniformFloat32(l, u, s, seed).toFloat16();
}
Tensor Tensor::float16(const Shape &s, const std::vector<uint16_t> &vs) {
  if (s.nelms_u64() != vs.size()) {
    throw error(getBadSizeString(s, "float16", vs.size()));
  }
  return copyFloat16(s, vs.data());
}

Tensor Tensor::arangeFloat16(float x0, float x1, float step) {
  return arangeFloat32(x0, x1, step).toFloat16();
}
std::vector<uint16_t> Tensor::getFloat16Vector_u16() const {
  return tData().getFloat16Vector_u16();
}

// Int64:
Tensor Tensor::copyInt64(const Shape &s, const int64_t *v) {
  return Tensor::tCopyData<int64_t>(s, v);
}
Tensor Tensor::int64(const Shape &s, std::vector<int64_t> &&vs) {
  return tMoveVector<int64_t>(s, std::move(vs));
}
Tensor Tensor::refInt64(const Shape &s, int64_t *p) { return tRefData(s, p); }
Tensor Tensor::toInt64() const {
  return Tensor(shape(), DType::Int64, tData().toInt64());
}
Tensor Tensor::int64(const Shape &s, const std::vector<int64_t> &vs) {
  return tCopyVector<int64_t>(s, vs);
}
Tensor Tensor::arangeInt64(int64_t x0, int64_t x1, int64_t step) {
  return tArange<int64_t>(x0, x1, step);
}
std::vector<int64_t> Tensor::getInt64Vector() const {
  return tData().getInt64Vector();
}
Tensor Tensor::int64(int64_t f) { return tScalar<int64_t>(f); }

// Unsigned64:
Tensor Tensor::copyUnsigned64(const Shape &s, const uint64_t *v) {
  return Tensor::tCopyData<uint64_t>(s, v);
}
Tensor Tensor::unsigned64(const Shape &s, std::vector<uint64_t> &&vs) {
  return tMoveVector<uint64_t>(s, std::move(vs));
}
Tensor Tensor::refUnsigned64(const Shape &s, uint64_t *p) {
  return tRefData(s, p);
}
Tensor Tensor::toUnsigned64() const {
  return Tensor(shape(), DType::Unsigned64, tData().toUnsigned64());
}
Tensor Tensor::unsigned64(const Shape &s, const std::vector<uint64_t> &vs) {
  return tCopyVector<uint64_t>(s, vs);
}
Tensor Tensor::arangeUnsigned64(uint64_t x0, uint64_t x1, uint64_t step) {
  return tArange<uint64_t>(x0, x1, step);
}
std::vector<uint64_t> Tensor::getUnsigned64Vector() const {
  return tData().getUnsigned64Vector();
}
Tensor Tensor::unsigned64(uint64_t f) { return tScalar<uint64_t>(f); }

// Int32:
Tensor Tensor::copyInt32(const Shape &s, const int32_t *v) {
  return Tensor::tCopyData<int32_t>(s, v);
}
Tensor Tensor::int32(const Shape &s, std::vector<int32_t> &&vs) {
  return tMoveVector<int32_t>(s, std::move(vs));
}
Tensor Tensor::refInt32(const Shape &s, int32_t *p) { return tRefData(s, p); }
Tensor Tensor::toInt32() const {
  return Tensor(shape(), DType::Int32, tData().toInt32());
}
Tensor Tensor::int32(const Shape &s, const std::vector<int32_t> &vs) {
  return tCopyVector<int32_t>(s, vs);
}
Tensor Tensor::arangeInt32(int32_t x0, int32_t x1, int32_t step) {
  return tArange<int32_t>(x0, x1, step);
}
std::vector<int32_t> Tensor::getInt32Vector() const {
  return tData().getInt32Vector();
}
Tensor Tensor::int32(int32_t f) { return tScalar<int32_t>(f); }

// Unsigned32:
Tensor Tensor::copyUnsigned32(const Shape &s, const uint32_t *v) {
  return Tensor::tCopyData<uint32_t>(s, v);
}
Tensor Tensor::unsigned32(const Shape &s, std::vector<uint32_t> &&vs) {
  return tMoveVector<uint32_t>(s, std::move(vs));
}
Tensor Tensor::refUnsigned32(const Shape &s, uint32_t *p) {
  return tRefData(s, p);
}
Tensor Tensor::toUnsigned32() const {
  return Tensor(shape(), DType::Unsigned32, tData().toUnsigned32());
}
Tensor Tensor::unsigned32(const Shape &s, const std::vector<uint32_t> &vs) {
  return tCopyVector<uint32_t>(s, vs);
}
Tensor Tensor::arangeUnsigned32(uint32_t x0, uint32_t x1, uint32_t step) {
  return tArange<uint32_t>(x0, x1, step);
}
std::vector<uint32_t> Tensor::getUnsigned32Vector() const {
  return tData().getUnsigned32Vector();
}
Tensor Tensor::unsigned32(uint32_t f) { return tScalar<uint32_t>(f); }

// Int16:
Tensor Tensor::copyInt16(const Shape &s, const int16_t *v) {
  return Tensor::tCopyData<int16_t>(s, v);
}
Tensor Tensor::int16(const Shape &s, std::vector<int16_t> &&vs) {
  return tMoveVector<int16_t>(s, std::move(vs));
}
Tensor Tensor::refInt16(const Shape &s, int16_t *p) { return tRefData(s, p); }
Tensor Tensor::toInt16() const {
  return Tensor(shape(), DType::Int16, tData().toInt16());
}
Tensor Tensor::int16(const Shape &s, const std::vector<int16_t> &vs) {
  return tCopyVector<int16_t>(s, vs);
}
Tensor Tensor::arangeInt16(int16_t x0, int16_t x1, int16_t step) {
  return tArange<int16_t>(x0, x1, step);
}
std::vector<int16_t> Tensor::getInt16Vector() const {
  return tData().getInt16Vector();
}
Tensor Tensor::int16(int16_t f) { return tScalar<int16_t>(f); }

// Unsigned16:
Tensor Tensor::copyUnsigned16(const Shape &s, const uint16_t *v) {
  return Tensor::tCopyData<uint16_t>(s, v);
}
Tensor Tensor::unsigned16(const Shape &s, std::vector<uint16_t> &&vs) {
  return tMoveVector<uint16_t>(s, std::move(vs));
}
Tensor Tensor::refUnsigned16(const Shape &s, uint16_t *p) {
  return tRefData(s, p);
}
Tensor Tensor::toUnsigned16() const {
  return Tensor(shape(), DType::Unsigned16, tData().toUnsigned16());
}
Tensor Tensor::unsigned16(const Shape &s, const std::vector<uint16_t> &vs) {
  return tCopyVector<uint16_t>(s, vs);
}
Tensor Tensor::arangeUnsigned16(uint16_t x0, uint16_t x1, uint16_t step) {
  return tArange<uint16_t>(x0, x1, step);
}
std::vector<uint16_t> Tensor::getUnsigned16Vector() const {
  return tData().getUnsigned16Vector();
}
Tensor Tensor::unsigned16(uint16_t f) { return tScalar<uint16_t>(f); }

// Int8:
Tensor Tensor::copyInt8(const Shape &s, const int8_t *v) {
  return Tensor::tCopyData<int8_t>(s, v);
}
Tensor Tensor::int8(const Shape &s, std::vector<int8_t> &&vs) {
  return tMoveVector<int8_t>(s, std::move(vs));
}
Tensor Tensor::refInt8(const Shape &s, int8_t *p) { return tRefData(s, p); }
Tensor Tensor::toInt8() const {
  return Tensor(shape(), DType::Int8, tData().toInt8());
}
Tensor Tensor::int8(const Shape &s, const std::vector<int8_t> &vs) {
  return tCopyVector<int8_t>(s, vs);
}
Tensor Tensor::arangeInt8(int8_t x0, int8_t x1, int8_t step) {
  return tArange<int8_t>(x0, x1, step);
}
std::vector<int8_t> Tensor::getInt8Vector() const {
  return tData().getInt8Vector();
}
Tensor Tensor::int8(int8_t f) { return tScalar<int8_t>(f); }

// Unsigned8:
Tensor Tensor::copyUnsigned8(const Shape &s, const uint8_t *v) {
  return Tensor::tCopyData<uint8_t>(s, v);
}
Tensor Tensor::unsigned8(const Shape &s, std::vector<uint8_t> &&vs) {
  return tMoveVector<uint8_t>(s, std::move(vs));
}
Tensor Tensor::refUnsigned8(const Shape &s, uint8_t *p) {
  return tRefData(s, p);
}
Tensor Tensor::toUnsigned8() const {
  return Tensor(shape(), DType::Unsigned8, tData().toUnsigned8());
}
Tensor Tensor::unsigned8(const Shape &s, const std::vector<uint8_t> &vs) {
  return tCopyVector<uint8_t>(s, vs);
}
Tensor Tensor::arangeUnsigned8(uint8_t x0, uint8_t x1, uint8_t step) {
  return tArange<uint8_t>(x0, x1, step);
}
std::vector<uint8_t> Tensor::getUnsigned8Vector() const {
  return tData().getUnsigned8Vector();
}
Tensor Tensor::unsigned8(uint8_t f) { return tScalar<uint8_t>(f); }

// Boolean:
Tensor Tensor::toBoolean() const {
  return Tensor(shape(), DType::Boolean, tData().toBool());
}
Tensor Tensor::boolean(const Shape &s, const std::vector<bool> &vs) {
  return Tensor(s, DType::Boolean, std::make_shared<AllocData<bool>>(vs));
}
Tensor Tensor::boolean(bool f) { return boolean({}, {f}); }

std::vector<bool> Tensor::getBooleanVector() const {
  return tData().getBoolVector();
}

namespace {

enum class BinaryType {
  Div = 0,
  Add,
  Sub,
  Mod,
  Mul,
  Pow,
  /* number of types in this enum: */ N
};
constexpr auto NBinaryTypes = static_cast<uint64_t>(BinaryType::N);

const std::map<std::string, BinaryType> &getBinaryTypes() {
  const static std::map<std::string, BinaryType> types{
      {"mul", BinaryType::Mul},
      {"div", BinaryType::Div},
      {"sub", BinaryType::Sub},
      {"add", BinaryType::Add},
      {"pow", BinaryType::Pow},
      {"mod", BinaryType::Mod}};

  // Logic check to catch obvious errors.
  if (types.size() != NBinaryTypes) {
    std::ostringstream oss;
    oss << "There are " << NBinaryTypes << " types in the BinaryType enum. "
        << "But the number of entries os the map from them to strings is "
        << types.size() << ". They should match.";
    throw error(oss.str());
  }

  return types;
}

using BinaryNames =
    std::array<std::string, static_cast<uint64_t>(BinaryType::N)>;

BinaryNames createBinaryNames() {
  BinaryNames names;

  const std::string unset = "none";

  for (auto &n : names) {
    n = unset;
  }
  const auto &types = getBinaryTypes();
  for (const auto &[name, type] : types) {
    names[static_cast<uint64_t>(type)] = name;
  }

  // Confirm that all BinaryTypes have a name now:
  for (const auto &n : names) {
    if (n == unset) {
      std::ostringstream oss;
      oss << "Failed to give all types in BinaryType a corresponding string. "
          << "Incomplete implementation. ";
      throw error(oss.str());
    }
  }
  return names;
}

const BinaryNames &getBinaryNames() {
  const static auto names = createBinaryNames();
  return names;
}

std::string getString(BinaryType t) {
  const auto &names = getBinaryNames();
  return names[static_cast<uint64_t>(t)];
}
std::ostream &operator<<(std::ostream &ost, BinaryType t) {
  ost << getString(t);
  return ost;
}

std::string getCanonical(const std::string &x) {
  // make the name lower case:
  auto l = poprithms::util::lowercase(x);
  if (l == "multiply") {
    l = "mul";
  }
  if (l == "divide") {
    l = "div";
  }
  if (l == "subtract") {
    l = "sub";
  }
  return l;
}
BinaryType getBinaryType(const std::string &x) {
  const auto l      = getCanonical(x);
  const auto &names = getBinaryNames();
  const auto &types = getBinaryTypes();
  const auto found  = types.find(l);
  if (found == types.cend()) {
    std::ostringstream oss;
    oss << "Failed to find " << x << " (or canonical form, " << l << ")"
        << " in the map of supported binary types: ";
    util::append(oss, std::vector{names.cbegin(), names.cend()});
    throw error(oss.str());
  }
  return found->second;
}

} // namespace

Tensor Tensor::binary(const std::string &opType, const Tensor &b) const {
  const auto ot = getBinaryType(opType);
  switch (ot) {
  case BinaryType::Mul:
    return mul(b);
  case BinaryType::Div:
    return divide(b);
  case BinaryType::Add:
    return add(b);
  case BinaryType::Sub:
    return subtract(b);
  case BinaryType::Mod:
    return mod(b);
  case BinaryType::Pow:
    return pow(b);
  default: {
    std::ostringstream oss;
    oss << "Failed to map BinaryType::" << ot
        << " to a host Tensor API call in binary. ";
    throw error(oss.str());
  }
  }
}

Tensor Tensor::binary_(const std::string &opType, const Tensor &b) const {
  const auto ot = getBinaryType(opType);
  switch (ot) {
  case BinaryType::Mul:
    return mul_(b);
  case BinaryType::Div:
    return divide_(b);
  case BinaryType::Add:
    return add_(b);
  case BinaryType::Sub:
    return subtract_(b);
  case BinaryType::Mod:
    return mod_(b);
  case BinaryType::Pow:
    return pow_(b);
  default: {
    std::ostringstream oss;
    oss << "Failed to map BinaryType::" << ot
        << " to a host Tensor API call in binary_. ";
    throw error(oss.str());
  }
  }
}

bool Tensor::isBinary(const std::string &x) {
  const auto &types = getBinaryTypes();
  const auto found  = types.find(getCanonical(x));
  return found != types.cend();
}

void Tensor::assertIsBinary(const std::string &x) {
  if (!isBinary(x)) {
    std::ostringstream oss;
    oss << "Failed in assertIsBinary(" << x << "). ";
    throw error(oss.str());
  }
}

} // namespace host
} // namespace compute
} // namespace poprithms
