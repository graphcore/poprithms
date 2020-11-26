// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_TENSOR_HPP
#define POPRITHMS_COMPUTE_HOST_TENSOR_HPP
#include <memory>
#include <sstream>

#include <poprithms/compute/host/usings.hpp>

namespace poprithms {
namespace compute {
namespace host {

class BaseData;

/**
 * A Tensor class for just-in-time host computation with poplar aliasing
 * semantics.
 *
 * It is similar to numpy.ndarray,
 * https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
 *
 * The main difference between this Tensor class and numpy.ndarray
 * is how aliasing works:
 *
 * 1) This class is explicit about when aliases are created
 * 1) Aliases can always be created.
 *
 * To illustrate that numpy's aliasing is not explicit, consider:
 *
 *  numpy code 1:
 *  <code>
 *       base = np.ones(9).reshape(3,3)
 *       x0 = base[0:3, 0:2].reshape(6,1)
 *       x0 *= 0
 *       print(base.sum())
 *       # prints 9
 *  </code>
 *
 *  numpy code 2:
 *  <code>
 *       base = np.ones(9).reshape(3,3)
 *       x0 = base[0:2, 0:3].reshape(2,3)
 *       x0 *= 0
 *       print(base.sum())
 *       # prints 3
 *  </code>
 *
 * In both cases, a slice of 6 elements is taken from a numpy.array of 9
 * elements, and all 6 elements are set to 0.
 *
 * In case 2, the values in the initial numpy.array \a base are set to 0 too,
 * because of aliasing. in case 1 they are not. Why is there a difference?
 * Because in numpy, aliases are created only if the resulting array can be
 * described by orthogonal strides.
 *
 * The main motivation for not using numpy aliasing rules for this class, is
 * to exactly match the behaviour of a poplar::Tensor with respects to
 * aliasing. A poplar::Tensor can model arbitrarily complex "view" changes,
 * and never silently fails to alias when the description is more complex than
 * can be described with orthogonal strides.
 *
 * This Tensor class uses PyTorch notation for aliasing: a suffix '_'
 * is used for all methods which create aliases. A method with suffix '_'
 * always creates an alias. A method without the '_' suffix never creates
 * an alias.
 *
 * PyTorch underscore convention:
 * https://pytorch.org/docs/stable/tensors.html?highlight=underscore
 *
 *
 * Another difference between this Tensor class and the numpy.ndarray class is
 * that there is no implicit type casting, and so both arguments to binary
 * operations must be of the same numerical type.
 *
 * */
class Tensor {
public:
  /**
   * Create a Float64 Tensor by copying data.
   *
   * Copy elements starting at pointer \a element0 to an internal allocation.
   *
   * \param shape The Shape of the Tensor.
   *
   * \param element0 The address of the first element of the Tensor.
   *                 Subsequent elements must be in row-major order, and
   *                 contiguous in memory.
   *
   * Behaviour is undefined if the number of contiguous doubles starting
   * from \a element0 is less than the number of elements in \a shape.
   * */
  static Tensor float64(const Shape &shape, const double *element0);

  /**
   * Create a Float64 Tensor by copying data.
   *
   * Copy the vector \a values into an internally managed allocation.
   *
   * \param shape The Shape of the Tensor.
   *
   * \param values The values of the Tensor, in row-major order.
   *
   * The size of \a values must be the same as the number of elements in \a
   * shape, if it is not an error is thrown.
   */
  static Tensor float64(const Shape &shape,
                        const std::vector<double> &values);

  /**
   * Create a Float64 Tensor by moving \a values into an internally managed
   * buffer.
   *
   * \param shape The Shape of the Tensor.
   *
   * \param values The values of the Tensor, in row-major order.
   *
   * The size of \a values must be the same as the number of elements in \a
   * shape.
   */
  static Tensor float64(const Shape &shape, std::vector<double> &&values);

  /**
   * Create a scalar Tensor of type Float64, with numerical value \a v.
   * */
  static Tensor float64(double v);

  /**
   * Create a Float64 Tensor which stores a reference to externally managed
   * memory.
   *
   * This constructor does not allocate any memory for Tensor elements.
   *
   * \param shape The Shape of the Tensor.
   *
   * \param element0 The address of the first element of the Tensor. The
   *                 elements must be in row-major order and contiguous in
   *                 memory.
   *
   * If the number of contiguous doubles starting from \a element0 is less
   * than the number of elements in \a shape, the behaviour is undefined.
   *
   * If the memory in \a element0 is deleted before this Tensor and all of its
   * derived values have been used, behaviour is undefined. The created Tensor
   * does not manage the lifetime of \a element0, this is the caller's
   * responsibilty.
   * */
  static Tensor refFloat64(const Shape &shape, double *element0);

  /**
   * Create a Tensor of type Float64 with values drawn independently from the
   * uniform distribution, U ~ Uniform[low, upp).
   *
   * See https://en.wikipedia.org/wiki/Uniform_distribution.
   *
   * Note that the random values drawn are platform invariant, as only C++
   * generators are usedm and no C++ distributions.
   * */
  static Tensor
  uniformFloat64(double low, double upp, const Shape &, uint32_t seed);

  /**
   * Create a Tensor with values linearly spaced between \a start and \a end,
   * with interval \a step. Based on the numpy.ndarray equivalent:
   * https://numpy.org/doc/stable/reference/generated/numpy.arange.html
   * */
  static Tensor arangeFloat64(double start, double stop, double step);

  /**
   * \return The values in this Tensor as Float64 values and in row-major
   *         order
   * */
  std::vector<double> getFloat64Vector() const;

  /**
   * Cast this Tensor to a Float64 Tensor.
   *
   * This method allocates a new buffer even if this Tensor is already of type
   * Float64. This is in keeping with the PyTorch _ method naming convention.
   * */
  Tensor toFloat64() const;

  /**
   * Float32 specific Tensor methods. \see corresponding Float64 methods.
   */
  static Tensor float32(const Shape &, const float *);
  static Tensor float32(const Shape &, const std::vector<float> &);
  static Tensor float32(const Shape &, std::vector<float> &&);
  static Tensor float32(float);
  static Tensor refFloat32(const Shape &, float *);
  static Tensor
  uniformFloat32(float low, float upp, const Shape &, uint32_t seed);
  static Tensor arangeFloat32(float start, float stop, float step);
  Tensor toFloat32() const;
  std::vector<float> getFloat32Vector() const;

  /**
   * Float16 specific Tensor methods. \see corresponding Float64 methods.
   */
  static Tensor float16(const Shape &, const uint16_t *);
  static Tensor float16(const Shape &, const std::vector<uint16_t> &);
  static Tensor float16(float);
  static Tensor refFloat16(const Shape &, uint16_t *);
  static Tensor
  uniformFloat16(float low, float upp, const Shape &, uint32_t seed);
  static Tensor arangeFloat16(float start, float stop, float step);
  Tensor toFloat16() const;
  std::vector<uint16_t> getFloat16Vector_u16() const;

  /**
   * Int64 specific Tensor methods and factory functions.
   *
   * \see The corresponding Float64 methods.
   * */
  static Tensor int64(const Shape &, const int64_t *);
  static Tensor int64(const Shape &, const std::vector<int64_t> &);
  static Tensor int64(const Shape &, std::vector<int64_t> &&);
  static Tensor int64(int64_t);
  static Tensor refInt64(const Shape &, int64_t *);
  static Tensor arangeInt64(int64_t start, int64_t stop, int64_t step);
  Tensor toInt64() const;
  std::vector<int64_t> getInt64Vector() const;

  /**
   * Unsigned64 specific Tensor methods and factory functions.
   *
   * \see The corresponding Float64 methods.
   * */
  static Tensor unsigned64(const Shape &, const uint64_t *);
  static Tensor unsigned64(const Shape &, const std::vector<uint64_t> &);
  static Tensor unsigned64(const Shape &, std::vector<uint64_t> &&);
  static Tensor unsigned64(uint64_t);
  static Tensor refUnsigned64(const Shape &, uint64_t *);
  static Tensor
  arangeUnsigned64(uint64_t start, uint64_t stop, uint64_t step);
  Tensor toUnsigned64() const;
  std::vector<uint64_t> getUnsigned64Vector() const;

  /** Int32 type specific Tensor methods and factory functions.
   *
   * \see the corresponding Float64 methods.
   * */
  static Tensor int32(const Shape &, const int32_t *);
  static Tensor int32(const Shape &, const std::vector<int32_t> &);
  static Tensor int32(const Shape &, std::vector<int32_t> &&);
  static Tensor int32(int32_t);
  static Tensor refInt32(const Shape &, int32_t *);
  static Tensor arangeInt32(int32_t start, int32_t stop, int32_t step);
  Tensor toInt32() const;
  std::vector<int32_t> getInt32Vector() const;

  /** Unsigned32 type specific Tensor methods and factory functions.
   *
   * \see the corresponding Float64 methods.
   * */
  static Tensor unsigned32(const Shape &, const uint32_t *);
  static Tensor unsigned32(const Shape &, const std::vector<uint32_t> &);
  static Tensor unsigned32(const Shape &, std::vector<uint32_t> &&);
  static Tensor unsigned32(uint32_t);
  static Tensor refUnsigned32(const Shape &, uint32_t *);
  static Tensor
  arangeUnsigned32(uint32_t start, uint32_t stop, uint32_t step);
  Tensor toUnsigned32() const;
  std::vector<uint32_t> getUnsigned32Vector() const;

  /** Int16 type specific Tensor methods and factory functions.
   *
   * \see the corresponding Float64 methods.
   * */
  static Tensor int16(const Shape &, const int16_t *);
  static Tensor int16(const Shape &, const std::vector<int16_t> &);
  static Tensor int16(const Shape &, std::vector<int16_t> &&);
  static Tensor int16(int16_t);
  static Tensor refInt16(const Shape &, int16_t *);
  static Tensor arangeInt16(int16_t start, int16_t stop, int16_t step);
  Tensor toInt16() const;
  std::vector<int16_t> getInt16Vector() const;

  /** Unsigned16 type specific Tensor methods and factory functions.
   *
   * \see the corresponding Float64 methods.
   * */
  static Tensor unsigned16(const Shape &, const uint16_t *);
  static Tensor unsigned16(const Shape &, const std::vector<uint16_t> &);
  static Tensor unsigned16(const Shape &, std::vector<uint16_t> &&);
  static Tensor unsigned16(uint16_t);
  static Tensor refUnsigned16(const Shape &, uint16_t *);
  static Tensor
  arangeUnsigned16(uint16_t start, uint16_t stop, uint16_t step);
  Tensor toUnsigned16() const;
  std::vector<uint16_t> getUnsigned16Vector() const;

  /** Int8 type specific Tensor methods and factory functions.
   *
   * \see the corresponding Float64 methods.
   * */
  static Tensor int8(const Shape &, const int8_t *);
  static Tensor int8(const Shape &, const std::vector<int8_t> &);
  static Tensor int8(const Shape &, std::vector<int8_t> &&);
  static Tensor int8(int8_t);
  static Tensor refInt8(const Shape &, int8_t *);
  static Tensor arangeInt8(int8_t start, int8_t stop, int8_t step);
  Tensor toInt8() const;
  std::vector<int8_t> getInt8Vector() const;

  /** Unsigned8 type specific Tensor methods and factory functions.
   *
   * \see the corresponding Float64 methods.
   * */
  static Tensor unsigned8(const Shape &, const uint8_t *);
  static Tensor unsigned8(const Shape &, const std::vector<uint8_t> &);
  static Tensor unsigned8(const Shape &, std::vector<uint8_t> &&);
  static Tensor unsigned8(uint8_t);
  static Tensor refUnsigned8(const Shape &, uint8_t *);
  static Tensor arangeUnsigned8(uint8_t start, uint8_t stop, uint8_t step);
  Tensor toUnsigned8() const;
  std::vector<uint8_t> getUnsigned8Vector() const;

  /**
   * Boolean type specific Tensor methods and factory functions.
   *
   * \see the corresponding Float64 methods.
   *
   * Certain constructors for other numeric types do not have Boolean
   * equivalents. This is due the peculiar behaviour of the std::vector<bool>
   * class.
   * */
  static Tensor boolean(const Shape &, const std::vector<bool> &);
  static Tensor boolean(bool);
  Tensor toBoolean() const;
  std::vector<bool> getBooleanVector() const;

  /** \return true if all the elements in this Tensor are 0. */
  bool allZero() const;

  /** \return true if none of the elements in this Tensor are 0. */
  bool allNonZero() const;

  /**
   *
   * \return true iff, for all a in this Tensor and b in \a rhs,
   *         |a - b| <= (atol + rtol * |b|).
   *
   * This method is based on the numpy.ndarray allClose method:
   * see https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
   *
   * \param rhs The Tensor to compare this Tensor to.
   *
   * \param relTol Relative tolerance.
   *
   * \param absTol Absolute tolerance.
   * */
  bool allClose(const Tensor &rhs, double relTol, double absTol) const;

  /**
   * Throw a descriptive error if \a allClose evaluates to false.
   * */
  void assertAllClose(const Tensor &rhs, double relTol, double absTol) const;

  bool allEquivalent(const Tensor &rhs) const {
    return allClose(rhs, 0., 0.);
  }

  void assertAllEquivalent(const Tensor &rhs) const {
    assertAllClose(rhs, 0., 0.);
  }

  DType dtype() const { return dtype_; }
  const Shape &shape() const { return shape_; }
  uint64_t rank_u64() const { return shape().rank_u64(); }
  uint64_t nelms_u64() const { return shape().nelms_u64(); }
  uint64_t nbytes() const {
    return nelms_u64() * ndarray::nbytes_u64(dtype());
  }

  /**
   * \return false if and only if (iff) all elements of this Tensor correspond
   *         to distinct addresses in memory. In other words, return true iff
   *         this Tensor has self-aliases.
   *
   * Tensors with self-aliases can be created by inplace view changing
   * operations, that is '_' suffixed, operations. For example,
   *
   * <code>
   *    auto a = Tensor::float32(1.2f).reshape_({1,1});
   *    auto b = concat_({a,a}, 0);
   * </code>
   *
   * In this example, b contains 2 elements with the same address, and so is
   * self-aliasing. As a second example,
   *
   * <code>
   *    auto a = Tensor::int32(7).reshape_({1,1}).expand_({2,2});
   * </code>
   *
   * In this example, all the elements of `a':
   *    [[ 7 7 ]
   *     [ 7 7 ]]
   *
   * correspond to the same element in memory, and so a is self-aliasing.
   * */
  bool containsAliases() const;

  /** Append information about this Tensor to \p outputStream */
  void append(std::ostream &outputStream) const;

  /**
   * \return A copy of this Tensor. The returned Tensor is a new memory
   *         allocation. This is consistent with the PyTorch '_' notation used
   *         for all Tensor methods.
   * */
  Tensor copy() const;

  /**
   * Reshape this Tensor.
   *
   * \param to The Shape of the returned Tensor. It must have the same
   *           number of elements as this Tensor.
   *
   * Recall: the _ suffix denotes aliasing / inplace. For view-changing
   * operations, such as reshape, the '_' suffixed version corresponds to the
   * poplar version.
   * */
  Tensor reshape(const Shape &to) const;
  Tensor reshape_(const Shape &) const;

  /**
   * Reshape to a rank-2 Tensor, where the size of the first dimension is the
   * product of dimensions in [0, axis). Specifically, if axis = 0, the
   * returned Shape is (1,nelms) and if axis = rank, the returned Shape is
   * (nelms, 1).
   * */
  Tensor flattenTo2d(uint64_t axis) const;
  Tensor flattenTo2d_(uint64_t axis) const;

  /**
   * Reshape to a rank-1 Tensor.
   * */
  Tensor flatten() const { return reshape({shape().nelms()}); }
  Tensor flatten_() const { return reshape_({shape().nelms()}); }

  /**
   * \return Reshape this Tensor by removing all dimensions which have size
   *         `1'. Note that `0's are not removed.
   * */
  Tensor squeeze() const { return reshape(shape().squeeze()); }
  Tensor squeeze_() const { return reshape_(shape().squeeze()); }

  /**
   * Reshape this Tensor by removing 1's in certain dimensions.
   *
   * \return This Tensor, with dimensions in \a removed. If the dimension
   *         size of any dimension \in dims is not 1, an error is thrown.
   *
   * Example:
   *   If this Tensor has Shape (1,4,1,5,1) and dims is (0,4), then the
   *   returned Tensor has Shape (4,1,5).
   * */
  Tensor squeeze(const std::vector<uint64_t> &dims) const;
  Tensor squeeze_(const std::vector<uint64_t> &dims) const;

  /**
   * Reshape this Tensor by inserting singleton dimensions.
   *
   * \param dims The dimensions where the output Shape will have 1's inserted.
   *
   * Example: If this Tensor has Shape (3,4) and dims=(0,2,3), then the
   *          returned Tensor has Shape (1,2,1,1,3,4).
   * */
  Tensor unsqueeze(const std::vector<uint64_t> &dims) const;
  Tensor unsqueeze_(const std::vector<uint64_t> &dims) const;

  Tensor unsqueeze(uint64_t d) const { return reshape(shape().unsqueeze(d)); }
  Tensor unsqueeze_(uint64_t d) const {
    return reshape_(shape().unsqueeze(d));
  }

  /**
   * Expand the Tensor using numpy broadcasting rules.
   * https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html
   * */
  Tensor expand(const Shape &) const;
  Tensor expand_(const Shape &) const;

  /** Take a slice of this Tensor between bounds \a l (inclusive) and \a u
   * (exclusive). \a l and \a u must have size equal to the rank of this
   * Tensor, and for all dimensions d, it is required that
   *     0 <= l[d] <= u[d] <= dim(d).
   * */
  Tensor slice(const Lower &l, const Upper &u) const;
  Tensor slice_(const Lower &l, const Upper &u) const;

  /**
   * Reverse this Tensor along certain dimensions
   *
   * \param dimensions The dimensions along which to reverse the Tensor.
   *                   Dimensions may appear multiple times, in wich case the
   *                   reversal is repeated once for every appearance.
   * */

  Tensor reverse(const std::vector<uint64_t> &dimensions) const;
  Tensor reverse_(const std::vector<uint64_t> &dimensions) const;

  /**
   * Reverse along a single dimension */
  Tensor reverse(uint64_t) const;
  Tensor reverse_(uint64_t) const;

  /**
   * Subsample elements from this Tensor.
   *
   * \param strides The interval in each dimension between the elements to
   *                sample The elements of strides must be strictly positive.
   *
   * Example: if this Tensor has Shape (12,5) and strides is (6,2), then the
   *          returned Tensor has Shape (ceil(12/6)=2, ceil(5/2)=3)
   *
   * Subsampling starts at element 0 in each dimension.
   *
   * Example: If this Tensor is
   *             [[ 0 1 2 3 4 ]
   *              [ 5 6 7 8 9 ]]
   *
   * and strides=(2,2), then the returned Tensor is [[ 0 2 4 ]]
   * */
  Tensor subSample(const std::vector<uint64_t> &strides) const;
  Tensor subSample_(const std::vector<uint64_t> &strides) const;

  /** A stride, which must be explicitly constructed to avoid muddling with
   * Dimension, etc. */
  struct Stride {
    explicit Stride(uint64_t s_) : s(s_) {}
    uint64_t s;
  };

  /** A dimension, which must be explicitly constructed to avoid muddling with
   * Stride, etc. */
  struct Dimension {
    explicit Dimension(uint64_t d_) : d(d_) {}
    uint64_t d;
  };

  /** Subsample along a single dimension  */
  Tensor subSample(Stride, Dimension) const;
  Tensor subSample_(Stride, Dimension) const;

  /**
   * Slice and concatenate this Tensor along axis \a dimension, and at indices
   * \a where. For example, if this Tensor is
   *
   *   [[ 0 1
   *      2 3 ]]
   *
   * then gathering along dimension=1 at indices where={0,0,1,0}, creates
   * Tensor,
   *
   *   [[ 0 0 1 0
   *      2 2 3 2 ]].
   * */
  Tensor gather(uint64_t dimension, const std::vector<int64_t> &where) const;
  Tensor gather_(uint64_t dimension, const std::vector<int64_t> &where) const;

  /** A generalization of a matrix transpose. */
  Tensor dimShuffle(const Permutation &) const;
  Tensor dimShuffle_(const Permutation &) const;

  /** Reverse the dimensions of this Tensor.
   *
   * For rank-2 Tensors, this is equivalent to a matrix transpose.
   *
   * If this Tensor has Shape (2,3,5), then the returned Tensor has Shape
   * (5,3,2).
   * */
  Tensor dimShuffle() const;
  Tensor dimShuffle_() const;

  /**
   * Elementwise binary operations, which use numpy-style broadcasting rules.
   * The versions with the suffix _ are inplace on this Tensor. Note that
   * there is no implicit type casting, so \p rhs must have the same numerical
   * type as this Tensor.
   * */
  Tensor add(const Tensor &rhs) const;
  Tensor add_(const Tensor &rhs) const;

  Tensor mul(const Tensor &rhs) const;
  Tensor mul_(const Tensor &rhs) const;

  Tensor subtract(const Tensor &rhs) const;
  Tensor subtract_(const Tensor &rhs) const;

  Tensor divide(const Tensor &rhs) const;
  Tensor divide_(const Tensor &rhs) const;

  Tensor mod(const Tensor &rhs) const;
  Tensor mod_(const Tensor &rhs) const;

  Tensor pow(const Tensor &rhs) const;
  Tensor pow_(const Tensor &rhs) const;

  /** These elementwise binary operations return Tensors of type Boolean. */
  Tensor operator<(const Tensor &rhs) const;
  Tensor operator<=(const Tensor &rhs) const;
  Tensor operator>(const Tensor &rhs) const;
  Tensor operator>=(const Tensor &rhs) const;
  Tensor operator==(const Tensor &rhs) const;

  /** Elementwise unary operations. The versions with the suffix _ act
   * inplace on the memory of this Tensor. */
  Tensor abs() const;
  Tensor abs_() const;

  Tensor ceil() const;
  Tensor ceil_() const;

  Tensor floor() const;
  Tensor floor_() const;

  Tensor mod(int64_t modulo) const;
  Tensor mod_(int64_t modulo) const;

  Tensor sqrt() const;
  Tensor sqrt_() const;

  /**
   * relu(x) = x*(x > 0)
   * */
  Tensor relu() const;
  Tensor relu_() const;

  /**
   * \return true: if and only if (iff) this and \a rhs have the same shape,
   *         same type, and some data addresses. Note that this does no
   *         numerical comparison of Tensors, and so 2 Tensors which are
   *         numerically identical will not compare equal unless their datas
   *         have the same addresses.
   */
  bool identicalTo(const Tensor &rhs) const;

  /**
   * Throw an error with a descriptive message if the DType argument differs
   * from this Tensor's type. Recall that DType is the numerical type. */
  void confirmType(DType) const;

  static void confirmNonEmptyConcat(uint64_t nToCat);

  /** Tensor concatenation. */
  static Tensor concat(const std::vector<Tensor> &, uint64_t axis);
  static Tensor concat_(const std::vector<Tensor> &, uint64_t axis);

  static Shapes getShapes(const Tensors &tensors);

  /** A handle into implementation details. Returns false iff
   * this Tensor corresponds to an allocation, returns true if it is a
   * "reference" to one or several allocations.
   * */
  bool implIsView() const;
  bool implIsOrigin() const { return !implIsView(); }

  /**
   * Construct a Tensor from a const void pointer
   *
   * \param t The numerical type of the Tensor
   *
   * \param shape The Shape of the Tensor
   *
   * \param data The data to copy into an internal buffer for the Tensor
   *
   * \see The safer constructors which take typed pointers.
   * */
  static Tensor copy(DType t, const Shape &shape, const void *data);

  /**
   * Construct a Tensor from a scalar value. The Tensor will be of type \a
   * type, constructed from casting \a v.
   * */
  static Tensor scalar(DType type, double v);

  /**
   * Return the row-major contiguous data of this Tensor, as a char vector
   *
   * \see The methods which return vectors of specific numerical types.
   * */
  std::vector<char> getNativeCharVector() const;

  /**
   * Cast this Tensor to one of DType \a type
   * */
  Tensor to(DType type) const;

private:
  // get the BaseData for each Tensor in tIns.
  static std::vector<const BaseData *> getBaseDataPtrs(const Tensors &tIns);

  const BaseData &tData() const { return *tData_; }
  Tensor(const Shape &shape__,
         DType dtype__,
         std::shared_ptr<BaseData> tData__)
      : shape_(shape__), dtype_(dtype__), tData_(std::move(tData__)) {}

  template <typename T>
  static Tensor tRandomUniform(T low, T upp, const Shape &, uint32_t seed);
  template <typename T> static Tensor tArange(T x0, T x1, T step);
  template <typename T> static Tensor tScalar(T);

  template <typename T>
  static Tensor tMoveVector(const Shape &, std::vector<T> &&);
  template <typename T>
  static Tensor tCopyVector(const Shape &, const std::vector<T> &);
  template <typename T> static Tensor tCopyData(const Shape &, const T *);
  template <typename T> static Tensor tRefData(const Shape &, T *);
  template <typename T> Tensor binary(const Tensor &) const;

  Shape shape_;
  DType dtype_;
  std::shared_ptr<BaseData> tData_;

  void confirmValidReshape(const Shape &) const;

  class Caster;
  class ScalarCaster;
};

class OptionalTensor {

public:
  ~OptionalTensor() = default;
  OptionalTensor() : t(Tensor::int32(0)), isSet_(false) {}
  OptionalTensor(const Tensor &t_) : t(t_), isSet_(true) {}
  OptionalTensor(Tensor &&t_) : t(std::move(t_)), isSet_(true) {}
  OptionalTensor(const OptionalTensor &) = default;
  OptionalTensor(OptionalTensor &&)      = default;
  OptionalTensor &operator=(const OptionalTensor &) = default;
  OptionalTensor &operator=(OptionalTensor &&) = default;

  /** Return the Tensor t */
  const Tensor &value() const;
  bool has_value() const { return isSet_; }

private:
  Tensor t;
  bool isSet_;
};

std::ostream &operator<<(std::ostream &, const Tensor &);

/** Out of place elementwise binary operations on Tensors */
Tensor operator+(const Tensor &a, const Tensor &b);
Tensor operator-(const Tensor &a, const Tensor &b);
Tensor operator*(const Tensor &a, const Tensor &b);
Tensor operator/(const Tensor &a, const Tensor &b);
Tensor operator%(const Tensor &a, const Tensor &b);

Tensor concat(const std::vector<Tensor> &, uint64_t axis);
Tensor concat_(const std::vector<Tensor> &, uint64_t axis);

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
