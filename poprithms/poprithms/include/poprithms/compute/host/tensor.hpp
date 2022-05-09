// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_TENSOR_HPP
#define POPRITHMS_COMPUTE_HOST_TENSOR_HPP

#include <memory>
#include <sstream>

#include <poprithms/ndarray/accessors.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>

namespace poprithms {
namespace compute {
namespace host {

using poprithms::ndarray::Dimension;
using poprithms::ndarray::Dimensions;
using poprithms::ndarray::Dims;
using poprithms::ndarray::DType;
using poprithms::ndarray::Ends;
using poprithms::ndarray::Shape;
using poprithms::ndarray::Shapes;
using poprithms::ndarray::Starts;
using poprithms::ndarray::Steps;
using poprithms::ndarray::Stride;
using poprithms::util::Permutation;
using NormalizedSliceParams = Shape::NormalizedSliceParams;
using Lower                 = Shape::Lower;
using Upper                 = Shape::Upper;

class Tensor;
using Tensors = std::vector<Tensor>;
class BaseData;

enum class CommutativeOp { Sum, Min, Max, Product };
std::ostream &operator<<(std::ostream &, CommutativeOp);
std::string str(CommutativeOp);

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
  static Tensor copyFloat64(const Shape &shape, const double *element0);

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
   * Create a Tensor, inferring its type from the template type T.
   * */
  template <typename T>
  static Tensor tensor(const Shape &s, std::vector<T> &&values) {
    return tMoveVector<T>(s, std::move(values));
  }

  /**
   * Create a Tensor, inferring its type from the template type T.
   * */
  template <typename T>
  static Tensor tensor(const Shape &s, const std::vector<T> &values) {
    return tCopyVector<T>(s, values);
  }

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
   * generators are used and no C++ distributions.
   * */
  static Tensor
  uniformFloat64(double low, double upp, const Shape &, uint32_t seed);

  /**
   * Return a sorted (low to high) rank-1 Tensor, with N values in the range
   * [0, range). The probability that i in [0, range) appears in the returned
   * Tensor is N/range, and probabilities are independent for all i. An error
   * is thrown if N > range.
   * */
  static Tensor sampleWithoutReplacementUnsigned64(uint64_t range,
                                                   uint64_t N,
                                                   uint32_t seed);

  /**
   * Return a Tensor of shape #s of type Unsigned64, with values in the range
   * [0, range). The values do not repeat if #r is Replacement::No.
   * */
  enum class Replacement { No = 0, Yes };
  static Tensor sampleUnsigned64(Replacement r,
                                 const Shape &s,
                                 uint64_t range,
                                 uint64_t seed);

  /**
   * Return a Tensor of type #t and of Shape #s, with #nUnmasked 1s, and all
   * the remaining values 0. An error is thrown if #nUnmasked exceeds the
   * size of #s.
   * */
  static Tensor
  mask(DType t, const Shape &s, uint64_t nUnmasked, uint32_t seed);

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
   * \return The #rowMajorIndex'th value in the tensor, cast to a double
   *         value.
   * */
  double getFloat64(uint64_t rowMajorIndex) const;

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
  static Tensor copyFloat32(const Shape &, const float *);
  static Tensor float32(const Shape &, const std::vector<float> &);
  static Tensor float32(const Shape &, std::vector<float> &&);
  static Tensor float32(float);
  static Tensor refFloat32(const Shape &, float *);
  static Tensor
  uniformFloat32(float low, float upp, const Shape &, uint32_t seed);
  static Tensor arangeFloat32(float start, float stop, float step);
  Tensor toFloat32() const;
  std::vector<float> getFloat32Vector() const;
  float getFloat32(uint64_t rowMajorIndex) const;

  /**
   * Float16 specific Tensor methods. \see corresponding Float64 methods.
   */
  static Tensor copyFloat16(const Shape &, const uint16_t *);
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
  static Tensor copyInt64(const Shape &, const int64_t *);
  static Tensor int64(const Shape &, const std::vector<int64_t> &);
  static Tensor int64(const Shape &, std::vector<int64_t> &&);
  static Tensor int64(int64_t);
  static Tensor refInt64(const Shape &, int64_t *);
  static Tensor arangeInt64(int64_t start, int64_t stop, int64_t step);
  Tensor toInt64() const;
  std::vector<int64_t> getInt64Vector() const;
  int64_t getInt64(uint64_t rowMajorIndex) const;

  /**
   * Create a Tensor of type Int64 with values drawn independently from the
   * the range [low, upp). The value upp is not sampled.
   *
   * Note that the random values drawn are platform invariant, as only C++
   * generators are used and no C++ distributions.
   *
   * It is required that low < upp.
   * */
  static Tensor
  randomInt64(int64_t low, int64_t upp, const Shape &, uint32_t seed);

  /**
   * Unsigned64 specific Tensor methods and factory functions.
   *
   * \see The corresponding Float64 methods.
   * */
  static Tensor copyUnsigned64(const Shape &, const uint64_t *);
  static Tensor unsigned64(const Shape &, const std::vector<uint64_t> &);
  static Tensor unsigned64(const Shape &, std::vector<uint64_t> &&);
  static Tensor unsigned64(uint64_t);
  static Tensor refUnsigned64(const Shape &, uint64_t *);
  static Tensor
  arangeUnsigned64(uint64_t start, uint64_t stop, uint64_t step);
  Tensor toUnsigned64() const;
  std::vector<uint64_t> getUnsigned64Vector() const;
  uint64_t getUnsigned64(uint64_t rowMajorIndex) const;
  static Tensor
  randomUnsigned64(uint64_t low, uint64_t upp, const Shape &, uint32_t seed);

  /** Int32 type specific Tensor methods and factory functions.
   *
   * \see the corresponding Float64 methods.
   * */
  static Tensor copyInt32(const Shape &, const int32_t *);
  static Tensor int32(const Shape &, const std::vector<int32_t> &);
  static Tensor int32(const Shape &, std::vector<int32_t> &&);
  static Tensor int32(int32_t);
  static Tensor refInt32(const Shape &, int32_t *);
  static Tensor arangeInt32(int32_t start, int32_t stop, int32_t step);
  Tensor toInt32() const;
  std::vector<int32_t> getInt32Vector() const;
  int32_t getInt32(uint64_t rowMajorIndex) const;
  static Tensor randomInt32(int low, int upp, const Shape &, uint32_t seed);

  /** \return A Tensor of zeros, of type \a d and shape \a s. */
  static Tensor zeros(DType d, const Shape &s);

  /** \return A Tensor of ones, of type \a d and shape \a s. */
  static Tensor ones(DType, const Shape &);

  /** Unsigned32 type specific Tensor methods and factory functions.
   *
   * \see the corresponding Float64 methods.
   * */
  static Tensor copyUnsigned32(const Shape &, const uint32_t *);
  static Tensor unsigned32(const Shape &, const std::vector<uint32_t> &);
  static Tensor unsigned32(const Shape &, std::vector<uint32_t> &&);
  static Tensor unsigned32(uint32_t);
  static Tensor refUnsigned32(const Shape &, uint32_t *);
  static Tensor
  arangeUnsigned32(uint32_t start, uint32_t stop, uint32_t step);
  Tensor toUnsigned32() const;
  std::vector<uint32_t> getUnsigned32Vector() const;
  uint32_t getUnsigned32(uint64_t rowMajorIndex) const;
  static Tensor
  randomUnsigned32(uint32_t low, uint32_t upp, const Shape &, uint32_t seed);

  /** Int16 type specific Tensor methods and factory functions.
   *
   * \see the corresponding Float64 methods.
   * */
  static Tensor copyInt16(const Shape &, const int16_t *);
  static Tensor int16(const Shape &, const std::vector<int16_t> &);
  static Tensor int16(const Shape &, std::vector<int16_t> &&);
  static Tensor int16(int16_t);
  static Tensor refInt16(const Shape &, int16_t *);
  static Tensor arangeInt16(int16_t start, int16_t stop, int16_t step);
  Tensor toInt16() const;
  std::vector<int16_t> getInt16Vector() const;
  int16_t getInt16(uint64_t rowMajorIndex) const;
  static Tensor
  randomInt16(int16_t low, int16_t upp, const Shape &, uint32_t seed);

  /** Unsigned16 type specific Tensor methods and factory functions.
   *
   * \see the corresponding Float64 methods.
   * */
  static Tensor copyUnsigned16(const Shape &, const uint16_t *);
  static Tensor unsigned16(const Shape &, const std::vector<uint16_t> &);
  static Tensor unsigned16(const Shape &, std::vector<uint16_t> &&);
  static Tensor unsigned16(uint16_t);
  static Tensor refUnsigned16(const Shape &, uint16_t *);
  static Tensor
  arangeUnsigned16(uint16_t start, uint16_t stop, uint16_t step);
  Tensor toUnsigned16() const;
  std::vector<uint16_t> getUnsigned16Vector() const;
  uint16_t getUnsigned16(uint64_t rowMajorIndex) const;
  static Tensor
  randomUnsigned16(uint16_t low, uint16_t upp, const Shape &, uint32_t seed);

  /** Int8 type specific Tensor methods and factory functions.
   *
   * \see the corresponding Float64 methods.
   * */
  static Tensor copyInt8(const Shape &, const int8_t *);
  static Tensor int8(const Shape &, const std::vector<int8_t> &);
  static Tensor int8(const Shape &, std::vector<int8_t> &&);
  static Tensor int8(int8_t);
  static Tensor refInt8(const Shape &, int8_t *);
  static Tensor arangeInt8(int8_t start, int8_t stop, int8_t step);
  Tensor toInt8() const;
  std::vector<int8_t> getInt8Vector() const;
  int8_t getInt8(uint64_t rowMajorIndex) const;
  static Tensor
  randomInt8(int8_t low, int8_t upp, const Shape &, uint32_t seed);

  /** Unsigned8 type specific Tensor methods and factory functions.
   *
   * \see the corresponding Float64 methods.
   * */
  static Tensor copyUnsigned8(const Shape &, const uint8_t *);
  static Tensor unsigned8(const Shape &, const std::vector<uint8_t> &);
  static Tensor unsigned8(const Shape &, std::vector<uint8_t> &&);
  static Tensor unsigned8(uint8_t);
  static Tensor refUnsigned8(const Shape &, uint8_t *);
  static Tensor arangeUnsigned8(uint8_t start, uint8_t stop, uint8_t step);
  Tensor toUnsigned8() const;
  std::vector<uint8_t> getUnsigned8Vector() const;
  uint8_t getUnsigned8(uint64_t rowMajorIndex) const;
  static Tensor
  randomUnsigned8(uint8_t low, uint8_t upp, const Shape &, uint32_t seed);

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
  bool getBoolean(uint64_t rowMajorIndex) const;
  static Tensor randomBoolean(const Shape &, uint32_t seed);

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
   *
   * Either this Tensor must numpy-domintate \a rhs, or \a rhs must
   * numpy-dominate this Tensor.
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

  /**
   * \return false if there are 2 elements in this Tensor which do not have
   *         exactly the same value. Otherwise return true.
   * */
  bool allValuesTheSame() const;

  DType dtype() const { return dtype_; }
  const Shape &shape() const { return shape_; }
  uint64_t rank_u64() const { return shape().rank_u64(); }
  uint64_t nelms_u64() const { return shape().nelms_u64(); }
  int64_t nelms() const { return shape().nelms(); }
  uint64_t dim(uint64_t d) const { return shape().dim(d); }
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
   * Return string such as
   *
   *  """   [[ 1 2 ]
   *         [ 3 4 ]]  """
   *
   * containing the values of this Tensor.
   *
   * See also the method \a append, which additionally adds information about
   * the Tensor (shape, type, allocation type, etc.)
   * */
  std::string values() const;

  /**
   * \return The value in this Tensor at index #rowMajorIndex, cast to a
   *         string.
   * */
  std::string valueAsStr(uint64_t rowMajorIndex) const;

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
   * Reshape by collapsing the dimensions in the the interval [i0, i1) into a
   * single dimension. \sa Shape::flatten.
   * */
  Tensor flatten(uint64_t i0, uint64_t i1) const;
  Tensor flatten_(uint64_t i0, uint64_t i1) const;

  /**
   * \return Reshape this Tensor by removing all dimensions which have size
   *         `1'. Note that `0's are not removed.
   * */
  Tensor squeeze() const { return reshape(shape().squeeze()); }
  Tensor squeeze_() const { return reshape_(shape().squeeze()); }

  /**
   * \return A scalar (rank-0) Tensor created from a single element of this
   *         Tensor (no aliasing).
   * */
  Tensor scalarFromElement(uint64_t rowMajorIndex) const;

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

  Tensor prependOnesReshape(uint64_t nOnes) const;
  Tensor prependOnesReshape_(uint64_t nOnes) const;

  /**
   * Expand the Tensor using numpy broadcasting rules.
   * https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html
   * */
  Tensor expand(const Shape &) const;
  Tensor expand_(const Shape &) const;

  /**
   * Take a slice of this Tensor between bounds \a l (inclusive) and \a u
   * (exclusive). \a l and \a u must have size equal to the rank of this
   * Tensor, and for all dimensions d, it is required that
   *     0 <= l[d] <= u[d] <= dim(d).
   * */
  Tensor slice(const Lower &l, const Upper &u) const;
  Tensor slice_(const Lower &l, const Upper &u) const;

  /**
   * Slice this Tensor in a single dimensions.
   * */
  Tensor slice(Dimension, uint64_t l, uint64_t u) const;
  Tensor slice_(Dimension, uint64_t l, uint64_t u) const;

  /**
   * Slice this Tensor in a multiple dimensions.
   *
   * \param dims unique and sorted dimensions of this Tensor to slice in
   *
   * \param l the lower bounds of the slice for each dimension in #dims,
   *          respectively.
   *
   * \param u the upper bounds of the slice for each dimension in #dims,
   *          respectively.
   *
   * dims, l, and u should all be of the same size.
   * */
  Tensor slice(const Dimensions &dims,
               const std::vector<uint64_t> &l,
               const std::vector<uint64_t> &u) const;
  Tensor slice_(const Dimensions &dims,
                const std::vector<uint64_t> &l,
                const std::vector<uint64_t> &u) const;

  /**
   * Slice this Tensor in dimension 0, between d and d+1. This Tensor must be
   * of strictly positive rank. The returned Tensor has rank 1 lower than this
   * Tensor. For example, if this Tensor is of shape (3,4,5). then at(1) has
   * shape (4,5). Specifically, t.at(d) is the same as
   * <code>
   *    t.slice(Dimension(0), d, d+1).squeeze({0});
   * </code>
   * */
  Tensor at(uint64_t d) const;

  /**
   * \param index The index in dimension 0 to slice in. index must be a
   *              scalar, unsigned integer type.
   *
   * \sa at(uint64_t).
   *
   * */
  Tensor at(const Tensor &index) const;

  /**
   * The inplace equivalent of at(uint64_t), returning a reference to some
   * elements of this Tensor.
   * */
  Tensor at_(uint64_t d) const;

  /**
   * The inplace equivalent of at(const Tensor &), this method returns a
   * reference to a slice of this Tensor.
   * */
  Tensor at_(const Tensor &index) const;

  /**
   * Reduction methods.
   *
   * The values in this Tensor are accumulated along singleton dimensions of
   * the output Shape, \a outShape.
   *
   * \param outShape The output Shape. It must be numpy broadcastable to this
   *                 Tensor's Shape. For example, if this Tensor has Shape
   *                 (2,3,5), then valid output Shapes are (2,1,5), (3,1),
   *                 (5,), (), etc. (2,3) and (2,5) are invalid Shapes.
   *
   * */
  Tensor reduceSum(const Shape &outShape) const;
  Tensor reduceMin(const Shape &outShape) const;
  Tensor reduceMax(const Shape &outShape) const;
  Tensor reduceProduct(const Shape &outShape) const;
  Tensor reduce(const Shape &, CommutativeOp) const;

  /**
   * Reduction methods, where the reduction is across all elements of this
   * Tensor, and the returned Tensor is rank-0.
   * */
  Tensor reduceSum() const { return reduceSum({}); }
  Tensor reduceMin() const { return reduceMin({}); }
  Tensor reduceMax() const { return reduceMax({}); }
  Tensor reduceProduct() const { return reduceProduct({}); }
  Tensor reduce(CommutativeOp cop) const { return reduce({}, cop); }

  /**
   * \return the l2-norm of this Tensor. That is, the square root of the sum
   *         of the squares of the values. */
  double l2norm() const;

  /**
   * Reduce a set of Tensors of the same size, using a particular commutative
   * operation.
   * */

  /**
   * Reduce a set of Tensors of the same size, using a particular commutative
   * operation. The result is stored in the first Tensor, an alias of which is
   * returned.
   * */
  static Tensor accumulate(const Tensors &ts, CommutativeOp);
  static Tensor accumulate_(const Tensors &ts, CommutativeOp);

  /**
   * Numpy-style [start:stop:step] slicing,
   * https://numpy.org/doc/stable/reference/arrays.indexing.html
   *
   * \param starts Indices at which to start the slice. These are for all
   *               dimensions in \a dims, unless \a dims is empty in which
   *               case it is just all dimensions. Starting values which are
   *               not in [0, dimSize) are canonicalized as:
   *
   * canonicalize(start):
   *     if (start < 0):        start += dimSize.
   *     if (start < 0):        start = 0.
   *     if (start >= dimSize): start = dimSize - 1.
   *     return start
   *
   * \param ends Indices at which to end (not including) the slice, for all
   *             dimensions in \a dims if \a dims is non-empty, else all
   *             dimensions. \a ends and \a starts must be the same size.
   *             Ending values which are not in [-1, dimSize] are
   *             canonicalized:
   *
   * canonicalize(end):
   *     if (end < 0):       end += dimSize.
   *     if (end < 0):       end = -1.
   *     if (end > dimSize): end = dimSize.
   *     return end
   *
   * Note that the canonicalization of start and end are slightly different,
   * as the start index is inclusive while end index is exclusive. This means
   * that canonicalized starts must lie in [0, dimSize) while ends must lie in
   * [-1, dimSize + 1). This is in agreement with the numpy spec.
   *
   * \param steps The stride to take when slicing from values in \a starts to
   *              \a ends. Defaults to +1. Values in \a steps must be
   *              non-zero.
   *
   * \param dims The dimensions along which to slice. If empty, then slicigin
   *             is in all dimensions, and \a starts and \a ends must of full
   *             rank.
   *
   * Some examples. Suppose this Tensor is
   *      [[ 0 1 2 3 ]
   *       [ 4 5 6 7 ]]
   *
   * starts=(0,0), ends=(2,3), steps=(), dims=():
   *      [[ 0 1 2 ]
   *       [ 3 5 6 ]]
   *
   * starts=(0), ends=(4), steps=(2), dims=(1)
   *      [[ 0 2 ]
   *       [ 4 6 ]]
   *
   * starts=(3,1), ends=(-10,2), steps=(-1,1), dims=(1,0)
   *      [[ 7 6 5 4 ]]
   *
   * \sa https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice
   *     which is also identical to this method.
   *
   * */
  Tensor slice(const Starts &starts,
               const Ends &ends,
               const Steps &steps,
               const Dims &dims) const;
  Tensor
  slice_(const Starts &, const Ends &, const Steps &, const Dims &) const;

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

  /**
   * Gather along all dimensions of this Tensor. This is equivalent to looping
   * over the vectors in \a where and applying the 1-d version of gather,
   * above.
   * */
  Tensor gather(const std::vector<std::vector<int64_t>> &where) const;
  Tensor gather_(const std::vector<std::vector<int64_t>> &where) const;

  /**
   * Scatter all the values in this Tensor into a Tensor of zeros, of Shape \a
   * outShape. The positions where this Tensor's values are scattered to are
   * defined by \a where. Specifically, the offsets in dimension \a d which
   * are scattered to are \a where[d]. As an example, if this Tensor is
   *
   * [[ 0 1 2 ]
   *    3 4 5 ]],
   *
   * and outShape is (3, 4), and
   *
   * \a where is ((0,2), (0,2,3)), then the returned Tensor is
   *
   *               0 . 2 3
   *               |   | |
   *               |   | |
   *    0 ----- [[ 0 0 1 2 ]
   *    .        [ 0 0 0 0 ]
   *    2 -----  [ 3 0 4 5 ]].
   *
   * */
  Tensor scatterToZero(const Shape &outShape,
                       const std::vector<std::vector<int64_t>> &where) const;

  /**
   * Scatter the values in this Tensor into the Tensor \a target. This is the
   * same as the method \a scatterToZero, except instead of having zeros in
   * the positions in the output Tensor which are not scattered to, the values
   * are selected from \a target.
   *
   * All offsets in \a where[d] must be less than \a target.dim(d).
   * */
  Tensor scatterTo(const Tensor &target,
                   const std::vector<std::vector<int64_t>> &where) const;

  /**
   * Create a boolean mask, which is true at position (p_0, ... p_{N-1}) if
   * p_0 is in \a whereTrue[0] and,
   * p_1 is in \a whereTrue[1] and,
   *    .
   *    .
   * p_{N-1} is in \a whereTrue[N-1].
   *
   * Example: shape = (2,5), and where = ((1), (0,2,4)) produces the mask
   *
   *  [[ 0 0 0 0 0 ]
   *   [ 1 0 1 0 1 ]].
   * */
  static Tensor
  scatterMask(const Shape &shape,
              const std::vector<std::vector<int64_t>> &whereTrue);

  /** A generalization of a matrix transpose. */
  Tensor dimShuffle(const Permutation &) const;
  Tensor dimShuffle_(const Permutation &) const;

  /**
   * Roll dimension \a dimIdx to the dimension \a newIdx
   *
   * The other dimensions remain in the same relative order
   *
   *  \param dimIdx     The dimension to move from
   *  \param newIdx     The dimension to move to
   *  \returns          The shuffled Tensor
   *  */
  Tensor dimRoll(Dimension dimIdx, Dimension newIdx) const;
  Tensor dimRoll_(Dimension dimIdx, Dimension newIdx) const;

  /**
   * Resize, or tile, this Tensor in a certain Dimension.
   *
   * As an example, if this Tensor is
   *   [[ 0 1 ]
   *    [ 2 3 ]],
   *
   * then resizing in Dimension \a d = 0 (horizontal) with Stride \a s = 2
   * results in
   *   [[ 0 0 1 1 ]
   *    [ 2 2 3 3 ]].
   *
   * */
  Tensor resize(Dimension d, Stride s) const;
  Tensor resize_(Dimension d, Stride s) const;

  /**
   * Resize the final dimension of this Tensor.
   * */
  Tensor resizeFinalDim(Stride) const;
  Tensor resizeFinalDim_(Stride) const;

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
   * The versions with the suffix _ are inplace on this Tensor.
   *
   * Note that there is no implicit type casting, so \p rhs must have the same
   * numerical type as this Tensor.
   * */
  Tensor add(const Tensor &rhs) const;
  Tensor add_(const Tensor &rhs) const;

  /**
   * Elementwise binary operations with a scalar. The returned Tensor is of
   * the same type and Shape as this Tensor. If the value v cannot be
   * represented exactly as this Tensor's type, an error is thrown. For
   * example, if this Tensor is of type Int32, then add(1.5) will result in an
   * error.
   * */
  Tensor add(double v) const { return add(safeScalar(dtype(), v)); }
  Tensor add_(double v) const { return add_(safeScalar(dtype(), v)); }

  Tensor mul(const Tensor &rhs) const;
  Tensor mul_(const Tensor &rhs) const;
  Tensor mul(double v) const { return mul(safeScalar(dtype(), v)); }
  Tensor mul_(double v) const { return mul_(safeScalar(dtype(), v)); }

  Tensor max(const Tensor &rhs) const;
  Tensor max_(const Tensor &rhs) const;
  Tensor max(double v) const { return max(safeScalar(dtype(), v)); }
  Tensor max_(double v) const { return max_(safeScalar(dtype(), v)); }

  Tensor min(const Tensor &rhs) const;
  Tensor min_(const Tensor &rhs) const;
  Tensor min(double v) const { return min(safeScalar(dtype(), v)); }
  Tensor min_(double v) const { return min_(safeScalar(dtype(), v)); }

  Tensor combine(const Tensor &, CommutativeOp) const;
  Tensor combine_(const Tensor &, CommutativeOp) const;

  Tensor subtract(const Tensor &rhs) const;
  Tensor subtract_(const Tensor &rhs) const;
  Tensor subtract(double v) const { return subtract(safeScalar(dtype(), v)); }
  Tensor subtract_(double v) const {
    return subtract_(safeScalar(dtype(), v));
  }

  Tensor divide(const Tensor &rhs) const;
  Tensor divide_(const Tensor &rhs) const;
  Tensor divide(double v) const { return divide(safeScalar(dtype(), v)); }
  Tensor divide_(double v) const { return divide_(safeScalar(dtype(), v)); }

  Tensor mod(const Tensor &rhs) const;
  Tensor mod_(const Tensor &rhs) const;

  Tensor pow(const Tensor &rhs) const;
  Tensor pow_(const Tensor &rhs) const;
  Tensor pow(double v) const { return pow(safeScalar(dtype(), v)); }
  Tensor pow_(double v) const { return pow_(safeScalar(dtype(), v)); }

  /**
   * Set the value of this Tensor to \a rhs.
   *
   * <code>
   *    // Three equivalent ways to update the values in a Tensor t:
   *    t.copyFrom_(rhs);
   *    t.update_(rhs);
   *    t.zeroAll_().add_(rhs);
   * </code>
   *
   * \param rhs The values to update this Tensor to. The shape of \a rhs must
   *            be expandable to the shape of this Tensor. For example, if
   *            this Tensor has shape (4,1,3), then some valid shapes for rhs
   *            are (), (3), and (4,1,1). Two invalid shapes are (4,3), and
   *            (4,5,3).
   * */
  Tensor copyFrom_(const Tensor &rhs) const;
  Tensor update_(const Tensor &rhs) const { return copyFrom_(rhs); }

  /**
   * Update a slice of this Tensor.
   *
   * \param updater A smaller Tensor than this Tensor, its values will replace
   *                a slice of this Tensor. It is of the same rank as this
   *                Tensor.
   *
   * \param dims The dimensions in which \a updater is potentially smaller
   * than this Tensor is.
   *
   * \param offset The indices in \a dims at which to start the replacement
   *               region in this Tensor.
   * */
  Tensor updatePart_(const Tensor &updater,
                     const Dimensions &dims,
                     const std::vector<uint64_t> &offset) const;

  Tensor dynamicUpdate_(const Tensor &updater,
                        const Dimensions &dims,
                        const std::vector<uint64_t> &offset) const {
    return updatePart_(updater, dims, offset);
  }

  /**
   * Set this 2-d Tensor to be sparse, with exactly one 1 in each row. If this
   * Tensor is of Shape (N, C), then indices is of length N, and has values in
   * the range [0, C).
   *
   * Example. This Tensor is of Shape (3,4) and indices is {0,1,0}, then this
   * Tensor is updated to,
   *
   *       [[ 1 0 0 0 ]
   *        [ 0 1 0 0 ]
   *        [ 1 0 0 0 ]].
   * */
  Tensor encodeOneHot_(const std::vector<uint64_t> &indices) const;

  /**
   * Add \a v to all elements of this Tensor.
   * */
  Tensor increment_(int64_t v) const;
  Tensor increment(int64_t v) const;

  /** \return True for all of strings [Pow, Mod, Add, Sub, Subtract, Div,
   *          Divide, Mul, Multiply] and for of their case variants (pow and
   *          POW are valid too, for example).
   **/
  static bool isBinary(const std::string &);
  static void assertIsBinary(const std::string &);

  /**
   * Perform matrix multiplication with \a rhs, using numpy v1.19 broadcasting
   * rules: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
   *
   * Note that this method should not be used in performance critical code, as
   * the implementation is not optimized.
   * */
  Tensor matmul(const Tensor &rhs) const;

  /**
   * Perform the binary operation described by \a type. For example, if type
   * is "Add", then this Tensor will be added to \a arg1.
   *
   * \param type The binary operation to perform. It must have isBinary(type)
   *             true.
   *
   * \param arg1 The right hand side argument of the binary operation.
   * */

  Tensor binary(const std::string &type, const Tensor &arg1) const;

  /** Inplace version of binary */
  Tensor binary_(const std::string &, const Tensor &arg1) const;

  /** These elementwise binary operations return Tensors of type Boolean. */
  Tensor operator<(const Tensor &rhs) const;
  Tensor operator<=(const Tensor &rhs) const;
  Tensor operator>(const Tensor &rhs) const;
  Tensor operator>=(const Tensor &rhs) const;
  Tensor operator==(const Tensor &rhs) const;

  /**
   * Elementwise unary operations. The versions with the suffix '_' acts
   * inplace on the memory of this Tensor.
   * */
  Tensor abs() const;
  Tensor abs_() const;

  /**
   * \return a Tensor if the same type as this tensor, which have value +1 if
   *         it negative, 0 if it is 0, and +1 if it is positive.
   * */
  Tensor sign() const;

  /**
   * e (2.71828...) to the power of this Tensor. This method is only available
   * to floating point Tensors.
   * */
  Tensor exp() const;
  Tensor exp_() const;

  /**
   * log base e (2.71828...), also known as the natural logarithm, of this
   * Tensor. This method is only available to floating point Tensors.
   * */
  Tensor log() const;
  Tensor log_() const;

  Tensor ceil() const;
  Tensor ceil_() const;

  Tensor floor() const;
  Tensor floor_() const;

  Tensor mod(int64_t modulo) const;
  Tensor mod_(int64_t modulo) const;

  /**
   * The negative of this Tensor.
   * */
  Tensor neg() const;
  Tensor neg_() const;

  /**
   * The square root of this Tensor. This method is only available to floating
   * point Tensors.
   * */
  Tensor sqrt() const;
  Tensor sqrt_() const;

  /**
   * The reciprocal of this Tensor.
   * */
  Tensor reciprocal() const;
  Tensor reciprocal_() const;

  /**
   * relu(x) = x*(x > 0)
   * */
  Tensor relu() const;
  Tensor relu_() const;

  Tensor sin() const;
  Tensor sin_() const;

  Tensor cos() const;
  Tensor cos_() const;

  /**
   * Set all values in this Tensor to 0.
   * */
  Tensor zeroAll_() const;

  /**
   * The non-inplace equivalent of zeroAll_, this method returns a new Tensor
   * of zeros with shape and type derived from this Tensor.
   * */
  Tensor zeros() const;

  /**
   * \return true: if and only if (iff) this and \a rhs have the same shape,
   *         same type, and some data addresses. Note that this does no
   *         numerical comparison of Tensors, and so 2 Tensors which are
   *         numerically identical will not compare equal unless their datas
   *         have the same addresses.
   */
  bool identicalTo(const Tensor &rhs) const;

  /**
   * \return true iff rhs has the same shape, same type, and same values as
   *         this Tensor. This is a weaker equivalence condition than
   *         #identicalTo, where the data addresses must be the same, not just
   *         the values.
   * */
  bool numericallyIdenticalTo(const Tensor &rhs) const;

  /**
   * Throw an error with a descriptive message if the DType argument differs
   * from this Tensor's type. Recall that DType is the numerical type. */
  void assertType(DType) const;

  static void assertNonEmptyConcat(uint64_t nToCat);

  /**
   * Throw an error with a descriptive message if this Tensor does (not)
   * contain aliases.
   * */
  void assertContainsAliases() const;
  void assertContainsNoAliases() const;

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
   * type, constructed from casting \a v. If the type is known at compile
   * time, then the native Tensor constructors should be used. For example,
   * Tensor::float32(1.5) is preferable to
   * Tensor::scalar(DType::Float32, 1.5).
   * */
  static Tensor scalar(DType type, double v);

  /**
   * As per the method 'scalar' but checks are run that #v is a valid #type.
   * This will become the default behaviour in a later release.
   * */
  static Tensor safeScalar(DType type, double v);

  Tensor scalarOfSameType(double v) const { return scalar(dtype(), v); }

  /**
   * \return The row-major contiguous data of this Tensor, as a char vector
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
  template <typename T>
  static Tensor tRandomInt(T low, T upp, const Shape &, uint32_t seed);
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

  // Grant acceess to the Serializer class which uses boost::serialization to
  // serialize Tensors. Boost handles the tricky task of serializing
  // shared_ptrs, polymorphic base pointers, etc.
  friend class Serializer;

  void assertValidReshape(const Shape &) const;

  /** Verify that the values in this Tensor can be scattered into a Tensor of
   * Shape \a out, at positions \a where. */
  void
  assertValidScatter(const Shape &out,
                     const std::vector<std::vector<int64_t>> &where) const;

  class Caster;
  class ScalarCaster;
  class Zeros;
  class Ones;

  void assertContainsAliases(bool) const;

public:
  /**
   * Certain backends require this cast to void *. We don't advise doing this
   * unless it is strictly required.
   *
   * If the underlying data is not contiguous, an error is thrown. Otherwise,
   * a void pointer to the #rowMajorIndex'th element is returned.
   * */
  void *getPtrToOriginData(uint64_t rowMajorIndex = 0) const;
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

Tensor scalar(DType t, double v);

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
