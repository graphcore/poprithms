#ifndef POPRITHMS_OUTLINE_LINEAR_TENSOR_HPP
#define POPRITHMS_OUTLINE_LINEAR_TENSOR_HPP

#include <string>
#include <vector>

#include <poprithms/outline/linear/linearusings.hpp>

namespace poprithms {
namespace outline {
namespace linear {

class Tensor {
public:
  Tensor(const Shape &s, DType t, TensorId i, const std::string &d)
      : shape_(s), type_(t), id_(i), debugStr_(d) {}

  const Shape &shape() const { return shape_; }
  DType type() const { return type_; }
  TensorId id() const { return id_; }
  const std::string &debugStr() const { return debugStr_; }
  const std::vector<OpId> &ops() const { return ops_; }
  bool hasOp(OpId id) const {
    return std::find(ops_.cbegin(), ops_.cend(), id) != ops_.cend();
  }
  void insertOp(OpId id) { ops_.push_back(id); }
  void append(std::ostream &ost) const;

private:
  const Shape shape_;
  const DType type_;
  const TensorId id_;
  const std::string debugStr_;

  // producer (if any) and consumers of this Tensor
  std::vector<OpId> ops_;
};

std::ostream &operator<<(std::ostream &, const Tensor &);

} // namespace linear
} // namespace outline
} // namespace poprithms

#endif
