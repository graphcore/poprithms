// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_OUTLINE_LINEAR_OP_HPP
#define POPRITHMS_OUTLINE_LINEAR_OP_HPP

#include <algorithm>
#include <string>
#include <vector>

#include <poprithms/outline/linear/linearusings.hpp>

namespace poprithms {
namespace outline {
namespace linear {

class Op {

public:
  Op(Color c, OpId i, Type t, const std::string &d)
      : color_(c), id_(i), type_(t), debugStr_(d) {}

  Color color() const { return color_; }
  OpId id() const { return id_; }
  Type type() const { return type_; }
  const std::string &debugStr() const { return debugStr_; }

  bool hasOpIn(OpId x) const {
    return std::find(opI_.cbegin(), opI_.cend(), x) != opI_.cend();
  }
  void insertOpIn(OpId x) { opI_.push_back(x); }
  uint64_t nOpsIn() const { return opI_.size(); }
  const std::vector<OpId> &getOpsIn() const { return opI_; }

  bool hasOpOut(OpId x) const {
    return std::find(opO_.cbegin(), opO_.cend(), x) != opO_.cend();
  }
  uint64_t nOpsOut() const { return opO_.size(); }
  void insertOpOut(OpId x) { opO_.push_back(x); }
  const std::vector<OpId> &getOpsOut() const { return opO_; }

  void insertIn(TensorId x, InIndex i);
  void insertOut(TensorId x, OutIndex i);

  void append(std::ostream &) const;

private:
  const Color color_;
  const OpId id_;
  const Type type_;
  const std::string debugStr_;

  // Store topological constraints (a.k.a. control dependencies) here.
  std::vector<OpId> opI_;
  std::vector<OpId> opO_;

  // Store the Tensor inputs and outputs here.
  std::vector<TensorId> ins_;
  std::vector<TensorId> outs_;
}; // namespace linear

std::ostream &operator<<(std::ostream &, const Op &);

} // namespace linear
} // namespace outline
} // namespace poprithms

#endif
