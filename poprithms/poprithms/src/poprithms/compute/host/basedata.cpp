// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "./include/basedata.hpp"

#include "./include/allocdata.hpp"
#include "./include/basedata.hpp"
#include "./include/externdecl.hpp"
#include "./include/pointerdata.hpp"
#include "./include/typedconcat.hpp"
#include "./include/typeswitch.hpp"
#include "./include/viewdata.hpp"

#include <poprithms/ndarray/dtype.hpp>

namespace poprithms {
namespace compute {
namespace host {

std::ostream &operator<<(std::ostream &ost, const BaseData &d) {
  d.append(ost);
  return ost;
}

void BaseData::assertSameTypes(const ConstDataPtrs &datas) {
  if (datas.empty()) {
    return;
  }
  for (auto ptr : datas) {
    if (ptr->dtype() != datas[0]->dtype()) {
      std::ostringstream oss;
      oss << "Failed in BaseData::assertSameTypes with types=( ";
      for (const auto x : datas) {
        oss << x->dtype() << ' ';
      }
      oss << ')';
      throw error(oss.str());
    }
  }
}

void BaseData::assertForConcat(const ConstDataPtrs &datas,
                               const Shapes &inShapes) {
  if (inShapes.size() != datas.size()) {
    throw error("Failure in BaseData::assertForConcat, Shapes and "
                "BaseDatas must have same sizes.");
  }
  BaseData::assertSameTypes(datas);
  Tensor::confirmNonEmptyConcat(datas.size());
}

// Aliasing concatenation
BaseDataSP BaseData::concat_(const ConstDataPtrs &datas,
                             const Shapes &shapes,
                             uint64_t axis) {
  Tensor::confirmNonEmptyConcat(datas.size());
  return typeSwitch<TypedConcat_, BaseDataSP>(
      datas[0]->dtype(), datas, shapes, axis);
}

BaseDataSP BaseData::concat(const ConstDataPtrs &datas,
                            const Shapes &shapes,
                            uint64_t axis) {
  Tensor::confirmNonEmptyConcat(datas.size());
  return typeSwitch<TypedConcat, BaseDataSP>(
      datas[0]->dtype(), datas, shapes, axis);
}

} // namespace host
} // namespace compute
} // namespace poprithms