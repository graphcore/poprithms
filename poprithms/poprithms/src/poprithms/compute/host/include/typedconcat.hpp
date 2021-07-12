// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPRITHMS_COMPUTE_HOST_TYPEDCONCAT_HPP
#define POPRITHMS_COMPUTE_HOST_TYPEDCONCAT_HPP

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>

#include <compute/host/error.hpp>
#include <compute/host/include/basedata.hpp>
#include <compute/host/include/baseoperators.hpp>
#include <compute/host/include/ieeehalf.hpp>
#include <compute/host/include/viewdata.hpp>
#include <poprithms/compute/host/viewchange.hpp>
#include <poprithms/util/printiter.hpp>

namespace poprithms {
namespace compute {
namespace host {

// Referencing / aliasing concatenation.
class TypedConcat_ {
public:
  template <typename T>
  static BaseDataSP
  go(const ConstDataPtrs &datas, const Shapes &inShapes, uint64_t axis) {

    BaseData::assertForConcat(datas, inShapes);

    // For all Tensors, get a ViewData (reference) of the data.
    std::vector<BaseDataSP> nonContigs(datas.size());
    std::vector<ViewData<T> *> ptrs(datas.size());
    for (uint64_t i = 0; i < datas.size(); ++i) {
      nonContigs[i] = datas[i]->toViewData_();
      ptrs[i]       = dynamic_cast<ViewData<T> *>(nonContigs[i].get());
    }

    // Get all the unique OriginData origins, for all Tensors.
    std::vector<std::shared_ptr<const OriginData<T>>> allOriginDatas;
    for (auto ptr : ptrs) {
      for (auto x : ptr->origins()) {
        if (std::find(allOriginDatas.cbegin(), allOriginDatas.cend(), x) ==
            allOriginDatas.cend()) {
          allOriginDatas.push_back(x);
        }
      }
    }

    // Remap all ViewData BaseDatas to the set of unique OriginDatas.
    // This canonicalization is required so that the OriginDatas referred to
    // by the individual ViewDatas being concatenated, agree.
    for (auto ptr : ptrs) {
      ptr->remapOriginDatas(allOriginDatas);
    }

    // Concatenate.
    std::vector<const uint64_t *> ptrIndices_;
    std::vector<const int64_t *> ptrOffsets_;
    for (auto ptr : ptrs) {
      ptrIndices_.push_back(ptr->indices().data());
      ptrOffsets_.push_back(ptr->offsets().data());
    }
    auto concatIndices_ =
        ViewChange<uint64_t>::concat(ptrIndices_, inShapes, axis);
    auto concatOffsets_ =
        ViewChange<int64_t>::concat(ptrOffsets_, inShapes, axis);

    return std::make_shared<ViewData<T>>(
        allOriginDatas, std::move(concatIndices_), std::move(concatOffsets_));
  }
};

// Non-aliasing concatenation.
class TypedConcat {
public:
  template <typename T>
  static BaseDataSP
  go(const ConstDataPtrs &datas, const Shapes &inShapes, uint64_t axis) {

    auto asOriginData = [](auto &&x) {
      return dynamic_cast<const OriginData<T> *>(x);
    };

    BaseData::assertForConcat(datas, inShapes);

    // For the Tensors indices that are NOT row-contiguous,
    // a row-contiguous version will be stored here.
    std::vector<BaseDataSP> rms(datas.size());

    std::vector<const T *> ptrs(datas.size());

    for (uint64_t i = 0; i < datas.size(); ++i) {
      const auto tdp = datas[i];
      if (!tdp->isOriginData()) {
        rms[i]  = datas[i]->toOriginData();
        ptrs[i] = asOriginData(rms[i].get())->dataPtr();
      } else {
        ptrs[i] = asOriginData(datas[i])->dataPtr();
      }
    }

    return std::make_shared<AllocData<T>>(
        ViewChange<T>::concat(ptrs, inShapes, axis));
  }
};

} // namespace host
} // namespace compute
} // namespace poprithms

#endif
