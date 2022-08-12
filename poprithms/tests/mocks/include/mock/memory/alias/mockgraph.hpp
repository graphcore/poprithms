// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef MOCKS_POPRITHMS_MEMORY_ALIAS_GRAPH_HPP
#define MOCKS_POPRITHMS_MEMORY_ALIAS_GRAPH_HPP

#include <gmock/gmock.h>

#include <poprithms/memory/alias/graph.hpp>

namespace mock::poprithms::memory::alias {

class MockGraph {
public:
  MockGraph();
  virtual ~MockGraph();

  MOCK_METHOD(::poprithms::memory::alias::TensorId,
              allocate,
              (::poprithms::ndarray::Shape,
               ::poprithms::memory::alias::Color));
  MOCK_METHOD(::poprithms::memory::alias::TensorId,
              clone,
              (::poprithms::memory::alias::TensorId,
               ::poprithms::memory::alias::CloneColorMethod));
};

extern MockGraph *mockAliasGraph_;

inline ::poprithms::memory::alias::TensorId makeTensor(std::size_t id) {
  return ::poprithms::memory::alias::TensorId(id);
}

} // namespace mock::poprithms::memory::alias
#endif // MOCKS_POPRITHMS_MEMORY_ALIAS_GRAPH_HPP
