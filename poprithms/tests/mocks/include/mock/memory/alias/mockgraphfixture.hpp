// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef MOCKS_POPRITHMS_MEMORY_ALIAS_GRAPH_FIXTURE_HPP
#define MOCKS_POPRITHMS_MEMORY_ALIAS_GRAPH_FIXTURE_HPP

#include <mock/memory/alias/mockgraph.hpp>

namespace mock::poprithms::memory::alias {

template <template <typename> typename Mock = ::testing::StrictMock>
class MockGraphFixture {
public:
  MockGraphFixture() {
    mock::poprithms::memory::alias::mockAliasGraph_ =
        static_cast<mock::poprithms::memory::alias::MockGraph *>(
            &mockAliasGraph);
  }

  ~MockGraphFixture() {
    mock::poprithms::memory::alias::mockAliasGraph_ = nullptr;
  }

protected:
  Mock<MockGraph> mockAliasGraph;
};

} // namespace mock::poprithms::memory::alias

#endif // MOCKS_POPRITHMS_MEMORY_ALIAS_GRAPH_FIXTURE_HPP
