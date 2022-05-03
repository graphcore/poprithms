// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <gmock/gmock.h>
#include <mock/memory/alias/mocktensor.hpp>

// We're testing that we can find the directory containing the mocks, and that
// we can create and run a google test with ctest. Consider removing this test
// when real tests arrive.
namespace {

class MockTensorTester {
public:
  static void getRank(const mock::poprithms::memory::alias::MockTensor &t) {
    auto x = t.shape().rank_u64();
  }
};

class Marionette {
public:
  virtual ~Marionette()     = default;
  virtual void step() const = 0;
};

class MockMarionette : public Marionette {
public:
  MOCK_METHOD(void, step, (), (const, override));
};

class Puppeteer {
public:
  const Marionette &f;
  Puppeteer(const Marionette &f_) : f(f_) {}
  void walk(uint64_t dst) {
    for (uint64_t i = 0; i < dst; ++i) {
      f.step();
    }
  }
};
} // namespace

using ::testing::AtLeast;
TEST(PuppeteerTest, Walking) {
  MockMarionette m;
  uint64_t nSteps{10};
  EXPECT_CALL(m, step()).Times(AtLeast(nSteps));
  Puppeteer painter(m);
  painter.walk(nSteps + 1);
}
