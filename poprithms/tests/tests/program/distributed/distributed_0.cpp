// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <iostream>
#include <set>
#include <sstream>

#include <poprithms/error/error.hpp>
#include <poprithms/program/distributed/helper.hpp>
#include <poprithms/program/distributed/program.hpp>

namespace {

using namespace poprithms::program::distributed;

class Op {
public:
  OpId id;
  SubGraphIds callees;
  CodeLocation location;
};
using Ops = std::vector<Op>;

// Mock class for testing Sequences.
class TestHelper : public Helper {
public:
  SubGraphIds userCallable() const final { return userCallable_; }

  OpIds schedule(SubGraphId sgId) const final {
    OpIds ids_;
    for (const auto &op : opsBySubGraph[sgId.get_u64()]) {
      ids_.push_back(op.id);
    }
    return ids_;
  }

  SubGraphIds callees(OpId opId) const final { return op(opId).callees; }

  OpId
  insert(SubGraphId sgId, const SubGraphIds &callees, CodeLocation location) {
    auto id = nOps;
    if (opsBySubGraph.size() <= sgId.get_u64()) {
      opsBySubGraph.resize(sgId.get_u64() + 1);
    }
    auto within = opsBySubGraph[sgId.get_u64()].size();

    subSubGraphIds.insert({id, {sgId, within}});
    opsBySubGraph[sgId.get_u64()].push_back({id, callees, location});

    ++nOps;
    return id;
  }

  CodeLocation codeLocation(OpId opId) const final {
    return op(opId).location;
  }

  const Op &op(OpId opId) const {
    auto local = subSubGraphIds.at(opId);
    return opsBySubGraph[local.first.get_u64()][local.second];
  }

  void setCallable(const SubGraphIds &ids) { userCallable_ = ids; }

  std::map<OpId, std::pair<SubGraphId, uint64_t>> subSubGraphIds;
  uint64_t nOps{0};
  std::vector<Ops> opsBySubGraph;
  SubGraphIds userCallable_;
};

void test0() {
  TestHelper th;

  auto sg0 = SubGraphId::createSubGraphId(0);
  auto sg1 = SubGraphId::createSubGraphId(1);

  // Graph 0: {host, host, ipu, host, host}
  th.insert(sg0, {}, CodeLocation::Host);
  th.insert(sg0, {}, CodeLocation::Host);
  th.insert(sg0, {}, CodeLocation::Ipu);
  auto h0 = th.insert(sg0, {}, CodeLocation::Host);
  auto h1 = th.insert(sg0, {}, CodeLocation::Host);

  // Graph 1 {ipu, ipu)
  th.insert(sg1, {}, CodeLocation::Ipu);
  th.insert(sg1, {}, CodeLocation::Ipu);
  th.setCallable({sg0, sg1});
  Sequences seqs(th);

  if (seqs.at(sg0).nPrograms() != 3) {
    throw poprithms::test::error(
        "Expected the first sequence to be divided into 3 programs (host, "
        "host), (ipu), (host, host)");
  }

  if (seqs.enginePrograms() !=
      Sequences::EngProgs{{sg0, 1}, {sg1, ProgramIndex{0}}}) {
    throw poprithms::test::error(
        "Expected 2 engine programs: program #1 in sequence #0 (the partial "
        "ipu code) and the solo program of sequence (sub-graph) #1.");
  }

  if (seqs.at(sg0).programs()[2].opIds() != OpIds{h0, h1}) {
    throw poprithms::test::error(
        "Expected the ops in program #2 to be different. ");
  }

  if (seqs.at(sg0).programs()[0].hasIpuCallId() ||
      !seqs.at(sg0).programs()[1].hasIpuCallId()) {
    throw poprithms::test::error("Expect programs to have ipu call ids if "
                                 "and only if they are engine programs");
  }
}

void test1() {

  const auto sg0 = SubGraphId::createSubGraphId(0);
  const auto sg1 = SubGraphId::createSubGraphId(1);
  const auto sg2 = SubGraphId::createSubGraphId(2);

  TestHelper th;
  // sub-graph 0 (sg0) is a sequence of 2 ops, one which is run on host, and
  // the other on ipu.
  th.insert(sg0, {}, CodeLocation::Host);
  th.insert(sg0, {}, CodeLocation::Ipu);

  // sub-graph 1 is a sequence of 2 ops, (1) a call op on host, then an op
  // ipu.
  th.insert(sg1, {sg0}, CodeLocation::Host);
  th.insert(sg1, {}, CodeLocation::Ipu);

  // sub-graph 2 also has 2 ops, the first is a call op on host, the second is
  // non-call op on host.
  th.insert(sg2, {sg1}, CodeLocation::Host);
  th.insert(sg2, {}, CodeLocation::Host);

  th.setCallable({sg2});
  Sequences seqs(th);

  // Logging when seqs is printed:
  //   SubGraphId=0: ((Host,ipuCallId=none,ops=(0)),(Ipu,ipuCallId=0,ops=(1)))
  //   SubGraphId=1: ((Host,ipuCallId=none,ops=(2)),(Ipu,ipuCallId=1,ops=(3)))
  //   SubGraphId=2: ((Host,ipuCallId=none,ops=(4,5)))
  //   Engine Programs:((0,1),(1,1))

  if (seqs.enginePrograms() != Sequences::EngProgs{{sg0, 1}, {sg1, 1}}) {
    throw poprithms::test::error(
        "The engine programs are the programs #1 in sg0 and sg1");
  }
}

void test2() {

  TestHelper th;

  const auto sg5 = SubGraphId::createSubGraphId(5);
  const auto sg0 = SubGraphId::createSubGraphId(0);
  const auto sg1 = SubGraphId::createSubGraphId(1);
  const auto sg2 = SubGraphId::createSubGraphId(2);

  th.insert(sg5, {}, CodeLocation::Ipu);

  auto a = th.insert(sg0, {sg5}, CodeLocation::Ipu);
  auto b = th.insert(sg0, {}, CodeLocation::Ipu);
  th.insert(sg0, {}, CodeLocation::None);
  auto c = th.insert(sg0, {}, CodeLocation::Ipu);
  th.insert(sg0, {}, CodeLocation::None);

  th.insert(sg1, {sg0}, CodeLocation::Ipu);
  th.insert(sg1, {}, CodeLocation::Ipu);
  th.insert(sg1, {sg0}, CodeLocation::Ipu);

  th.insert(sg2, {sg1}, CodeLocation::Ipu);

  th.setCallable({sg0, sg2});

  //     SubGraphId=0: ((Ipu,ipuCallId=0,ops=(1,2,4)))
  //     SubGraphId=1: ((Ipu,ipuCallId=none,ops=(6,7,8)))
  //     SubGraphId=2: ((Ipu,ipuCallId=1,ops=(9)))
  //     SubGraphId=5: ((Ipu,ipuCallId=none,ops=(0)))

  Sequences seqs(th);

  if (seqs.enginePrograms() != Sequences::EngProgs{{sg0, 0}, {sg2, 0}}) {
    throw poprithms::test::error(
        "All of the sequences have a single program (because the Nones are "
        "skipped and there are no Hosts). Sequences 2 and 0 are callable.");
  }

  if (seqs.at(sg0).programs()[0].opIds() != OpIds{a, b, c}) {
    throw poprithms::test::error(
        "Incorrect OpIds in sub-graph 0. Is this because of the None "
        "locations?");
  }
}

void test3() {

  const auto sg0 = SubGraphId::createSubGraphId(0);
  const auto sg1 = SubGraphId::createSubGraphId(1);
  const auto sg2 = SubGraphId::createSubGraphId(2);

  TestHelper th;

  th.insert(sg0, {}, CodeLocation::Ipu);
  th.insert(sg0, {}, CodeLocation::Ipu);

  th.insert(sg1, {}, CodeLocation::Host);
  th.insert(sg1, {}, CodeLocation::Host);

  th.insert(sg2, {}, CodeLocation::Host);

  // 2 callees? This reprsents code for an IfOp. Something like:
  //    if (a == 0){ run on host } else { run on ipu }
  th.insert(sg2, {sg0, sg1}, CodeLocation::Host);

  th.insert(sg2, {}, CodeLocation::Ipu);

  th.setCallable({sg2});

  Sequences seqs(th);

  if (seqs.enginePrograms() != Sequences::EngProgs{{sg0, 0}, {sg2, 1}}) {
    throw poprithms::test::error("2 engine programs expected");
  }
}

void test4() {

  const auto sg0 = SubGraphId::createSubGraphId(0);
  const auto sg1 = SubGraphId::createSubGraphId(1);
  const auto sg2 = SubGraphId::createSubGraphId(2);
  const auto sg3 = SubGraphId::createSubGraphId(3);

  TestHelper th;

  th.insert(sg0, {}, CodeLocation::Ipu);
  th.insert(sg0, {}, CodeLocation::Ipu);

  th.insert(sg1, {}, CodeLocation::Ipu);
  th.insert(sg1, {sg0}, CodeLocation::Ipu);

  th.insert(sg2, {}, CodeLocation::Ipu);
  th.insert(sg2, {sg1}, CodeLocation::Ipu);
  th.insert(sg2, {sg0, sg1}, CodeLocation::Ipu);

  th.insert(sg3, {sg2}, CodeLocation::Ipu);
  th.insert(sg3, {}, CodeLocation::Host);
  th.insert(sg3, {sg2}, CodeLocation::Ipu);

  th.setCallable({sg1, sg3});

  Sequences seqs(th);

  if (seqs.enginePrograms() !=
      Sequences::EngProgs{{sg1, 0}, {sg3, 0}, {sg3, 2}}) {
    std::ostringstream oss;
    oss << seqs;
    throw poprithms::test::error(
        "Expected 3 engine programs. The apparent duplication of the call "
        "from sg3 into sg2 cannot be combined into a single ipu call id. " +
        oss.str());
    ;
  }
}

} // namespace

int main() {
  test0();
  test1();
  test2();
  test3();
  test4();
}
