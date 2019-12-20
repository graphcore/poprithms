int main() {
  // In this example, we show how the other user priorities can override
  // minimsing allocation liveness.
  //
  // The basic idea is, for the highest priority Op, create an Alloc with a
  // very large weight, and make it correspond to an Op which is the producer
  // of all Ops in the Graph which don't have producers. For each priority,
  // for each Op, choose a corresponding weight.

  return 0;
}
