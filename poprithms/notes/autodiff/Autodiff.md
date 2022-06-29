A quick guide to using poprithms autodiff in your project:

1) Get familiar with the class guide::Objective, which defines the overall 
   objective of differentiating a graph. 

2) Implement a class which inherits from guide::GraphInfo, which defines 
   (without any calculus) how gradients flow through ops, and which tensors are 
   required to differentiate an op. Note that input and output indices are 
   assumed to be contiguous (as in all poprithms projects). 

3) With (1) and (2) above, the 'first level' of differentiation is possible: 
   guide::Guide will be able to tell you which tensors must be recomputed, 
   which (intermediate) tensors have gradients, the order in which the gradient 
   operations will be run, etc. It will NOT actually do any graph construction, 
   that happens at the 'second level' (see below)

4) To be able to actually generate a gradient graph in your project, you will 
   need to implement a class which inherits from the abstract class, 
   core::GraphMutator. This class has pure virtual methods such as 'createZero', 
   'createVariable', and 'add', each of which you will need to implement to 
   create an Op in your project's graph. This is where you must implement your 
   'calculus' -- we don't do any calculus for you :-). 

5) Finally, with 1, 2, and 4 above, you perform auto-differentiation with the 
   core::Autodiff class. See for example the example in the test directory (core_0.cpp);

Some notes:
- There is no "Sum" operation used to accumulate the gradients along different paths.
  Instead, a tree of adds is created.
