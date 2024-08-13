# MLIR

简单的，跟随官方教程的chapter来学习

This tutorial is divided in the following chapters:

- Chapter #1: Introduction to the Toy language and the definition of its AST.
- Chapter #2: Traversing the AST to emit a dialect in MLIR, introducing base MLIR concepts. Here we show how to start attaching semantics to our custom operations in MLIR.
- Chapter #3: High-level language-specific optimization using pattern rewriting system.
- Chapter #4: Writing generic dialect-independent transformations with Interfaces. Here we will show how to plug dialect specific information into generic transformations like shape inference and inlining.
- Chapter #5: Partially lowering to lower-level dialects. We’ll convert some of our high level language specific semantics towards a generic affine oriented dialect for optimization.
- Chapter #6: Lowering to LLVM and code generation. Here we’ll target LLVM IR for code generation, and detail more of the lowering framework.
- Chapter #7: Extending Toy: Adding support for a composite type. We’ll demonstrate how to add a custom type to MLIR, and how it fits in the existing pipeline.