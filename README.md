# MLC-Learning

[English](./README-en.md)

学习深度学习编译的笔记。这个领域影响力比较大的课程是陈天奇的[MLC course](https://mlc.ai/summer22-zh/schedule)，但是他花了大量的篇幅在TVM上，使得课程看上去像是TVM tutorial。

这个仓库的目标是记录我学习MLC的过程，作为笔记和实验的存储仓库。

## Contents

### [LLVM](./LLVM/)

主要学习：

1. LLVM前中后端的编译过程（IR转换过程）
2. PassManager等组件的梳理
3. [Kaleidoscope](https://github.com/JYRoy/Kaleidoscope)项目：官方教程，使用LLVM开发自己的Kaleidoscope语言

以了解LLVM IR、LLVM编译流程和学会使用LLVM开发深度学习编译器为目标。

### [MLIR](./MLIR/)

跟随MLIR官方的教程 (Toy Tutorial)[https://mlir.llvm.org/docs/Tutorials/Toy/]，实现一个toy language

主要学习：

1. AST方式
2. Dialect定义方式
3. Pattern Rewrite System
4. Transform、lowering方式

### [Triton](./Triton/)

主要学习：

1. Triton 语法
2. Triton Kernel经典例子
3. Triton的编译过程
4. Triton中的重要Passes
5. 基于Triton的项目调研

以写Kernel和二次开发Triton底层为目标。

## Future

也许是更基础的编译理论（毕竟不熟），也许是TVM，也许是其他。。。