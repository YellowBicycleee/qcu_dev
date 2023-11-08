# INTRODUCTION

目前正在写的主要是从`dslash`到`cg inverter`部分的内容。

接下来将从Wilson Dslash ----> Mpi Dslash ---->Clover Dslash----->CG INVERTER顺序进行重构。

核函数尽可能放在`include/kernel`下，另外为了照顾DCU，所有`__device__`函数都会加上`__forceinline__`以达到inline效果。

在本项目中默认维度为4维，Nd = 4, Nc = 3, Ns = 4。

gauge输入格式为[Nd, Lt, Lz, Ly, Lx, Nc, Nc, 2]，最后一个2表示2个double组成一个Complex，至于实际上Complex存的是单精度还是双精度，我们暂且认为都是双精度，也就是double。经过奇偶预处理后变为[Nd, cb, Lt, Lz, Ly, Lx/2, Nc, Nc, 2]

Vector输入格式为[Lt, Lz, Ly, Lx, Ns, Nc, 2]，奇偶预处理后[cb, Lt, Lz, Ly, Lx, Ns, Nc, 2]