# 2. Wilson Dslash

输入格式为input vector + output vector ------> WilsonDslash-------->output vector

## 2.1 存储结构

### 2.1.1 Gauge

#### Naive存储结构

gauge的存储结构，在进行奇偶预处理前的格式为`[Nd, Lt, Lz, Ly, Lx, Nc, Nc, rc=2]`，进行奇偶预处理后，`x`方向的长度收缩一半，每个维度的gauge分成两半，即`half_Lx = Lx / 2`，新的格式为`[Nd, cb=2, Lt, Lz, Ly, half_Lx, Nc, Nc, rc=2]`。这也就是所谓的`Naive`，不做存储结构上的改变，这也就是整个QCU接受的来自外部的数据格式。

#### Coalescing存储结构

在CUDA程序中，相邻程序访问相邻地址，由于memory coalescing，会导致访存速度提升，从而一定程度提升总体性能。对于单次的Dslash，由于要从naive vector转为coalescing vector，再进行wilson dslash，再将结果转回naive vector，单次dslash的性能可能不如不转数据结构。

我们假设Complex由两个双精度浮点数组成，也就是Complex = (real: double, imag: double)，


## 2.2 提供接口

## 2.3 接口介绍