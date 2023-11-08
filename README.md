# qcu_dev

before reconstructing new qcu, I use this repo to reconstruct my own qcu, in that the old qcu is too hard to fix bugs or add functions.

## How to Build Qcu

Assume you have MPI and CUDA, after you clone my repo, `cd qcu_dev` into my directory, then `cmake -B build` and `cmake --build build -j 12`, then you get `libqcu.so` in `build`.