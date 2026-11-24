# minigrad-cpp: an autograd engine in C++17

![build](https://github.com/25poojam/minigrad-cpp/actions/workflows/build.yml/badge.svg)

A small **reverse-mode automatic differentiation** engine and neural-network library written in modern C++ with zero dependencies. It implements the core idea behind PyTorch's `autograd`: build a computation graph during the forward pass, then walk it backwards to compute every gradient with the chain rule.

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), re-implemented in C++ with attention to memory management and graph depth.

```cpp
Value a(2.0), b(-3.0);
Value c = a * b + a.tanh();
c.backward();
std::cout << a.grad();   // dc/da, computed automatically
```

## Demo: learning a non-linear boundary

A 337-parameter MLP learns to separate two interleaving half-moons (98% training accuracy, ~2 s):

```
..........oo..oo............oo.ooo........................##
........o..oo.oo...............ooo..o.................######
.....................###......oo..................##########
....o.....o........###xx##.....................###xx########
...........o.....####x######......o.o......#######x##x######
.......oo.......#####xx#######.........#############xx######
..........o....########x##x########o###########xx####x######
```

Full output: [`assets/moons_output.txt`](assets/moons_output.txt)

## Features

- **Ops:** `+ - * /`, `pow`, `exp`, `log`, `tanh`, `relu`, plus mixed `Value`/`double` arithmetic
- **Backward pass:** iterative topological sort, so it handles 200,000+-deep graphs without stack overflow
- **Memory-safe graph:** `shared_ptr` ownership, no reference cycles, and an iterative destructor to avoid recursive teardown. Verified clean under AddressSanitizer and UBSan.
- **NN library:** `Neuron`, `Layer`, `MLP`, `zero_grad()`, `parameters()`
- **Tests:** every operation checked against numerical derivatives; CI on Linux, macOS and Windows

## Build and run

Requires CMake ≥ 3.16 and a C++17 compiler.

```bash
git clone https://github.com/25poojam/minigrad-cpp.git
cd minigrad-cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

./build/moons                      # train the demo
ctest --test-dir build --output-on-failure   # run tests
```

## Project layout

```
include/minigrad/value.hpp   Value + Node: the autograd core
include/minigrad/nn.hpp      Neuron / Layer / MLP
src/value.cpp                Forward ops, local derivatives, backward()
examples/moons.cpp           Training demo with ASCII decision boundary
tests/test_value.cpp         Gradient tests (no external framework)
```

## Design notes

- **Why raw pointers in the backward lambdas?** If a node's `backward_fn` captured a `shared_ptr` to itself, the node would own itself and never be freed. The output node already owns its parents, so raw pointers are safe for the lifetime of the closure.
- **Why `+=` for gradients?** A value used in two places receives gradient from both paths (multivariable chain rule). `test_value_used_twice_accumulates` guards against this classic bug.
- **Why an iterative destructor?** Default `shared_ptr` teardown of a long chain recurses once per node and crashes on deep graphs. I found this when the deep-graph test segfaulted.

## Possible extensions

- [ ] Tensor (vector/matrix) values instead of scalars, for real speed
- [ ] Python bindings with pybind11
- [ ] Graphviz export of the computation graph
- [ ] Adam optimizer
