# fcbenchtool
A simple benchmark tool to measure the performance of the C++ code generate by the Faust compiler. The idea is, from `foo.dsp`, to generate various `foo1.cpp`,  `foo2.cpp`, etc. implementations, by varing the Faust compiler options. Then we compare the performances of all these implementations by translating them into binary using `fcbenchtool foox.cpp` and then executing the resulting binary `foo1`. 

The `fcbenchtool` wraps the original `foox.cpp` source file between a header and footer equiped to measure the performance of the code, and compiles it with `-Ofast` optimizations and `-march=native`. The resulting binary can then be executed and it will provide timing information in millisecond.  

## Installation
The script is installed in the `/usr/local/bin` directory, and its dependencies in the `/usr/local/share/fcbenchtool` directory. 

```bash
sudo ./install.sh
```

## Usage

1. Generate C++ code from a Faust file.
2. Compile the generated C++ code with `fcbenchtool` to inject code that measures the execution time of the compute method for a count of 44100 samples.
3. Execute the resulting binary. The result is expressed in milliseconds. The program will iterate the measurement until it has the same minimal result for at least 1000 iterations.


```bash
faust foo.dsp -o foo.cpp
fcbenchtool foo.cpp
sudo ./foo
```

