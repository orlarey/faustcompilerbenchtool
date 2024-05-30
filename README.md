# fcbenchtool
A simple benchmark tool to measure the performance of the C++ code generate by the Faust compiler. 

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

