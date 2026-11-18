# TinyML on Arm: PyTorch and ExecuTorch on Corstone-320 FVP

This project takes a PyTorch feedforward neural network, quantizes it, compiles it ahead of time (AoT) for the Arm Ethos-U85 NPU, and embeds it in a bare-metal C++ firmware binary. The firmware runs on the Arm Corstone-320 Fixed Virtual Platform (FVP), which simulates a Cortex-M85 with an Ethos-U85.

## Project structure

~~~text
tinyML/
├── models/
│   └── simple_nn.py      # 2-layer PyTorch neural network (10 -> 5 -> 2)
├── run_pipeline.sh       # Automated AoT compilation, cross-build, and FVP execution script
├── requirements.txt      # Core Python package versions
├── .gitignore            # Excludes build outputs, .pte binaries, and caches
└── README.md             # Setup and reproduction guide
~~~

## Architecture pipeline

~~~text
[PyTorch Model] ──► [ExecuTorch AoT Compiler] ──► [Bare-Metal ARM ELF] ──► [Corstone-320 FVP]
 (simple_nn.py)      (Quantized + Ethos-U85)      (arm_executor_runner)    (Simulated Hardware)
~~~

1. `models/simple_nn.py` defines `SimpleNN` (`Linear(10, 5) -> ReLU -> Linear(5, 2)`) and exposes `ModelUnderTest` and `ModelInputs` for ExecuTorch export.
2. The compiler quantizes the 32-bit float weights to 8-bit integers and delegates the graph operators to the Ethos-U85 NPU. The resulting `.pte` flatbuffer is about 3.3 KB.
3. `arm-none-eabi-gcc` builds the ExecuTorch runtime and embeds the `.pte` flatbuffer in a bare-metal firmware executable.
4. `FVP_Corstone_SSE-320` boots the virtual Cortex-M85 and Ethos-U85 subsystem and prints cycle counts and class predictions over UART0.

## Environment setup (Ubuntu 22.04 / 24.04 / 26.04)

### 1. System dependencies
~~~bash
sudo apt update && sudo apt install -y python3-dev python3-venv gcc g++ gcc-13 g++-13 make patchelf
~~~

### 2. Python virtual environment and ExecuTorch
~~~bash
python3 -m venv $HOME/executorch-venv
source $HOME/executorch-venv/bin/activate

cd $HOME
git clone https://github.com/pytorch/executorch.git
cd executorch
git checkout release/1.0
git submodule sync
git submodule update --init --recursive

# Patch missing cstdint header for newer GCC versions & build with GCC 13
sed -i '1i #include <cstdint>' extension/llm/tokenizers/third-party/sentencepiece/src/sentencepiece_processor.h
CC=gcc-13 CXX=g++-13 ./install_executorch.sh
~~~

### 3. Corstone-320 FVP and Arm GNU toolchain
~~~bash
export CC=gcc-13
export CXX=g++-13
cd $HOME/executorch
./examples/arm/setup.sh --i-agree-to-the-contained-eula
source $HOME/executorch/examples/arm/ethos-u-scratch/setup_path.sh
~~~

> On Ubuntu 26.04+ (glibc 2.41+), clear the legacy executable stack flag on the precompiled Corstone-320 shared libraries so `dlopen` can load them:
> ~~~bash
> find $HOME/executorch/examples/arm/ethos-u-scratch/FVP-corstone320 -name "*.so*" -exec patchelf --clear-execstack {} + 2>/dev/null
> ~~~

## Running the project

From the root of this repository:

~~~bash
./run_pipeline.sh
~~~

### Expected output
~~~text
I [executorch:arm_executor_runner.cpp:578 runner_init()] PTE Model data loaded. Size: 3392 bytes.
I [executorch:arm_executor_runner.cpp:1100 run_model()] Starting running 1 inferences...
I [executorch:arm_executor_runner.cpp:1124 run_model()] 1 inferences finished
I [executorch:arm_executor_runner.cpp:875 print_outputs()] 1 outputs:
Output[0][0]: (float) ...
Output[0][1]: (float) ...
I [executorch:arm_executor_runner.cpp:1238 main()] Program complete, exiting.
~~~