#!/usr/bin/env bash
set -e

export ET_HOME=${ET_HOME:-$HOME/executorch}
export executorch_DIR=\$ET_HOME/build
export FETCHCONTENT_UPDATES_DISCONNECTED=ON

source "\$HOME/executorch-venv/bin/activate"
source "\$ET_HOME/examples/arm/ethos-u-scratch/setup_path.sh"

SCRIPT_DIR="\$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/models/simple_nn.py" "$ET_HOME/examples/arm/simple_nn.py"

echo "==> 1. Compiling & quantizing PyTorch model to .pte for Ethos-U85..."
cd "\$ET_HOME"
python -m examples.arm.aot_arm_compiler \
  --model_name=examples/arm/simple_nn.py \
  --delegate \
  --quantize \
  --target=ethos-u85-256 \
  --system_config=Ethos_U85_SYS_DRAM_Mid \
  --memory_mode=Sram_Only

echo "==> 2. Building bare-metal ExecuTorch libraries & Arm runner..."
cmake -S "\${ET_HOME}" \
  -B "\${executorch_DIR}" \
  --preset arm-baremetal \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "\$executorch_DIR" --target install --parallel

cmake -S "\${ET_HOME}/examples/arm/executor_runner" \
  -B "\${ET_HOME}/examples/arm/executor_runner/cmake-out" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="\$ET_HOME/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake" \
  -DTARGET_CPU=cortex-m85 \
  -DET_PTE_FILE_PATH:PATH="\$ET_HOME/simple_nn_arm_delegate_ethos-u85-256.pte" \
  -DETHOS_SDK_PATH:PATH="\$ET_HOME/examples/arm/ethos-u-scratch/ethos-u" \
  -DETHOSU_TARGET_NPU_CONFIG=ethos-u85-256 \
  -DPYTHON_EXECUTABLE="\$HOME/executorch-venv/bin/python3" \
  -DSYSTEM_CONFIG=Ethos_U85_SYS_DRAM_Mid

cmake --build "\$ET_HOME/examples/arm/executor_runner/cmake-out" --parallel -- arm_executor_runner

echo "==> 3. Running inference on Corstone-320 FVP..."
FVP_Corstone_SSE-320 \
  -C mps4_board.subsystem.ethosu.num_macs=256 \
  -C mps4_board.visualisation.disable-visualisation=1 \
  -C vis_hdlcd.disable_visualisation=1 \
  -C mps4_board.telnetterminal0.start_telnet=0 \
  -C mps4_board.uart0.out_file='-' \
  -C mps4_board.uart0.shutdown_on_eot=1 \
  -a "\$ET_HOME/examples/arm/executor_runner/cmake-out/arm_executor_runner"
