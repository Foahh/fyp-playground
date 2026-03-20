# Getting started - How to evaluate a model on STM32N6 development board

---

*for STM32 target, based on ST Edge AI Core Technology 4.0.0*

r1.1

- [x] STM32N6 with Neural-ART accelerator™

# Overview

This article is a guide describing the different steps to quickly evaluate a quantized model on an STM32N6 development board. It uses a specific built-in test application and a set of utilities to deploy and run a model on an [STM32N6](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html) device, which embeds an [ST Neural ART accelerator™](stneuralart_programming_model.html).

## Setting up a work environment

A prerequisite to start using the ST Edge AI Core CLI is to install the latest version of the ST Edge AI Core CLI, as described in the [*Installer*](modular_installer.html) article.

> [`%STEDGEAI_CORE_DIR%`](setting_env.html) indicates the root location where the ST Edge AI Core components are installed.

- Python 3.9+ environment.
- STM32CubeProgrammer supporting the STM32N6 devices ([STM32CubeProg](https://www.st.com/en/development-tools/stm32cubeprog.html)).
- One of the following toolchains or IDEs for STM32 Arm® Cortex®-M-based MCUs must be installed.
  - STMicroelectronics - STM32CubeIDE version 1.15.1 or later ([STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html)).
  - IAR Systems - IAR Embedded Workbench® IDE - Armv9.30.1+ ([www.iar.com/iar-embedded-workbench](https://www.iar.com/)).
  - GNU Arm Embedded Toolchain ([Arm GNU Toolchain](https://developer.arm.com/Tools%20and%20Software/GNU%20Toolchain)).
- STM32N6 development board. [STM32N6570-DK](https://www.st.com/en/evaluation-tools/stm32n6570-dk.html) board is recommended. A Nucleo development board can be used, but the UCs are limited because no external RAM is available.

The [`$STEDGEAI_CORE_DIR/scripts/N6_scripts/config.json'`](#ref_tools_config_json) file should be updated to indicate the paths to find the external tools. Note that in this lab, the [`'config_n6l.json'`](#ref_tools_config_n6l_json) file is not used.

<details>
<summary><strong>Built-in test application</strong></summary>

Location:  
`NPU_VALIDATION = $STEDGEAI_CORE_DIR/Projects/STM32N6570-DK/Applications/NPU_Validation`

- IAR project entry point:  
  `$NPU_VALIDATION/EWARM/Project.eww`
- GCC project entry point:  
  `$NPU_VALIDATION/armgcc/Makefile`

The [built-in test application](stneuralart_stm32n6_projects.html#ref_npu_validation_project) is a generic application allowing to test a deployed model. Some features can be modified (See `$NPU_VALIDATION/Core/Inc/app_config.h` file).

- The application uses only the AXIRAM1 (DTCM and ITCM are not used).
  - No support for execution from flash.
  - Include also the code/data sections for the generated `network.c` with the epoch blobs.
- The internal AXIRAM2 and NPURAMs (AXIRAM3 up to AXIRAM6) memories are dedicated to the [NPU memory subsystem](stneuralart_programming_model.html#ref_npu_memory_sub_system).
  - The extFLASH (128 Mbytes) and extRAM (32 Mbytes) are also reserved.
- The [NPUCACHE](stneuralart_programming_model.html#ref_npu_cache) is always configured and cannot be used as normal RAM.
- MCU cache is enabled (`USE_MCU_DCACHE`/`USE_MCU_ICACHE` c-defines can be used to disable it).
- Overdrive mode is defined by default (NPU@1GHz, MCU@800Mhz)
  - Normal conf and 400MHz only configurations: see `USE_OVERDRIVE`, `NO_OVD_CLK400` C-defines.
  - *No Low-power config*, this application cannot be used to measure the power consumption.
- Serial COM setting: 921600-8-n-s.
  - Default baud rate can be changed - `USE_UART_BAUDRATE` up to 2764800.
- [NPU runtime config](stneuralart_api_and_stack.html#ref_stack_configuration): `LL_ATON_OSAL_BARE_METAL`, `LL_ATON_RT_ASYNC`..

</details>

# Getting started

This section describes the different steps to deploy and to profile a model:

1. [Generate](#ref_getting_started_generate)
2. [Build and flash](#ref_getting_started_build_and_flash)
3. [Profile](#ref_getting_started_profile) and [evaluate the accuracy](#ref_getting_started_evaluate)

> **Note**  
> For this lab, a pretrained model respecting the [required quantization scheme](quantization.html#ref_quant_scheme) has been imported from the *STMicroelectronics – STM32 model zoo* GitHub ([repo](https://github.com/STMicroelectronics/stm32ai-modelzoo)). The following image classification model, fully trained on Flowers dataset is used: [mobilenet_v2_0.35_224_fft_int8.tflite](https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/main/image_classification/pretrained_models/mobilenetv2/ST_pretrainedmodel_public_dataset/food-101/mobilenet_v2_0.35_224_fft/mobilenet_v2_0.35_224_fft_int8.tflite).

<details>
<summary>Creating a user working folder</summary>

For this lab, a user working folder, such as `'user_n6_getting_started'`, is created. The downloaded TFLite file is copied into this folder, and all commands are executed from within it.

</details>

## Generate the specialized c-files {#ref_getting_started_generate}

The following command generates specialized configuration C files with minimal default options. When no argument is provided to the `'--st-neural-art'` option, a [default compilation profile file](stneuralart_neural_art_compiler.html#ref_st_neural_art_option) is used, and the associated `"default"` setting is applied. This profile uses the mandatory [ST Neural-ART compiler options](stneuralart_neural_art_compiler.html#ref_aton_compiler_options) and a memory pool descriptor file that allows the use of all available memory on the board. Only the first megabyte is reserved for the application, while the external flash is used to store the [weights and parameters](glossary.html#ref_weights_params_buffer) of the deployed model.

- Default compilation profile file:  
  `$STEDGEAI_CORE_DIR/Utilities/windows/targets/stm32/resources/neural_art.json`
- Memory-pool descriptor file:  
  `$STEDGEAI_CORE_DIR/Utilities/windows/targets/stm32/resources/mpools/stm32n6.mpool`

```batch
$> stedgeai generate -m mobilenet_v2_0.35_224_fft_int8.tflite --target stm32n6 --st-neural-art
```

Generated files are stored in the default `'st_ai_output'` folder. The default file-prefix `'network_'` and c-name suffix `'network'` are used. The `-n/--name` option (respectively `-o/--output` option) can be used to override the c-name suffix (respectively output folder).

> **Warning**  
> When generating for validation purposes, do not change the c-name suffix.

```text
<user_n6_getting_started>\st_ai_output\mobilenet_v2_0.35_224_fft_int8_OE_3_1_0.onnx
<user_n6_getting_started>\st_ai_output\mobilenet_v2_0.35_224_fft_int8_OE_3_1_0_Q.json
<user_n6_getting_started>\st_ai_output\network.c
<user_n6_getting_started>\st_ai_output\network.h
<user_n6_getting_started>\st_ai_output\network_atonbuf.xSPI2.raw
<user_n6_getting_started>\st_ai_output\network_c_info.json
<user_n6_getting_started>\st_ai_output\network_generate_report.txt
```

**Note:** The `network.h` file is not strictly required to deploy the generated model. Instead, C defines are used by the built-in test application to adapt its behavior.

- The [intermediate files](stneuralart_programming_model.html#ref_aot_flow) (`<model_file_name>_OE_X_Y_Y[_Q].[onnx | json]`) are also provided. Advanced user can use them later with the [ST Neural-ART compiler](stneuralart_neural_art_compiler.html).
- The `'network_generate_report.txt'` provides the main information (txt format) about the imported model and how it is deployed. It provides a summary about the usage of the memories, used options and epoch types. The `'network_c_info.json'` file contains the same information but with a [well-defined JSON format](network_c_info_json.html).

The following command uses the minimal profile enhanced with the optimization for speed.

```batch
$> stedgeai generate -m mobilenet_v2_0.35_224_fft_int8.tflite --target stm32n6 --st-neural-art profile-allmems--O3
```

Available tool built-in profiles are described in the [*ST Neural-ART compiler primer*](stneuralart_neural_art_compiler.html#ref_built_in_tool_profiles) article.

## Build and load the built-in test application {#ref_getting_started_build_and_flash}

The following command without options can be used to:

- Prepare the compiler environment (`iar` or `gcc`, see `"compiler_type"` from [`$STEDGEAI_CORE_DIR/scripts/N6_scripts/config.json'`](#ref_tools_config_json) file).
- Update the C-project with the generated specialized files.
  - Location: `$STEDGEAI_CORE_DIR/Projects/STM32N6570-DK/Applications/NPU_Validation/`
- Convert the raw memory initializer files to an Intel-hex format to be used by the flasher application
- Build the c-project.
  - Calling makefile entry point for `gcc`-based project.
  - Calling the IAR project file for `iar`-based project.
- Load the application and the memory initializers - Note that the firmware image is loaded through a debug session. The board should be configured in **dev mode**.
- Start the application.

```bash
$> python $STEDGEAI_CORE_DIR/scripts/N6_scripts/n6_loader.py
```

```text
XXX -- Preparing compiler IAR
XXX -- Setting a breakpoint in main.c at line 127 (before the infinite loop)
XXX -- Copying network.c to project: <user_n6_getting_started>\st_ai_output\network.c -> $PROJECT\X-CUBE-AI\App\network.c
XXX -- Extracting information from the c-file
XXX -- Converting memory files in results/<model>/generation/ to Intel-hex with proper offsets
XXX -- arm-none-eabi-objcopy.exe --change-addresses 0x70000000 -Ibinary -Oihex network_atonbuf.xSPI2.raw network_atonbuf.xSPI2.hex
XXX -- Resetting the board...
XXX -- Patching macro file for automatically loading Intel-Hex files upon breakpoint
XXX -- Flashing memory xSPI2 -- 623.969kB
XXX -- Flashing memory xSPI2 -- return code OK
XXX -- Dumping macro file
XXX -- Building project (conf= N6-DK)
XXX -- Compilation successful
XXX -- Running the program & loading internal memories
XXX -- Loading memories successful
XXX -- Start operation achieved successfully
```

At this stage, the board is ready to receive a command from the host machine (serial COM). [AiRunner package](ai_runner_python_module.html) or [*validate* command](command_line_interface.html#validate-command) of the ST Edge AI CLI can be used.

> **Warning**  
> Do not disconnect the board until the end of the experimentation: the application is stored in RAM and not in external flash (power-cycling the board erases the firmware).  
> After a power-cycle, be sure to run this step before running a new evaluation with the AI Runner.

#### Use a USB client {#ref_use_usbc_link}

By default, communication with the test application is handled via USART through the ST-Link interface at a baud rate of 921,600. To achieve higher data transfer speeds, the development board’s USB Type-C® connector (available on DK or Nucleo boards) can be utilized. The firmware includes a complete USB device stack with a virtual COM port, enabling data transfer rates exceeding 20 Mbps, significantly accelerating communication compared to USART.

To generate the test application with the USB stack, the `--build-config N6-DK-USB` option can be used (or update the `'config_n6l.json'` file accordingly).

```bash
$> python $STEDGEAI_CORE_DIR/scripts/N6_scripts/n6_loader.py --build-config N6-DK-USB
```

> **Note**  
> Please note that a second USB cable is required to connect the board to the development environment. The fisrt is always mandatory to load the test application and to power the board.

## Evaluate the performance (profiling) {#ref_getting_started_profile}

The following command allows performing a quick profiling of the deployed model (refer to [*How to use the AiRunner package*](ai_runner_python_module.html) article for more details). Ten samples randomly generated are used to perform the inferences.

```batch
$> export PYTHONPATH=$STEDGEAI_CORE_DIR/scripts/ai_runner
$> python $STEDGEAI_CORE_DIR/scripts/ai_runner/examples/checker.py -d serial:921600 --perf-only -b 10
```

The first part of the log shows the main information of the deployed model and the used runtime including the device settings.

The next part shows the inference time by epoch. Note that the MCU/CPU cycles are used to report the execution time of the different [epoch phases](stneuralart_programming_model.html#ref_epoch_definition). In the context of the used profiling application, the MCU clock is proportional to the NPU clock.

> **Note**  
> These data are also reported with the [`'validate'` command](#ref_getting_started_evaluate) used to evaluate the accuracy.

## Evaluate the accuracy {#ref_getting_started_evaluate}

### Quick evaluation

The following command allows performing a quick evaluation of the deployed model ([validation on target mode](command_line_interface.html#ref_validation_on_target_overview)). Note that the baud rate should be specified because `'115200'` baud is used by default by the ST Edge AI Core CLI application.

```batch
$> stedgeai validate -m mobilenet_v2_0.35_224_fft_int8.tflite --target stm32n6 --mode target -d serial:921600
```

By default, input samples are generated randomly, independent of the task or dataset used for training or calibrating the model. This random generation is often [not representative](evaluation_metrics.html#ref_metric_interpretation) of the real data distribution, as it assumes a uniform distribution across channels. Consequently, this can lead to suboptimal reported results. To achieve more accurate validation, it is **recommended** (see [next section](#ref_with_representative_data)) to provide representative preprocessed data. This can be done using a npy/npz file containing preprocessed images, which are typically generated during the validation of the quantized model.

```batch
Computing the metrics...
...
  Confusion matrix (axis=-1) - 5 classes (10 samples)
  ---------------------------------
  C0        0    .    .    .    .
  C1        .    0    .    .    .
  C2        .    .    0    .    .
  C3        .    .    .    0    .
  C4        .    1    .    .    9

 Evaluation report (summary)
 ----------------------------------- ... -----------------------------------------------------------
 Output       acc      rmse          ...    nse        cos        tensor
 ----------------------------------- ... -----------------------------------------------------------
 X-cross #1   90.00%   0.086785592   ...    0.865483   0.962875   'nl_69', 10 x int8(1x5), m_id=[69]
 ----------------------------------- ... -----------------------------------------------------------
```

### With representative data {#ref_with_representative_data}

The following command uses a npy/npz file containing preprocessed images, which can be typically generated during the validation of the quantized model.

```batch
$> stedgeai validate -m mobilenet_v2_0.35_224_fft_int8.tflite --target stm32n6 --mode target -d serial:921600 \
     -vi input_20_images.npy
```

```text
Computing the metrics...
...
  Confusion matrix (axis=-1) - 5 classes (20 samples)
  ---------------------------------
  C0        5    .    .    .    .
  C1        .    3    .    .    .
  C2        .    .    3    .    .
  C3        .    .    .    5    .
  C4        .    .    .    .    4

 Evaluation report (summary)
 ----------------------------------- ... -----------------------------------------------------------
 Output       acc       rmse         ...    nse        cos        tensor
 ----------------------------------- ... -----------------------------------------------------------
 X-cross #1   100.00%   0.023515496  ...    0.996440   0.998586   'nl_69', 20 x int8(1x5), m_id=[69]
 ----------------------------------- ... -----------------------------------------------------------
```

# Advanced use-cases

## Enable the epoch controller mode

### Create the user configuration files

To enable the [epoch controller](stneuralart_programming_model.html#ref_hw_epoch_controller) mode, the `'--enable-epoch-controller'` option should be passed to NPU compiler. A new profile can be added in the tool built-in profiles file or a user [compilation profiles file](stneuralart_neural_art_compiler.html#ref_compilation_profiles_json_file) can be created.  
In this lab, a set of configuration files is created locally by copying them from the tool built-in configuration files.

Copy the default configuration files in the current folder:

```bash
$> cp -r $STEDGEAI_CORE_DIR/Utilities/windows/targets/stm32/resources/mpools/ .
$> cp $STEDGEAI_CORE_DIR/Utilities/windows/targets/stm32/resources/neural_art.json .
```

Add a new profile in the `neural_art.json` file:

```json
..
"allmems--O3-ec" : {
  "memory_pool": "./mpools/stm32n6.mpool",
  "options": "--native-float --mvei --cache-maintenance --Ocache-opt --enable-virtual-mem-pools 
              --Os --optimization 3 --enable-epoch-controller"
},
..
```

### Generate

Generate the specialized c-files using the freshly created profile:

```batch
$> stedgeai generate -m mobilenet_v2_0.35_224_fft_int8.tflite --target stm32n6 --st-neural-art allmems--O3-ec@neural_art.json
```

```text
<user_n6_getting_started>\st_ai_output\mobilenet_v2_0.35_224_fft_int8_OE_3_1_0.onnx
<user_n6_getting_started>\st_ai_output\mobilenet_v2_0.35_224_fft_int8_OE_3_1_0_Q.json
<user_n6_getting_started>\st_ai_output\network.c
<user_n6_getting_started>\st_ai_output\network_atonbuf.xSPI2.raw
<user_n6_getting_started>\st_ai_output\network_ecblobs.h
<user_n6_getting_started>\st_ai_output\network_generate_report.txt
```

A new `'network_ecblobs.h'` is created. It contains the different blobs or command streams (C-array form). The `'network.c` file includes the `'network_ecblobs.h'` file.

```cpp
ECBLOB_CONST_SECTION
static const uint64_t _ec_blob_1 [] =
{
  0x000094edca057a7aUL, 0x0000005d0000003dUL, 0x5c00004200802241UL, 0x1500006008000000UL,
  0x000700005c027c00UL, 0x000040005c070043UL, 0x5c0c00430000ffe0UL, 0x1041204200000000UL,
...
```

### Build and load

As done previously, build and load the program using the provided script:

```bash
$> python $STEDGEAI_CORE_DIR/scripts/N6_scripts/n6_loader.py
```

```text
...
XXX -- Copying network.c to project: <user_n6_getting_started>\st_ai_output\network.c -> $PROJECT\X-CUBE-AI\App\network.c
XXX -- Copying network_ecblobs.h to project: <user_n6_getting_started>\st_ai_output\network_ecblobs.h -> $PROJECT\X-CUBE-AI\App\network_ecblobs.h
...
XXX -- Start operation achieved successfully
```

As expected, the blobs header is copied correctly to the project and is then used when compiling the final program.

### Profile

Performance measurement can then be done using the `checker.py` script, as shown before:

```batch
$> export PYTHONPATH=$STEDGEAI_CORE_DIR/scripts/ai_runner
$> python $STEDGEAI_CORE_DIR/scripts/ai_runner/examples/checker.py -d serial:921600 --perf-only -b 10
```

## Change the I/O data format

To facilitate the deployment of a given model, the [I/O data format](how_to_change_io_data_type_format.html) can be modified to meet the firmware constraints.

> **Note**  
> The options `'--output/input-data-type'` and `'--inputs/outputs-ch-position'` are used during the [export passes](stneuralart_programming_model.html#ref_intermediate_files) of the ST Edge AI Core CLI.  
> No extra [ST Neural-ART compiler options](stneuralart_neural_art_compiler.html) are needed to handle this use-case.

### Change the data type of I/O

The model used in this lab has float32 outputs; outputs can be converted to int8 with:

```bash
$> stedgeai generate -m mobilenet_v2_0.35_224_fft_int8.tflite --target stm32n6 \
    --st-neural-art allmems--O3-ec@neural_art.json --output-data-type int8
```

### Change the channel position of I/O

The following options allow deploying a typical ONNX model -*channel-first* format (Pytorch world)- in an embedded image sensor pipeline -*channel-last* format. In the example below, a transpose (channel-first to channel-last) and converter (int8 to uint8) layers are added. To keep the output in float32, the `--output-data-type float32` can be used.

```bash
$> stedgeai generate -m <model-path-onnx-qdq.onnx> --target stm32n6 --st-neural-art <st-neural-art-profile>
         --inputs-ch-position chlast --input-data-type uint8
```

Note, that when the `'validate'` command is used the same extra options should be also used.

# n6_loader configurations

## External tools (config.json) {#ref_tools_config_json}

The location of the required tools is done in a file called `'config.json'` which must be located in the same directory of the `'n6_loader.py'` script. An example of such a file is given below. This file is in a json-format that allows comments.

Location:  
`$STEDGEAI_CORE_DIR/scripts/N6_scripts/config.json`

```json
{
  // Set Compiler_type to either gcc or iar
  "compiler_type": "gcc",
  // Set Compiler_binary_path to your bin/ directory where IAR or GCC can be found
  //     If "Compiler_type" == gcc, then gdb_server_path shall point to where ST-LINK_gdbserver.exe can be found
  "gdb_server_path": "C:/cubeIDE/STM32CubeIDE/plugins/com.st.stm32cube.ide.mcu.externaltools.stlink-gdb-server.win32_2.1.100.202309131603/tools/bin",
  "gcc_binary_path": "C:/cubeIDE/STM32CubeIDE/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.11.3.rel1.win32_1.1.100.202309151323/tools/bin/",
  "iar_binary_path": "C:/IAR/IAR9.30.1/common/bin/",
  // Full path to arm-none-eabi-objcopy.exe
  "objcopy_binary_path": "C:/cubeIDE/STM32CubeIDE/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.11.3.rel1.win32_1.1.100.202309151323/tools/bin/arm-none-eabi-objcopy.exe",
  // Cube Programmer binary path
  "cubeProgrammerCLI_binary_path": "C:/CubeProgN6/bin/STM32_Programmer_CLI.exe",
  // make path if not present in $PATH
  "make_binary_path": "C:/make/bin/make.exe",
  // Cube IDE path to use automatic detection of (gdb_server, gcc, objcopy, cubeProgrammer, make)
  "cubeide_path":"C:/Users/foobar/TOOLS/STM32CubeIDE_x.y.z/STM32CubeIDE"
}
```

Omitting a line for a tool (or commenting it) results in the tool searching in the system path as a last resort (e.g it is possible not to write the `"make_binary_path"` if `'make.exe'` is in the system path)

> **Warning**  
> Paths here shall use forward slashes (`/`): backslashes are forbidden in Windows-paths, one should replace them all by forward slashes.

> **Warning**  
> Some paths refer to executable files (represented by `.exe` suffix in the example) and some paths refer to directories (ending by `/` in the example).

In this section, either gcc or iar toolchain can be selected:

- For IAR users:
  - Set `"compiler_type"` to `"iar"`
  - Set `"iar_binary_path"` with the path containing IAR binaries (ending in bin)
- For GCC users:
  - Set `"compiler_type"` to `"gcc"`
  - Set `"gcc_binary_path"` with the path containing arm-none-eabi-gcc binaries (ending in bin)
  - Set `"gdb_server_path"` to the path containing stlink gdbserver (ending in bin)

> **Tip**  
> If a version of STM32CubeIDE (N6-compatible) is available, providing the path to this installation in the json file is a good practice, as it will be used to fill-in paths for gdb_server, gcc, objcopy, cubeprogrammer, make.

The file can be then shortened to something like:

```json
{
  "compiler_type": "gcc",
  "cubeide_path":"C:/Users/foobar/TOOLS/STM32CubeIDE_x.y.z/STM32CubeIDE"
}
```

or

```json
{
  "compiler_type": "iar",
  "iar_binary_path": "C:/IAR/IAR9.30.1/common/bin/",
  "cubeide_path":"C:/Users/foobar/TOOLS/STM32CubeIDE_x.y.z/STM32CubeIDE"
}
```

## Test application (config_n6l.json) {#ref_tools_config_n6l_json}

To use the `n6_loader.py` script, the location of the C-project used for validation, the location of the generated specialized files, and various options can be provided through a specific project configuration file: `config_n6l.json`. This file is in a json-format that allows comments.

```json
// This file is for configuring N6 loader, the util used to copy STEdgeAI outputs into a project,
// compile the project, and load the results on the board.
{
  // The 2lines below are _only used if you call n6_loader.py ALONE (memdump is optional and will be the parent dir of network.c by default)
  "network.c": "C:/st_ai_output/network.c",
  //"memdump_path": "C:/Users/foobar/CODE/stm.ai/stm32ai_output",
  // Location of the "validation" project  + build config name to be built (if applicable)
  "project_path": "C:/Projects/NPU_Validation",
  // If using the NPU_Validation project, valid build_conf names are "N6-DK", "N6-DK-USB", "N6-Nucleo", "N6-Nucleo-USB"
  "project_build_conf": "N6-DK",
  // Skip programming weights to earn time (but lose accuracy) -- useful for performance tests
  "skip_external_flash_programming": false,
  "skip_ram_data_programming": false
}
```

This file configures the behaviour of the `n6_loader.py` script.

- `"network.c"` shall indicate where to find the specialized “.c” file that was output from the STEdgeAI generate call.
- `"project_path"` shall indicate where the “NPU_Validation” is located on the user’s computer.
- `"project_build_config"` shall be kept to “N6-DK” when using a Discovery Kit board. (*And see [below for explanations about N6-DK-USB](stneuralart_getting_started.html#ref_troubleshoot_quickenvalid)*)
- `"skip_external_flash_programming"` and `"skip_ram_data_programming"` allow the user to prevent programming part of the data on board. The values for this field are `true` and `false`.

> **Warning**  
> Be aware, that to set the `"skip_xxx"` parameters to `"False"` hcan be useful to do a quick check, however, it results in bad accuracy. In some cases, it may even hang the inference.

When everything is set up, the syntax is:

```shell
python n6_loader.py --n6-loader-config ./config_n6l.json
```

When not provided, all the configuration fields are filled automatically by default values that consider that the script is in its installation location, with the NPU_Validation project in its default location,..

# Outputs of the generate command

When a call to ST Edge AI core CL generate is done, information is shown in the console:

- The ST Neural-ART compiler command line is shown (for later use if needed)
- If the ST Neural-ART compilation was successful, an overview of the memory mapping and forecast of the number of epochs is shown by the command line
- Then the list of generated files is shown

A successful compilation example when calling STEdgeAI is shown below (the output format is subject to changes).

```text
ST Edge AI Core v3.0.0
 >>>> EXECUTING NEURAL ART COMPILER
   /atonn_compiler/windows/atonn.exe -i "/stedgeai/st_ai_output/mnist_int8_io_i8_OE_3_3_1.onnx" --json-quant-file "/stedgeai/st_ai_output/mnist_int8_io_i8_OE_3_3_1_Q.json" -g "network.c" --load-mdesc "/atonn_compiler/configs/stm32n6.mdesc" --load-mpool "/stedgeai/new_root/my_mpools/stm32 n6.mpool" --save-mpool-file "/stedgeai/st_ai_ws/neural_art__network/stm32 n6.mpool" --out-dir-prefix "/stedgeai/st_ai_ws/neural_art__network/" --mvei --cache-maintenance --enable-virtual-mem-pools --native-float --Ocache-opt --all-buffers-info --output-info-file "c_info.json"
 <<<< DONE EXECUTING NEURAL ART COMPILER
 
 Exec/report summary (generate)
 --------------------------------------------------------------------------------------------------------------------
 model file         :   /stedgeai/models/mnist_int8_io_i8.tflite
 type               :
 c_name             :   Default
 options            :   allocate-inputs, allocate-outputs
 optimization       :   balanced
 target/series      :   stm32n6npu
 workspace dir      :   /stedgeai/st_ai_ws
 output dir         :   /stedgeai/st_ai_output
 model_fmt          :   sa/sa per tensor
 model_name         :   mnist_int8_io_i8
 model_hash         :   0xbe0a77fcd37f19c0f475d4e7bc5e94fc
 params #           :   20,410 items (20.00 KiB)
 --------------------------------------------------------------------------------------------------------------------
 input 1/1          :   'Input_0_out_0', int8(1x1x28x28x1), 784 Bytes, QLinear(0.003921569,-128,int8), activations
 output 1/1         :   'Quantize_14_out_0', int8(1x1x10), 10 Bytes, QLinear(0.003906250,-128,int8), activations
 macc               :   0
 weights (ro)       :   20,625 B (20.14 KiB) (1 segment) / -61,015(-74.7%) vs float model
 activations (rw)   :   4,060 B (3.96 KiB) (1 segment) *
 ram (total)        :   4,060 B (3.96 KiB) = 4,060 + 0 + 0
 --------------------------------------------------------------------------------------------------------------------
 (*) 'input'/'output' buffers can be used from the activations buffer

Computing AI RT data/code size (target=stm32n6npu)..

Compilation details
   ---------------------------------------------------------------------------------
Compiler version: 0.4.0-892
Compiler arguments:  -i /stedgeai/st_ai_output/mnist_int8_io_i8_OE_3_3_1.onnx --json-quant-file /stedgeai/st_ai_output/mnist_int8_io_i8_OE_3_3_1_Q.json -g network.c --load-mdesc /atonn_compiler/configs/stm32n6.mdesc --load-mpool /stedgeai/new_root/my_mpools/stm32 n6.mpool --save-mpool-file /stedgeai/st_ai_ws/neural_art__network/stm32 n6.mpool --out-dir-prefix /stedgeai/st_ai_ws/neural_art__network/ --mvei --cache-maintenance --enable-virtual-mem-pools --native-float --Ocache-opt --all-buffers-info --output-info-file c_info.json
====================================================================================
Memory usage information
   ---------------------------------------------------------------------------------
        flexMEM    [0x34000000 - 0x34000000]:          0  B /          0  B  (  0.00 % used) -- weights:          0  B (  0.00 % used)  activations:          0  B (  0.00 % used)    
        cpuRAM1    [0x34064000 - 0x34064000]:          0  B /          0  B  (  0.00 % used) -- weights:          0  B (  0.00 % used)  activations:          0  B (  0.00 % used)    
        cpuRAM2    [0x34100000 - 0x34200000]:          0  B /      1.000 MB  (  0.00 % used) -- weights:          0  B (  0.00 % used)  activations:          0  B (  0.00 % used)    
        npuRAM3    [0x34200000 - 0x34270000]:          0  B /    448.000 kB  (  0.00 % used) -- weights:          0  B (  0.00 % used)  activations:          0  B (  0.00 % used)    
        npuRAM4    [0x34270000 - 0x342E0000]:      3.965 kB /    448.000 kB  (  0.89 % used) -- weights:          0  B (  0.00 % used)  activations:      3.965 kB (  0.89 % used)    
        npuRAM5    [0x342E0000 - 0x34350000]:          0  B /    448.000 kB  (  0.00 % used) -- weights:          0  B (  0.00 % used)  activations:          0  B (  0.00 % used)    
        npuRAM6    [0x34350000 - 0x343C0000]:          0  B /    448.000 kB  (  0.00 % used) -- weights:          0  B (  0.00 % used)  activations:          0  B (  0.00 % used)    
        octoFlash  [0x70000000 - 0x74000000]:     20.142 kB /     64.000 MB  (  0.03 % used) -- weights:     20.142 kB (  0.03 % used)  activations:          0  B (  0.00 % used)    
        hyperRAM   [0x90000000 - 0x92000000]:          0  B /     32.000 MB  (  0.00 % used) -- weights:          0  B (  0.00 % used)  activations:          0  B (  0.00 % used)    

Total:                                            24.106 kB                                  -- weights:     20.142 kB                  activations:      3.965 kB
====================================================================================
Used memory ranges
   ---------------------------------------------------------------------------------
        npuRAM4    [0x34270000 - 0x342E0000]: 0x34270000-0x34270FE0
        octoFlash  [0x70000000 - 0x74000000]: 0x70000000-0x700050A0
====================================================================================
Epochs details
   ---------------------------------------------------------------------------------
Total number of epochs                               5
>> pure software (SW) epochs                         1
>> hybrid epochs (using both software and hardware)  0
>> pure hardware (HW or EC) epochs                   4  (implemented in 0 epoch controller blobs/meta epochs)

+----------+------+---------+
| epoch ID | Type | Details |
+----------+------+---------+
| epoch_2  |  HW  |         |
| epoch_3  |  HW  |         |
| epoch_4  |  HW  |         |
| epoch_5  |  HW  |         |
| epoch_6  |  SW  | Softmax |
+----------+------+---------+
====================================================================================
 Requested memory size by section - "stm32n6npu" target
 ------------------- -------- -------- ------ -------
 module                  text   rodata   data     bss
 ------------------- -------- -------- ------ -------
 network_runtime.a     13,488        0      0       0
 network.o                580    4,441    488       0
 lib (toolchain)*           0        0      0       0
 ll atonn runtime       7,234    2,305      0      29
 ------------------- -------- -------- ------ -------
 RT total**            21,302    6,746    488      29
 ------------------- -------- -------- ------ -------
 weights                    0   20,625      0       0
 activations                0        0      0   4,060
 io                         0        0      0     794
 ------------------- -------- -------- ------ -------
 TOTAL                 21,302   27,371    488   4,883
 ------------------- -------- -------- ------ -------
 *  toolchain objects (libm/libgcc*)
 ** RT AI runtime objects (kernels+infrastructure)

  Summary - "stm32n6npu" target
  --------------------------------------------------
               FLASH (ro)      %*   RAM (rw)      %
  --------------------------------------------------
  RT total         28,536   58.0%        517   9.6%
  --------------------------------------------------
  TOTAL            49,161              5,371
  --------------------------------------------------
  *  rt/total

 Generated files (5)
 -----------------------------------------------------------------------------
<output-directory-path>/mnist_int8_io_i8_OE_3_0_0.onnx
<output-directory-path>/mnist_int8_io_i8_OE_3_0_0_Q.json
<output-directory-path>/network.c
<output-directory-path>\network.h
<output-directory-path>\network_c_info.json
<output-directory-path>/network_atonbuf.xSPI2.raw
<output-directory-path>\stai_network.c
<output-directory-path>\stai_network.h

Creating txt report file <output-folder>/network_generate_report.txt
elapsed time (generate): 21.563s
```

Only the `network.c` of the “Generated files” section above contains the c-file to be added to an STM32N6-Neural-ART-c-project.  
All the “raw” files contain [memory initializers](stneuralart_memory_initializers.html) for weights (and activations) that shall be programmed to memory before doing an inference.

# Troubleshooting

## Basic checklist when an issue occurs when following the steps above

- Ensure that the board is properly powered up (The discovery-kit board is a bit power-hungry and needs *a good USB port and a good USB cable*)
- Ensure the switches on the board are properly set up (the only switches being SW1 and SW2) + the jumper JP2 on the discovery-kit.
- Ensure all the software/firmware used come from the *same* delivery of tools (the backwards-compatibility is not ensured and using, e.g., old validation projects will most likely result in full failure)
- Ensure that no errors have occurred during any of the steps:
  - Generation
    - Check that the STM32N6-NeuralArt options have been correctly set `--target stm32n6 --st-neural-art ...`
    - Check that there was no error at all
  - n6_loader
    - Check the toolchain used is correct (configured in the `config.json` file)
    - Check the files copied are correct (source / destination found in the `config_n6l.json` file)
    - Check that there is no error like “command not found”
    - If applicable, check that the external flash programming is done without errors
    - Check that the compilation goes through without errors (or check `compile.log`)
    - Check that there is a success message at the end
    - Extra, set `skip_external_flash_programming` and `skip_ram_data_programming` to “false” in `config_n6l.json` file
  - Validation
    - Ensure that the connection speed has been specified on the command line: `--desc serial:921600`
    - For connection issues, see below

## Cannot connect to the board / connection issues

When doing a validation on target, it is possible that some connection issues appear. This can be due to multiple causes, but the main ones are

- Another software is currently using the UART (for example, terminal software connected to the ST-Link UART)
  - Disconnect the other UART snooper.
- Multiple ST boards are connected on the computer, and the automatic discovery fails (`invalid firmware`)
- OR some other peripherals are connected to the computer and leads to issues when doing automatic discovery of ST-Link (infinite loop, with no timeout when trying to connect to the board)
  - Help `stedgeai` application to find the correct port, `--desc serial:921600` asks the validation to be done over UART, with a speed of 921600 bps (as set up by the default validation project). To force the use of a given port, the syntax is as follows (for example, using windows, and forcing communication on COM17) `--desc serial:COM17:921600`.

## Timeout (50000 ms) during validation

Such a timeout signals an inference that has been started properly but that hangs forever. Though this should never happen, double check the checklist above, and report a problem with the files used as inputs, the command lines used and the output generated files.

## Validating my model takes forever… {#ref_troubleshoot_quickenvalid}

### Speed-up the communication

The validation process can be time-consuming because a large amount of data is sent and received through UART, especially for models with large input tensors. To speed up serial communication, you can use the [board’s USB connector](#ref_use_usbc_link):

- Connect a USB cable from the `'USB1'` connector on the Discovery-Kit board (or `'USB'` connector on the Nucleo board) to the computer used for validation
- Choose the `N6-DK-USB` (or `N6-Nucleo-USB`) build configuration for the test application (also called NPU_Validation project) (this can be done in the `n6_loader` configuration file: `project_build_conf`)
- When validating, `--desc serial` will use the maximum usable speed. If you explicitly set a baud rate, it will be ignored.

### Quicken model analysis

Part of the time taken for validation is due to the fact that ST Edge AI has to find information about the model it is about to validate, to generate proper inputs. After doing a `generate` step, all the information about the input/output tensors needed for the validation are contained in the [c_info file](network_c_info_json.html). This file can be favorably used to speed up things during validation.

This is done by using the `--val-json <path_to_c_info>` when calling the validation step.
