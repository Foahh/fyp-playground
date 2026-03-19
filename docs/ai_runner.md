# How to use the AiRunner package

**ST Edge AI Core Technology 4.0.0**  
**r1.0**

This file is converted from "STEdgeAI/4.0/Documentation/ai_runner_python_module.html"

---

## Overview

This article explains how to use the `stm_ai_runner` Python package, also known as `ai_runner`, to profile and validate a deployed C-model. As illustrated in the following figure, the model can be deployed either on a target device or on the host. The AiRunner object provides a simple and unified interface for inference and profiling, allowing users to inject data, execute inference, and retrieve predictions.

The `stm_ai_runner` Python package is also integrated into the ST Edge Core AI CLI, used by the [`validate`](command_line_interface.html#validate-command) command. It can also be used independently to extend the default validation process. End-users such as data scientists or ML/AI designers can update, with minor adaptations, classical validation Python scripts to validate the deployed model with real datasets and metrics.

Multiple back-ends are supported. In this article, two main configurations are considered:

- **Execution on host**  
  Through the `generate` command, the user can create a shared library, or DLL, using the [`--dll`](stm32_command_line_interface.html#ref_generate_command_ext) option, containing the specialized C files. This shared library exports the embedded C-API functions, either the [legacy](embedded_client_api_legacy.html) API or the [st-ai](embedded_client_stai_api.html) API, which are bound in the Python environment to export a common API.

- **Execution on a physical target**  
  The specialized files are linked with a generic embedded test application, also called the aiValidation application. On the host side, a simple message-based protocol on top of a serial protocol exposes a set of services for discovering and using the deployed models.

## Setting up a work environment

The following Python packages should be installed in a Python 3.x environment to use the `stm_ai_runner` package. It is recommended to use a virtual environment.

```text
protobuf<3.21
tqdm
colorama
pyserial
numpy
```

To be able to import the `stm_ai_runner` package, set the `PYTHONPATH` environment variable:

```bash
export PYTHONPATH=$STEDGEAI_CORE_DIR/scripts/ai_runner:$PYTHONPATH
```

> `%STEDGEAI_CORE_DIR%` represents the root location where the ST Edge AI Core components are installed, typically in a path like `"<tools_dir>/STEdgeAI/2.1/"`.

> **Tip**  
> The `stm_ai_runner` package communicates with the board using a protocol based on the `Nanopb` module version 0.3.x. `Nanopb` is a plain-C implementation of Google’s Protocol Buffers data format. The `stm_ai_runner` package is fully compatible with protobuf versions below 3.21. For more information, visit the [Nanopb website](https://jpa.kapsi.fi/nanopb/).  
>  
> If a more recent version of `protobuf` is required and the package cannot be downgraded, the following environment variable can be used:
>
> ```bash
> export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
> ```

## Generating the model for execution on host

To generate a model that can be executed on the host machine, use the [`--dll`](stm32_command_line_interface.html#ref_stm32_shared_lib_option) option. By default, the shared library is generated in the `<workspace-directory-path>\inspector_network\workspace\lib\` folder.

```bash
$ stedgeai generate -m <model_path> --target stm32 --c-api st-ai --dll
...

 Generated files (6)
 ------------------------------------------------------------------------------------------
 <workspace-directory-path>\inspector_network\workspace\generated\network_data.h
 <workspace-directory-path>\inspector_network\workspace\generated\network_data.c
 <workspace-directory-path>\inspector_network\workspace\generated\network_details.h
 <workspace-directory-path>\inspector_network\workspace\generated\network.h
 <workspace-directory-path>\inspector_network\workspace\generated\network.c
 <workspace-directory-path>\inspector_network\workspace\lib\libai_network.dll

Creating txt report file <output-directory-path>\network_generate_report.txt
```

> **Note**  
> The specialized C files used to generate the shared library are also generated in the same directory.

To check the generated shared library, the [`validate`](command_line_interface.html#validate-command) command can be used with the `-d file:st_ai_ws` option. This indicates that the `libai_network.dll` from the `st_ai_ws` folder should be used.

```bash
$ stedgeai validate -m <model_path> --target stm32 --mode target -d file:st_ai_ws
```

The `checker.py` script can also be used without options:

```bash
$ python $STEDGEAI_CORE_DIR/scripts/ai_runner/examples/checker.py
```

## Generating the model for execution on a physical target

To use the `stm_ai_runner` Python package with a model executing on a physical target, the target should be flashed with firmware that includes the generic built-in aiValidation application and the specialized C files.

For the `stm32n6` target, which requires NPU support, a quick and typical process is described in the article [_How to evaluate a model on an STM32N6 board_](stneuralart_getting_started.html). For other `stm32xx` targets, the X-CUBE-AI UI plug-in can be leveraged, as detailed in the user manual [_Getting started with X-CUBE-AI Expansion Package for Artificial Intelligence (AI)_](https://www.st.com/resource/en/user_manual/dm00570145.pdf).

To check the deployed C-model on the target, the [`validate`](command_line_interface.html#validate-command) command can be used with the `-d/--desc serial` option.

```bash
$ stedgeai validate -m <model_path> --target stm32 --mode target -d serial
```

The `checker.py` script can also be used with the `-d/--desc serial` option:

```bash
$ python $STEDGEAI_CORE_DIR/scripts/ai_runner/examples/checker.py -d serial
```

By default, for STM32N6 board, the `-d serial:921600` option should be used.

## Getting started - Minimal script

The following code shows a minimal script to perform model inference with random input data running on a physical target and to display profiling information.

```python
import sys
import argparse

from stm_ai_runner import AiRunner

desc = 'serial'

# create AiRunner object
runner = AiRunner()
# connection
runner.connect(desc)
# display and retrieve model info (optional)
runner.summary()
model_info: dict = runner.get_info()
input_details: list[dict] = runner.get_input_infos()  # = model_info['inputs']
output_details: list[dict] = runner.get_output_infos()  # = model_info['outputs']
# generate the random input data
inputs: list[np.ndarray] = runner.generate_rnd_inputs(batch_size=2)
# perform the inference
mode: AiRunner.Mode = AiRunner.Mode.PER_LAYER
outputs, profiler = runner.invoke(inputs, mode=mode)
# display the profiling info
runner.print_profiling(inputs, profiler, outputs)
# disconnect
runner.disconnect()
```

This excerpt is part of the `$STEDGEAI_CORE_DIR/scripts/ai_runner/examples` folder. See `minimal.py` and `checker.py`.

## AiRunner API

The `$STEDGEAI_CORE_DIR/scripts/ai_runner/examples` folder provides different simple scripts using the AiRunner API.

## Connection

### `connect()`

The `connect()` method allows binding an AiRunner object to a given ST AI runtime. The `desc` parameter specifies the back-end or driver to use.

```python
import sys
from stm_ai_runner import AiRunner

desc = ...

runner = AiRunner()
runner.connect(desc)

if not runner.is_connected:
    print('No c-model available, use the --desc/-d option to specifiy a valid path/descriptor')
    print(f' {runner.get_error()}')
    sys.exit(1)
...
```

#### `desc` parameter

Format:

```text
<protocol/backend>[:<parameters>]
```

The first part of the descriptor defines the back-end or driver used to perform the connection with a runtime embedding one or more deployed models. The definition of the `parameters` field is driver-specific.

| back-end/driver | description |
|---|---|
| `lib:parameter` | Used to bind a shared library exporting the embedded C-API. The `parameter` argument indicates the full file path or the root folder containing the shared library, for example `lib:./my_model`. Note that the `lib:` field can be omitted if a valid folder or file is provided. |
| `serial[:parameter]` | Used to open a connection with a physical target through a serial link. The target should be flashed with a specific built-in profiling application, the aiValidation application, embedding the deployed models. |

#### Parameters for the serial driver

Format:

```text
:<com port>[:<baud-rate>]
```

The parameter argument is optional. By default, an autodetection mechanism is applied to discover a connected board at 115200 baud, or 921600 for ISPU. The baud rate should match the value defined in the firmware.

- Set the baud rate to 921600:

  ```bash
  $ stedgeai ... -d serial:921600
  ```

- Set a specific COM port:

  ```bash
  $ stedgeai ... -d serial:COM4     # Windows environment
  $ stedgeai ... -d /dev/ttyACM0    # Linux-like environment
  ```

- Set both COM port and baud rate:

  ```bash
  $ stedgeai ... -d serial:COM4:921600
  ```

### Typical connection errors

- No shared library found. `desc` designates a folder without a valid shared library file.

  ```text
  invalid/unsupported "st_ai_ws/:" descriptor
  ```

- Provided generated shared library is invalid. The error message indicates that the shared library has been generated without weights. This can appear when the `validate` command has been performed in the default `./st_ai_ws/` directory.

  ```text
  E801(HwIOError): No weights are available (1549912 bytes expected)
  ```

- The STM32 board is not connected. Autodetect mode.

  ```text
  E801(HwIOError): No SERIAL COM port detected (STM board is not connected!)
  ```

- COM port is already opened by another application such as TeraTerm.

  ```text
  E801(HwIOError): could not open port 'COM6': PermissionError(13,
                       'Access is denied.', None, 5)
  ```

- STM32 board is not flashed with a valid aiValidation firmware.

  ```text
  E801(HwIOError): Invalid firmware - COM6:115200
  ```

### `names`

Multiple models can be deployed in a given AI runtime environment running on board. Each model should be deployed with a specific `c-name` used as a selector. The `names` method can be used to retrieve the list of available C-models.

```python
available_models: list[str] = runner.names
print(available_models)
# ['network0', 'network1', ...]
```

### `AiRunnerSession()`

To facilitate the use of a specific named C-model, the `session(name: Optional[str])` method returns a dedicated handler object called `AiRunnerSession`. This object provides the same methods as the `AiRunner` object for using a deployed model.

```python
runner = AiRunner()
runner.connect(desc)
...
session: AiRunnerSession = runner.session('network_2')
```

## Model information

### `get_info()`

The `get_info(name: Optional[str] = None)` method retrieves detailed information, in dict form, for a given model.

```python
model_info: dict = runner.get_info()  # equivalent to runner.get_info(available_models[0])
```

#### Model dict

| key | type | description |
|---|---|---|
| `version` | tuple | Version of the dict, `(2, 0)` |
| `name` | str | C-name of the model, `--name` option of the code generator |
| `compile_datetime` | str | Date-time when the model was compiled |
| `n_nodes` | int | Number of deployed C-nodes implementing the model |
| `inputs` | list[dict] | [Input tensor descriptions](#tensor-dict) |
| `outputs` | list[dict] | [Output tensor descriptions](#tensor-dict) |
| `hash` | Optional[str] | MD5 hash of the original model file |
| `weights` | Optional[int, list[int]] | Accumulated size in bytes of the weights/params buffers |
| `activations` | Optional[int, list[int]] | Accumulated size in bytes of the activations buffer |
| `macc` | Optional[int] | Equivalent number of [macc](evaluation_metrics.html#ref_complexity) |
| `rt` | str | Short description of the used AI runtime API |
| `runtime` | dict | Main properties of the AI runtime/environment |
| `device` | dict | Main properties of the device supporting the AI runtime |

#### Tensor dict

| key | type | description |
|---|---|---|
| `name` | str | Name of tensor, C-string |
| `shape` | tuple | Shape |
| `type` | np.dtype | Data type |
| `scale` | Optional[np.float32] | Scale value if quantized |
| `zero_point` | Optional[np.int32] | Zero-point value if quantized |

#### AI runtime/environment dict

| key | type | description |
|---|---|---|
| `protocol` | str | Description of the used back-end/driver |
| `name` | str | Short description of the used AI runtime API |
| `tools_version` | tuple | Version of the tools used to deploy the model |
| `rt_lib_desc` | str | Description of the AI runtime libraries used |
| `version` | tuple | Version of the AI runtime libraries used |
| `capabilities` | list[AiRunner.Caps] | Capabilities of the AI runtime |

Capabilities:

| capability | description |
|---|---|
| `AiRunner.Caps.IO_ONLY` | Minimal mandatory capability allowing data injection and prediction retrieval |
| `AiRunner.Caps.PER_LAYER` | Capability to report intermediate tensor information without data |
| `AiRunner.Caps.PER_LAYER_WITH_DATA` | Capability to report intermediate tensor information including data |

#### Device dict

| key | type | description |
|---|---|---|
| `dev_type` | str | Target name |
| `desc` | str | Short description of the device including main frequencies |
| `dev_id` | Optional[str] | Device ID, target-specific |
| `system` | str | Short description of the platform |
| `sys_clock` | Optional[int] | MCU frequency in Hz |
| `bus_clock` | Optional[int] | Main system bus frequency in Hz |
| `attrs` | Optional[list[str]] | Target-specific attributes |

### `get_input_infos()`, `get_output_infos()`

The `get_input_infos(name: Optional[str] = None)` and `get_output_infos(name: Optional[str] = None)` methods retrieve detailed information about the input and output tensors.

```python
model_inputs: list[dict] = runner.get_input_infos()   # equivalent to runner.get_input_infos(available_models[0])
model_outputs: list[dict] = runner.get_output_infos() # equivalent to runner.get_output_infos(available_models[0])
```

## Perform the inference

### `invoke()`

The `invoke(inputs: Union[np.ndarray, List[np.ndarray]])` method performs inference with the input data. It returns a tuple containing the predictions, `outputs`, and a Python dictionary with the profiling information, `profiler`.

```python
# perform the inference
mode: AiRunner.Mode = AiRunner.Mode.PER_LAYER
outputs, profiler = runner.invoke(inputs, mode=mode)
```

#### `mode` parameter

The `mode` parameter consists of OR-ed flags that set the AI runtime mode. It depends on the returned capabilities.

| mode | description |
|---|---|
| `AiRunner.Mode.IO_ONLY` | Standard execution. Only predictions are dumped; intermediate information is not reported |
| `AiRunner.Mode.PER_LAYER` | Descriptions of the intermediate nodes are reported without data |
| `AiRunner.Mode.PER_LAYER_WITH_DATA` | Intermediate data are also dumped when supported |
| `AiRunner.Mode.PERF_ONLY` | No input data are sent to the target, and results are not dumped |

#### Profiling dict

| key | type | description |
|---|---|---|
| `info` | dict | Model/AI runtime information. Refer to the model dict section |
| `mode` | `AiRunner.Mode` | Used mode |
| `c_durations` | List[float] | Inference time in ms by sample |
| `c_nodes` | Optional[List[dict]] | Profiled C-node information. One entry per node. `PER_LAYER` or `PER_LAYER_WITH_DATA` mode must be used |
| `debug` | str | C-name of the model |

> **Warning**  
> The returned profiling information depends on the AI runtime environment and/or target. For example, if the deployed model is executed on the host, information about cycles is not returned because such information is not relevant: the kernels are not optimized for the host or development machine.

#### C-node dict

| key | type | description |
|---|---|---|
| `name` | str | Name of the node |
| `m_id` | int | Optional associated index of the layer from the original model |
| `layer_type` | int | Runtime-specific node type ID |
| `layer_desc` | str | Short description of the node |
| `type` | List[np.ndarray] | Data type of the associated output tensors |
| `shape` | List[Tuple[int]] | Shape of the associated output tensors |
| `scale` | Optional[List[np.float32]] | Scale value if quantized |
| `zero_point` | Optional[List[np.int32]] | Zero-point value if quantized |
| `c_durations` | List[float] | Inference time of the node in ms by sample |
| `clks` | Optional[List[Union[int, list[int]]]] | Number of MCU/CPU clocks to execute the node, depending on AI runtime/target |
| `data` | Optional[List[np.ndarray]] | When available through `AiRunner.Caps.PER_LAYER_WITH_DATA`, dumped output tensor data after each node execution |

## Services

### `summary()`

The `summary(name: Optional[str] = None)` method displays a summary of the information provided by [`get_info(name: Optional[str] = None)`](#get_info).

```python
runner.summary()
```

- Summary output - host mode
- Summary output - target mode

### `generate_rnd_inputs()`

The `AiRunner.generate_rnd_inputs(name: Optional[str])` method is a helper service that generates input data for a given model. The `val` parameter sets the range of the uniformly distributed data over the interval `[low, high[`. The default is `[-1.0, 1.0[` for floating-point types.

```python
inputs: Union[np.ndarray, List[np.ndarray]] = runner.generate_rnd_inputs(name='network', batch_size=2)
```

### `print_profiling()`

The `print_profiling(inputs, profiler, outputs)` method displays a summary of the profiling information returned by [`invoke()`](#invoke).

```python
# perform the inference
outputs, profiler = runner.invoke(inputs)
# display the profiling info
runner.print_profiling(inputs, profiler, outputs)
```

- Profiling output - target mode

## Examples

Location: `%STEDGEAI_CORE_DIR%/script/ai_runner/example/`

- `checker.py` provides an example of using the `ai_runner` module including profiling information.

  ```bash
  # Try to load the shared library located in the default location: ./st_ai_ws.
  # It displays a summary and performs two inferences with random data.
  $ python checker.py

  # As previously, but it performs a connection with a STM32 board (auto-detect mode)
  $ python checker.py -d serial

  # Set the expected COM port and baudrate
  $ python checker.py -d serial:COM6:115200
  ```

- `tflite_test.py` provides a typical example to compare the outputs of the generated C-model against the predictions from `tf.lite.Interpreter`.

- `mnist` provides a complete example with two scripts allowing you to train a model (`train.py`) and test (`test.py`) with the generated C-model.
