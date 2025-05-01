# Scientific Computing Model Benchmarking Framework

This project provides a framework for benchmarking the performance (throughput, latency, memory usage) of various machine learning models designed for scientific computing, especially those processing unstructured grid data.

## Requirements

* Python 3.x
* PyTorch
* OmegaConf
* NumPy
* Xarray
* Pandas

## Installation and Setup

1. **Clone the repository:**

   ```
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Configure Dataset Path:** Edit the `config/setup_config.json` file. Update `dataset.base_path` to point to the directory where your datasets (in `.nc` format) are stored. You can also adjust the dataset `name`, data splits (`train_size`, `val_size`, `test_size`), `batch_size`, etc., here.

3. **Configure Compute Device:** Set the desired compute device (e.g., `"cuda:0"` or `"cpu"`) in the `basics` section of `config/setup_config.json`.

## Adding Your Own Model

To benchmark your own PyTorch model within this framework, follow these steps:

1. **Model Implementation:**

   - Place your model's Python code in the `src/model/` directory (or a subdirectory).
   - Your model should be a subclass of `torch.nn.Module`.
   - Ensure its `forward` method accepts the input tensors provided by the data loader. Refer to the `_step` function in `benchmark_script.py` and the `forward` method signatures of existing models (like GINO, Transolver) to understand the parameters your model needs to accept (this might include conditional inputs `c`, coordinates `x`, dictionaries with specific keys, etc., depending on your data and the adapter used).

2. **Register Your Model:**

   - Open the `src/model/__init__.py` file.

   - Add the name of your model (as a string) to the `supported_models` list.

   - Add an `elif` branch within the `init_model` function to handle the instantiation of your model class. You will need to pass `input_size`, `output_size`, and the model-specific configuration (`config`) loaded from its JSON file.

     ```
     # In src/model/__init__.py
     
     # Import your model class
     from .your_model_file import YourModelClass # Assuming your model is in your_model_file.py
     
     supported_models = [
         "transolver",
         "gnot",
         "goat2d_fx",
         "gino",
         "your_model_name" # <-- Add your model name here
     ]
     
     def init_model(
             input_size:int = None,
             output_size:int = None,
             model:str = "transolver",
             config:Optional[dataclass] = None # config comes from the model's JSON file
                     ):
         # ... existing initialization code for other models ...
     
         elif model.lower() == "your_model_name":
             # If your model also uses OmegaConf for config management (optional):
             # your_config = merge_config(YourModelClass.ModelConfig, config)
             # return YourModelClass(your_config)
     
             # Or, if passing the 'args' object from JSON directly:
             return YourModelClass(
                 input_channels=input_size,  # Or the input parameter name your model expects
                 output_channels=output_size, # Or the output parameter name your model expects
                 **config # Pass all parameters under 'args' from the JSON file to the model constructor
             )
         else:
             raise ValueError(f"model {model} not supported currently!")
     ```

3. **Create Model Configuration File:**

   - Create a new JSON file for your model in the `config/model_config/` directory (e.g., `your_model_name.json`).
   - Refer to the structure of existing configuration files (like `gino.json` or `transolver.json`).
   - Define at least the following fields:
     - `model.name`: Must exactly match the model name you used in `src/model/__init__.py`.
     - `model.input_channels`: The number of input channels your model expects.
     - `model.output_channels`: The number of output channels your model produces.
     - `model.adapter`: (Optional) If your model requires specific input/output formatting different from the default, specify an adapter name (defined in `src/utils/io_adapter.py`). For example, `gino` and `goat` use their own adapters. Use `"default"` if no special handling is needed.
     - `model.args`: An object containing all hyperparameters and specific arguments required by your model's `__init__` function. These parameters will be passed to your model via the `init_model` function.

## Data Handling

- The framework expects data in NetCDF (`.nc`) format.
- Dataset loading, preprocessing (including normalization based on the training set), and splitting are handled in `src/data/dataset.py`.
- Dataset metadata (like variable names, group names, domain ranges) is defined in the `DATASET_METADATA` dictionary within `src/data/dataset.py`. Ensure your dataset has a corresponding entry, or modify the loading logic as needed.
- The currently used dataset and its parameters are selected in `config/setup_config.json`.

## Running Benchmarks

The main script for running benchmarks is `benchmark_script.py`.

1. **Single Model Benchmark:** To run the benchmark for a specific model configuration:

   ```
   python benchmark_script.py -c config/model_config/your_model_name.json
   ```

   This will perform the following actions:

   - Load the specified dataset and model configuration.
   - Initialize the model.
   - Find the maximum possible batch size for inference and training modes without causing Out-Of-Memory (OOM) errors.
   - Measure inference and training throughput (samples/second) at the determined maximum batch sizes.
   - Measure inference and training latency (ms/sample or ms/step) at batch size 1.
   - Measure peak GPU memory usage for each scenario.
   - Print the results to the console and save them to a CSV file (e.g., `results/test/your_model_name.csv`).

2. **Scalability Sweeps:** To run scalability tests (varying input grid sizes and/or model parameters):

   - Define your sweep configurations within the `SWEEPS` dictionary in `benchmark_script.py`. Structure your sweep under your model's name (e.g., `"YourModelName"`). Define different sweep types like `"model_scale"` or `"input_scale"`, specifying `grid_sizes` and `model_variants` (list of dictionaries with parameters to override from the base config).

   - Run the script with the `--scalability` flag, specifying the base model config:

     ```
     python benchmark_script.py -c config/model_config/your_model_name.json --scalability
     ```

   - The script will iterate through the defined grid sizes and model variants for the specified base model, running performance benchmarks for each combination.

   - The results of scalability sweeps are saved or appended to a CSV file in the `results/` directory (e.g., `results/YourModelName.csv`). You can use the `plot_scalability.py` script to visualize these results.

## Adapters

The `src/utils/io_adapter.py` file defines adapters. Adapters bridge the gap between the standard data format provided by the DataLoader (often a tuple or dictionary containing `inputs`, `labels`, `coords`) and the specific input format expected by a model's `forward` method (often a dictionary with specific keys).

- **Collate:** Organizes a list of samples into the batch structure required by the model (usually a dictionary).
- **to_device:** Moves the collated batch data to the specified compute device.

If your model accepts inputs in a non-standard way (e.g., requires a dictionary with specific key names), you might need to create a custom adapter class (inheriting from `IOAdapter`), register it in the `_ADAPTERS` dictionary, and specify its name in the `model.adapter` field of your model's JSON configuration file.