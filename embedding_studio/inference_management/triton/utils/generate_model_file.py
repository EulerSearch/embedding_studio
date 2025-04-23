import inspect
from typing import List

from torch import nn


def get_imports_from_modules(sequential_model) -> List[str]:
    """
    Extract necessary imports from the modules used in the sequential model.

    This function analyzes a PyTorch model to determine all the module classes used
    and generates import statements for them.

    :param sequential_model: A PyTorch model to analyze for module imports
    :return: A sorted list of import statements needed for the model
    """
    imports = set()  # Use a set to avoid duplicate imports

    # Iterate through all modules in the model
    for module in sequential_model.modules():
        module_class = module.__class__.__name__  # Get the class name
        module_module = module.__module__  # Get the module (package) name
        module_import = f"from {module_module} import {module_class}"

        # Only add imports that aren't from __main__ or builtins
        if module_module != "__main__" and module_module != "builtins":
            # Check if we already have an import for this class
            if not any(module_class in imp for imp in imports):
                imports.add(module_import)

    # Return a sorted list of imports for consistent ordering
    return sorted(list(imports))


def generate_model_py(
    model: nn.Module,
    filename: str,
    embedding_studio_path: str = "/embedding_studio",
):
    """
    Generate a Python script file that can reconstruct a PyTorch model.

    This function creates a standalone Python file that can rebuild the given model
    when loaded by Triton Inference Server. It handles non-Sequential models.

    :param model: The PyTorch model to generate code for
    :param filename: Path where the generated Python script will be saved
    :param embedding_studio_path: Path to the embedding_studio package for imports
    :return: None
    """
    # Start building the model.py file content with necessary imports
    model_code = f'import sys\nsys.path.append("{embedding_studio_path}")\nimport torch\nfrom torch import nn\n'
    # Add all required module imports from the model
    model_code += "\n".join(get_imports_from_modules(model))
    model_code += "\n\n"

    # Add import for configuration classes if any modules use them
    extra_imports = set()
    for module in model.modules():
        if hasattr(module, "config"):
            config_class_name = type(module.config).__name__
            config_module_name = module.config.__class__.__module__
            import_statement = (
                f"from {config_module_name} import {config_class_name}"
            )
            extra_imports.add(import_statement)
    for import_statement in extra_imports:
        model_code += f"{import_statement}\n"

    # Define a function to recursively add modules to the model code
    def add_modules(layer, layer_name, indent_level=2):
        """
        Recursively build code to reconstruct model layers.

        :param layer: The layer or module to generate code for
        :param layer_name: Name to give this layer in the generated code
        :param indent_level: Current indentation level for formatting
        """
        nonlocal model_code
        indent = "    " * indent_level  # Calculate indentation

        # If this is a container module that has children
        if isinstance(layer, nn.Module):
            model_code += (
                f"{indent}self.add_module('{layer_name}', nn.Module())\n"
            )
            # Recursively process children
            for name, sub_layer in layer.named_children():
                add_modules(
                    sub_layer, f"{layer_name}_{name}", indent_level + 1
                )
        else:
            # Process leaf modules (actual layers)
            args = inspect.getfullargspec(layer.__init__).args
            # Handle models that require config objects
            if "config" in args:
                config_index = args.index("config")
                config_class = layer.__init__.__annotations__.get(
                    args[config_index]
                )
                if config_class:
                    # Create an instance of the config class with default values
                    config_instance = config_class()
                    config_args = ", ".join(
                        [
                            f"{k}={v!r}"
                            for k, v in vars(config_instance).items()
                        ]
                    )
                    model_code += f"{indent}config = {config_class.__name__}({config_args})\n"
                    model_code += f"{indent}self.add_module('{layer_name}', {layer.__class__.__name__}(config))\n"
                else:
                    model_code += f"{indent}self.add_module('{layer_name}', {layer.__class__.__name__}())\n"
            else:
                # Try to get constructor arguments from the layer's attributes
                layer_args = inspect.getfullargspec(layer.__init__).args
                default_args = {
                    arg: getattr(layer, arg)
                    for arg in layer_args
                    if arg != "self" and hasattr(layer, arg)
                }
                default_arg_str = ", ".join(
                    [f"{k}={v!r}" for k, v in default_args.items()]
                )
                model_code += f"{indent}self.add_module('{layer_name}', {layer.__class__.__name__}({default_arg_str}))\n"

    # Define the main model class that Triton will load
    model_code += "class TritonPythonModel(nn.Module):\n"
    model_code += "    def __init__(self):\n"
    model_code += "        super().__init__()\n\n"

    # Add all modules to the TritonPythonModel class
    add_modules(model, "model")

    # Write the completed code to the specified file
    with open(filename, "w") as f:
        f.write(model_code)


def generate_sequential_model_py(
    sequential_model: nn.Sequential,
    filename: str,
    embedding_studio_path: str = "/embedding_studio",
):
    """
    Generate a Python script file that can reconstruct a PyTorch Sequential model.

    This function creates a standalone Python file that can rebuild the given Sequential
    model when loaded by Triton. It's specialized for handling Sequential models.

    :param sequential_model: The PyTorch Sequential model to generate code for
    :param filename: Path where the generated Python script will be saved
    :param embedding_studio_path: Path to the embedding_studio package for imports
    :return: None
    """
    # Start building the model.py file content with necessary imports
    model_code = f'import sys\nsys.path.append("{embedding_studio_path}")\nimport torch\nfrom torch import nn\n'
    # Add all required module imports based on what's in the model
    model_code += "\n".join(get_imports_from_modules(sequential_model))
    model_code += "\n\n"

    # Add import for configuration classes if any modules use them
    extra_imports = set()
    for module in sequential_model.modules():
        if hasattr(module, "config"):
            config_class_name = type(module.config).__name__
            config_module_name = module.config.__class__.__module__
            import_statement = (
                f"from {config_module_name} import {config_class_name}"
            )
            extra_imports.add(import_statement)
    for import_statement in extra_imports:
        model_code += f"{import_statement}\n"

    # Define a function to recursively add modules to the model code
    def add_modules(layer, layer_name, indent_level=2):
        """
        Recursively build code to reconstruct model layers.

        :param layer: The layer or module to generate code for
        :param layer_name: Name to give this layer in the generated code
        :param indent_level: Current indentation level for formatting
        """
        nonlocal model_code
        indent = "    " * indent_level  # Calculate indentation

        # If this is another Sequential container
        if isinstance(layer, nn.Sequential):
            model_code += (
                f"{indent}self.add_module('{layer_name}', nn.Sequential())\n"
            )
            # Recursively process each layer in the sequence
            for idx, sub_layer in enumerate(layer):
                add_modules(sub_layer, f"{layer_name}_{idx}", indent_level + 1)
        else:
            # Process individual layers
            args = inspect.getfullargspec(layer.__init__).args
            # Handle models that require config objects
            if "config" in args:
                config_index = args.index("config")
                config_class = layer.__init__.__annotations__.get(
                    args[config_index]
                )
                if config_class:
                    # Create a default config object and extract its parameters
                    config_instance = config_class()
                    config_args = ", ".join(
                        [
                            f"{k}={v!r}"
                            for k, v in vars(config_instance).items()
                        ]
                    )
                    model_code += f"{indent}config = {config_class.__name__}({config_args})\n"
                    model_code += f"{indent}self.add_module('{layer_name}', {layer.__class__.__name__}(config))\n"
                else:
                    model_code += f"{indent}self.add_module('{layer_name}', {layer.__class__.__name__}())\n"
            else:
                # Extract constructor arguments from the layer's attributes
                layer_args = inspect.getfullargspec(layer.__init__).args
                default_args = {
                    arg: getattr(layer, arg)
                    for arg in layer_args
                    if arg != "self" and hasattr(layer, arg)
                }
                default_arg_str = ", ".join(
                    [f"{k}={v!r}" for k, v in default_args.items()]
                )
                model_code += f"{indent}self.add_module('{layer_name}', {layer.__class__.__name__}({default_arg_str}))\n"

    # Define the main Sequential model class that Triton will load
    model_code += "class TritonPythonModel(nn.Sequential):\n"
    model_code += "    def __init__(self):\n"
    model_code += "        super().__init__()\n\n"

    # Add each layer in the Sequential model
    for idx, layer in enumerate(sequential_model):
        add_modules(layer, f"{idx}")

    # Write the completed code to the specified file
    with open(filename, "w") as f:
        f.write(model_code)
