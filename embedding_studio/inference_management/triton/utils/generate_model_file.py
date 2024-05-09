import inspect
from typing import List

from torch import nn

# TODO: All code generation needs to be transferred directly into the template.

def get_imports_from_modules(sequential_model) -> List[str]:
    """
    Extract necessary imports from the modules used in the sequential model.
    """
    imports = set()
    for module in sequential_model.modules():
        module_class = module.__class__.__name__
        module_module = module.__module__
        module_import = f"from {module_module} import {module_class}"
        if module_module != "__main__" and module_module != "builtins":
            if not any(module_class in imp for imp in imports):
                imports.add(module_import)
    return sorted(list(imports))


def generate_model_py(
    model: nn.Module,
    filename: str,
    embedding_studio_path: str = "/embedding_studio",
):
    # Start building the model.py file content
    model_code = f'import sys\nsys.path.append("{embedding_studio_path}")\nimport torch\nfrom torch import nn\n'
    model_code += "\n".join(get_imports_from_modules(model))
    model_code += "\n\n"

    # Add import for configuration class
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
        nonlocal model_code
        indent = "    " * indent_level
        if isinstance(layer, nn.Module):
            model_code += (
                f"{indent}self.add_module('{layer_name}', nn.Module())\n"
            )
            for name, sub_layer in layer.named_children():
                add_modules(
                    sub_layer, f"{layer_name}_{name}", indent_level + 1
                )
        else:
            args = inspect.getfullargspec(layer.__init__).args
            if "config" in args:
                config_index = args.index("config")
                config_class = layer.__init__.__annotations__.get(
                    args[config_index]
                )
                if config_class:
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

    model_code += "class TritonPythonModel(nn.Module):\n"
    model_code += "    def __init__(self):\n"
    model_code += "        super().__init__()\n\n"

    # Add layers to the TritonPythonModel class
    add_modules(model, "model")

    # Write the code to a file
    with open(filename, "w") as f:
        f.write(model_code)


def generate_sequential_model_py(
    sequential_model: nn.Sequential,
    filename: str,
    embedding_studio_path: str = "/embedding_studio",
):
    # Start building the model.py file content
    model_code = f'import sys\nsys.path.append("{embedding_studio_path}")\nimport torch\nfrom torch import nn\n'
    model_code += "\n".join(get_imports_from_modules(sequential_model))
    model_code += "\n\n"

    # Add import for configuration class
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
        nonlocal model_code
        indent = "    " * indent_level
        if isinstance(layer, nn.Sequential):
            model_code += (
                f"{indent}self.add_module('{layer_name}', nn.Sequential())\n"
            )
            for idx, sub_layer in enumerate(layer):
                add_modules(sub_layer, f"{layer_name}_{idx}", indent_level + 1)
        else:
            args = inspect.getfullargspec(layer.__init__).args
            if "config" in args:
                config_index = args.index("config")
                config_class = layer.__init__.__annotations__.get(
                    args[config_index]
                )
                if config_class:
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

    model_code += "class TritonPythonModel(nn.Sequential):\n"
    model_code += "    def __init__(self):\n"
    model_code += "        super().__init__()\n\n"

    # Add layers to the TritonPythonModel class
    for idx, layer in enumerate(sequential_model):
        add_modules(layer, f"{idx}")

    # Write the code to a file
    with open(filename, "w") as f:
        f.write(model_code)
