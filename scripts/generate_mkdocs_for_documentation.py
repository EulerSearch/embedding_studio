from pathlib import Path
from collections import defaultdict
import yaml


def build_tree(base_dir: Path, root_prefix: str):
    tree = lambda: defaultdict(tree)
    file_tree = tree()

    for path in sorted(base_dir.rglob("*.md")):
        relative_path = path.relative_to(root_prefix)
        parts = relative_path.parts
        current = file_tree
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1].replace('.py.md', '')] = str(path).replace('docs/', '')

    return file_tree


def convert_to_yaml_style(tree):
    def walk(subtree):
        output = []
        for key, value in sorted(subtree.items()):
            if isinstance(value, str):
                # File entry
                output.append({key: value})
            else:
                # Folder, recurse
                output.append({key: walk(value)})
        return output

    return walk(tree)


if __name__ == "__main__":
    docs_root = Path("docs/embedding_studio")
    tree = build_tree(docs_root, docs_root)

    nav_structure = convert_to_yaml_style(tree)

    with open("pretty_nav_output.yml", "w") as f:
        yaml.dump(nav_structure, f, sort_keys=False, allow_unicode=True)

    print("✅ Prettified nav written to pretty_nav_output.yml")
