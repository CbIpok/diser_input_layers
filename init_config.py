import os
import json


def find_unique_file(directory):
    """Return the single file in a directory; raise if the count differs from one."""
    files = [f for f in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, f))]
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one file in {directory}, found {len(files)}")
    return os.path.join(directory, files[0])


def get_size_from_most(path):
    """Extract grid dimensions (x, y) from a MOST bathymetry file.

    Reads until it finds a line with at least two integer tokens and returns them."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    x = int(parts[0])
                    y = int(parts[1])
                    return x, y
                except ValueError:
                    continue
    raise RuntimeError(f"Failed to detect grid size in MOST file {path}")


def create_config(output_dir="data"):
    """Create config.json by scanning for a MOST bathymetry file in data/bath.

    Computes the grid size from the file and stores the bathymetry path in the output directory."""
    bath_dir = os.path.join(output_dir, "bath")
    os.makedirs(output_dir, exist_ok=True)

    bath_path = find_unique_file(bath_dir)
    x, y = get_size_from_most(bath_path)

    config = {
        "size": {"x": x, "y": y},
        "bath_path": bath_path
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    create_config()
