import os
import json

def find_unique_file(directory):
    """
    Ищет единственный файл в указанной директории.
    Если файлов нет или их больше одного — бросит исключение.
    """
    files = [f for f in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, f))]
    if len(files) != 1:
        raise RuntimeError(f"Ожидался ровно один файл в {directory}, найдено: {len(files)}")
    return os.path.join(directory, files[0])

def get_size_from_most(path):
    """
    Читает файл MOST и извлекает из первой непустой строки два целых числа (x, y).
    При необходимости скорректировать парсинг под конкретный формат.
    """
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
    raise RuntimeError(f"Не удалось извлечь размеры из файла {path}")

def create_config(output_dir="data"):
    """
    Находит файл MOST в data/bath, парсит из него x, y и
    создает config.json в папке data.
    """
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

