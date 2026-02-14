"""
fileio.py文件提供了文件的输入输出功能:
1.一般文件读取/编写操作
2.JSON文件读取/编写操作
3.递归删除文件操作 
"""

# 导入区
import json
import os


# 1.读取文件文本
def read_file_text(filepath: str) -> str:
    with open(os.path.normpath(filepath), "r", encoding="utf-8") as f:
        return f.read()

# 2.读取文件文本行
def read_file_lines(filepath: str) -> list[str]:
    with open(os.path.normpath(filepath), "r", encoding="utf-8") as f:
        return f.read().split("\n")

# 3.编写文件文本
def write_file_text(filepath: str, content: str, append: bool = False) -> None:
    write_mode = "a" if append else "w"
    with open(os.path.normpath(filepath), write_mode, encoding="utf-8") as f:
        f.write(content)

# 4.编写文件文本行
def write_file_lines(filepath: str, content: list[str], append: bool = False) -> None:
    write_mode = "a" if append else "w"
    with open(os.path.normpath(filepath), write_mode, encoding="utf-8") as f:
        for line in content:
            f.write(line)
            f.write("\n")

# 5.读取JSON文件, 若无则进行创建
def read_json(filepath: str) -> dict:
    if not os.path.exists(filepath):
        write_json(filepath, {})
    with open(os.path.normpath(filepath), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 6.编写JSON文件
def write_json(filepath: str, data: dict) -> None:
    with open(os.path.normpath(filepath), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# 7.递归移除文件
def remove_files(filepath: str) -> None:
    filepath = os.path.normpath(filepath)
    for filename in os.listdir(filepath):
        subfilepath = os.path.join(filepath, filename)
        if os.path.isdir(subfilepath):
            remove_files(subfilepath)
        if os.path.isfile(subfilepath):
            os.remove(subfilepath)
    os.rmdir(filepath)
