from PIL import Image
from PIL.ImageChops import screen

from new_wzry_ai.config.default_config import TemplateRegion

class PositionAnalyzer:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, image_path = "/template/minimap_area/minimap_filled.png"):
        if self._initialized:
            return

        x1, y1, _ , _ = TemplateRegion.CHARACTER_AREA
        self.origin_offset = (x1, y1)
        self.availability_grid = []

        img = Image.open(image_path).convert("RGB")
        self.width, self.height = img.size

        # 创建并填充网格
        self.availability_grid = [[1 for _ in range(self.width)] for _ in range(self.height)]
        for x in range(self.width):
            for y in range(self.height):
                r, g, b = img.getpixel((x, y))
                if r > 200 and g < 50 and b < 50:
                    self.availability_grid[y][x] = 0
        self._initialized = True

    def is_available(self, coord):
        adjusted_x = coord[0] - self.origin_offset[0]
        adjusted_y = coord[1] - self.origin_offset[1]

        if adjusted_y < 0 or adjusted_y >= self.height:
            return True
        if adjusted_x < 0 or adjusted_x >= self.width:
            return True

        return self.availability_grid[adjusted_y][adjusted_x] != 0

    def save_grid_to_file(self, filename, use_symbols=False):
        """
        将网格保存到文本文件
        :param filename: 输出文件名
        :param use_symbols: 是否使用符号表示（False则输出0/1数字）
        """
        # 确定输出范围
        output_height = self.height
        output_width = self.width


        # 准备输出内容
        with open(filename, 'w', encoding='utf-8') as f:
            # 写入元信息
            f.write(f"Original Size: {self.width}x{self.height}\n")
            f.write(f"Output Size: {output_width}x{output_height}\n")
            f.write("Format: " + ("Symbols" if use_symbols else "Digital") + "\n\n")
            # 生成每行内容
            for y in range(output_height):
                line = []
                for x in range(output_width):
                    cell = self.availability_grid[y][x]
                    if use_symbols:
                        line.append("█" if cell == 0 else " ")
                    else:
                        line.append(str(cell))

                # 添加行号并写入
                f.write(f"{y:04d}: {''.join(line)}\n")


# 使用示例
if __name__ == "__main__":
    analyzer = PositionAnalyzer("../template/minimap_area/minimap_filled.png")
    # 保存完整大文件（慎用）
    analyzer.save_grid_to_file("full_grid.txt")