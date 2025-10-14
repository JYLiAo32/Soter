import colorama
from colorama import Fore, Style, Back

red_prefix = Fore.RED
gre_prefix = Fore.GREEN
light_prefix = Fore.LIGHTWHITE_EX
default_color = '\033[0m'


def color_text(color, text):
    if color == 'red':
        return red_prefix+text+default_color
    elif color == 'gre':
        return gre_prefix+text+default_color
    assert ('Not support color: {}.'.format(color))


# 初始化 colorama（支持跨平台终端颜色）
colorama.init(autoreset=True)  # autoreset=True 确保每次打印后自动重置颜色

# 颜色映射表：键为用户友好的颜色名，值为 colorama 的颜色常量
COLOR_MAP = {
    "black": Fore.BLACK,
    "red": Fore.RED,
    "green": Fore.GREEN,
    "yellow": Fore.YELLOW,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "cyan": Fore.CYAN,
    "white": Fore.WHITE,
    "reset": Fore.RESET,
    ###########
    "warning": Fore.YELLOW,
    "note": Fore.BLUE,
    "info": Fore.BLUE,
    "error": Fore.RED,
}

def colored_print(text: str, color: str = None, bold: bool = False) -> None:
    # 处理颜色
    color_code = COLOR_MAP.get(color.lower(), "") if color else ""

    # 处理加粗样式
    style_code = Style.BRIGHT if bold else ""

    # 拼接颜色、样式和文本，最后重置样式（双重保险，即使 autoreset 失效）
    formatted_text = f"{style_code}{color_code}{text}{Style.RESET_ALL}"

    # 打印最终文本
    print(formatted_text)

if __name__ == '__main__':
    colored_print("Hello, World!", color="green", bold=True)
    colored_print("Warning: This is a warning message.", color="warning")
    colored_print("Error: This is an error message.", color="error")
    colored_print("Note: This is a note message.", color="note")