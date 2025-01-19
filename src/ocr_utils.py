import os
from dotenv import load_dotenv
import logging
from termcolor import colored
from rapidocr_onnxruntime import RapidOCR
import time
import sys
from typing import List

"""
if encountered:`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`
apt install libgl1-mesa-glx
"""
load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from src.ocr_model import create_textline_from_data, NewTextLine


def polygon_to_markdown(text_lines: List[NewTextLine]) -> str:
    """
    将文本行转换为Markdown表格。

    Args:
        text_lines (List[NewTextLine]): 包含文本行及其边界框信息的列表。

    Returns:
        str: 生成的Markdown表格字符串。
    """
    # 计算所有文本行的平均高度
    total_height = 0
    for text_line in text_lines:
        height = text_line.bbox[3] - text_line.bbox[1]
        total_height += height
    average_height = total_height / len(text_lines) if text_lines else 0

    # 动态设置 y_tolerance 为平均高度的一半
    y_tolerance = average_height / 2

    # 首先根据Y坐标和高度将文本行分组 考虑容差值
    lines = []
    for text_line in text_lines:
        y_coord = text_line.bbox[1]
        height = text_line.bbox[3] - text_line.bbox[1]

        # 查找是否存在在容差范围内的Y坐标和高度
        found = False
        for line in lines:
            # 检查Y坐标和高度是否在容差范围内
            if (
                abs(y_coord - line["y"]) <= y_tolerance
                and abs(height - line["height"]) <= y_tolerance
            ):
                line["text_lines"].append(text_line)
                found = True
                break

        # 如果没有找到匹配的行 则创建一个新的行
        if not found:
            lines.append({"y": y_coord, "height": height, "text_lines": [text_line]})

    # 对每一行的文本按照X坐标排序
    sorted_lines = []
    for line in sorted(lines, key=lambda x: x["y"]):
        text_lines = line["text_lines"]
        # 按照bbox的X坐标（即左上角的X坐标）排序
        sorted_line = sorted(text_lines, key=lambda x: x.bbox[0])
        sorted_lines.append(sorted_line)

    # 生成Markdown表格
    markdown_table = []
    headers = []
    alignments = []

    # 生成表头
    for i in range(len(sorted_lines[0])):
        headers.append("Column {}".format(i + 1))
        alignments.append(":---:")  # 默认居中对齐

    markdown_table.append("| " + " | ".join(headers) + " |")
    markdown_table.append("| " + " | ".join(alignments) + " |")

    # 生成表格内容
    for line in sorted_lines:
        row = []
        prev_x_end = -1  # 记录上一个文本的结束X坐标
        for text_line in line:
            # 如果当前文本的起始X坐标与上一个文本的结束X坐标有重叠 则合并到同一单元格
            if text_line.bbox[0] < prev_x_end:
                if row:
                    row[-1] += "<br>" + text_line.text  # 多行文本用 <br> 连接
            else:
                row.append(text_line.text)
            prev_x_end = text_line.bbox[2]  # 更新上一个文本的结束X坐标
        # 将同一行的文本用 " | " 连接起来
        markdown_table.append("| " + " | ".join(row) + " |")

    # 用换行符连接每一行
    markdown_output = "\n".join(markdown_table)

    return markdown_output


def perform_ocr(ocr_engine, pic_path_or_base64):
    """
    使用指定的OCR引擎对图片进行OCR处理 并返回处理结果。

    :param ocr_engine: OCR引擎实例 需支持直接调用并返回OCR结果。
    :param pic_path_or_base64: 图片路径或base64编码的字符串。
    :return: OCR处理后的Markdown格式文本。
    """
    # 记录OCR处理时间
    start_time = time.time()
    rapid_ocr_result, _ = ocr_engine(pic_path_or_base64)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(colored(f"OCR耗时: {elapsed_time:.2f}秒", "green"))
    logger.debug(colored(f"文本框数量: {len(rapid_ocr_result)}", "green"))

    # 处理OCR结果
    text_lines = []
    if rapid_ocr_result is None:
        rapid_ocr_result = []
    for line in rapid_ocr_result:
        text_line = create_textline_from_data(line)
        text_lines.append(text_line)

    # 将文本框转换为Markdown格式
    markdown_start = polygon_to_markdown(text_lines)

    return markdown_start


if __name__ == "__main__":
    # python src/ocr_utils.py
    test_pics = [
        "no_git_oic/photo.png",
        "no_git_oic/excel.png",
        "no_git_oic/fig2.jpg",
        "no_git_oic/Snipaste_2025-01-19_12-26-16.png",
    ]
    test_id = 0
    start_time = time.time()
    ocr_engine = RapidOCR()
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(colored(f"初始化模型耗时: {elapsed_time:.2f}秒", "green"))

    markdown_start = perform_ocr(ocr_engine, test_pics[test_id])
    logger.info(colored(f"OCR-result:\n{markdown_start}", "green"))
