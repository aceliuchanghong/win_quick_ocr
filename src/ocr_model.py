from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class NewTextLine:
    """
    Represents a line of text detected by OCR.

    Attributes:
        polygon (List[Tuple[float, float]]): The polygon coordinates of the text line.
        confidence (float): The confidence score of the OCR result.
        text (str): The recognized text.
        bbox (List[float]): The bounding box of the text line in the format [x_min, y_min, x_max, y_max].
    """

    polygon: List[Tuple[float, float]]
    confidence: float
    text: str
    bbox: List[float]


def create_textline_from_data(
    data: Tuple[List[Tuple[float, float]], str, float]
) -> NewTextLine:
    """
    Creates a NewTextLine object from OCR data.

    Args:
        data (Tuple[List[Tuple[float, float]], str, float]): A tuple containing the polygon coordinates,
            recognized text, and confidence score.

    Returns:
        NewTextLine: An instance of NewTextLine representing the text line.

    Raises:
        ValueError: If the input data is invalid or missing required fields.
    """
    if not data or len(data) < 3:
        raise ValueError("Invalid input data. Expected polygon, text, and confidence.")

    polygon, text, confidence = data

    if not polygon or not isinstance(polygon, list):
        raise ValueError("Polygon must be a non-empty list of coordinates.")

    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string.")

    if not isinstance(confidence, (float, int)) or confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be a float between 0 and 1.")

    # Calculate bounding box from polygon
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]

    bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    return NewTextLine(polygon=polygon, confidence=confidence, text=text, bbox=bbox)
