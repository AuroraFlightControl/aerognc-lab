import pytest

from src.aerognc.core.StopConditions.Greofence import Circle, Rectangle

def test_circle_contains():
    circle = Circle(center_x=0.0, center_y=0.0, radius=5.0)

    assert circle.contains(0.0, 0.0) is True
    assert circle.contains(0.0, 5.0) is True
    assert circle.contains(5.0, 5.0) is False
    assert circle.contains(5.0, 0.0) is True
    assert circle.contains(-5.0, 0.0) is True
    assert circle.contains(0.0, -5.0) is True


def test_rectangle_contains():
    rectangle = Rectangle(min_x = 0.0, max_x=5.0, min_y=0.0, max_y=5.0)

    assert rectangle.contains(0.0, 0.0) is True
    assert rectangle.contains(2.5, 2.5) is True
    assert rectangle.contains(5.0, 5.0) is True
    assert rectangle.contains(6.0, 0.0) is False
    assert rectangle.contains(0.0, 6.0) is False
    assert rectangle.contains(-2.5, -2.5) is False   