"""Contour/HSV ROI selection adapted from datasetRoi.ipynb (second definition)."""
from dataclasses import dataclass
import cv2
import numpy as np


class NoRegionError(ValueError):
    """The input has no usable external contour."""


@dataclass
class ROIConfig:
    crop_size: int = 448
    threshold: int = 64
    max_candidates: int = 4
    dominant_area_ratio: float = 20.0
    area_weight: float = 0.7
    brown_weight: float = 0.2
    yellow_weight: float = 0.8
    brown_lower: tuple = (10, 100, 20)
    brown_upper: tuple = (20, 255, 200)
    yellow_lower: tuple = (20, 100, 100)
    yellow_upper: tuple = (30, 255, 255)

    def __post_init__(self):
        if self.crop_size < 1 or self.max_candidates < 1 or not 0 <= self.threshold <= 255 or self.dominant_area_ratio <= 0:
            raise ValueError('Invalid ROI configuration')


def locate_roi(image, config=None):
    """Return clipped crop coordinates and selection details for BGR uint8 input.

    Coordinates are [x0, y0, x1, y1], with exclusive right/bottom bounds.
    Crops meeting an image edge can be smaller than crop_size; no padding/resize.
    """
    config = config or ROIConfig()
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected a BGR uint8 image with three channels')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    # Preserve THRESH_TRUNC, not a binary threshold.
    _, thresholded = cv2.threshold(equalized, config.threshold, 255, cv2.THRESH_TRUNC)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise NoRegionError('No external contour')
    areas = [cv2.contourArea(contour) for contour in contours]
    ordered = sorted(range(len(contours)), key=lambda i: areas[i], reverse=True)
    candidates = ordered[:config.max_candidates]
    selected = ordered[0]
    scores = None
    if len(ordered) > 1 and areas[ordered[0]] / (areas[ordered[1]] or 1.0) < config.dominant_area_ratio:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brown = cv2.inRange(hsv, np.array(config.brown_lower), np.array(config.brown_upper))
        yellow = cv2.inRange(hsv, np.array(config.yellow_lower), np.array(config.yellow_upper))
        features = np.zeros((len(candidates), 3), dtype=float)
        for row, index in enumerate(candidates):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, contours, index, 255, -1)
            # Preserve the source's countNonZero on the masked BGR red channel.
            features[row] = [areas[index],
                np.count_nonzero((mask != 0) & (brown != 0) & (image[:, :, 2] != 0)),
                np.count_nonzero((mask != 0) & (yellow != 0) & (image[:, :, 2] != 0))]
        span = np.ptp(features, axis=0)
        normalized = np.divide(features-features.min(axis=0), span,
                               out=np.zeros_like(features), where=span!=0)
        scores = normalized @ np.array([config.area_weight, config.brown_weight, -config.yellow_weight])
        selected = candidates[int(np.argmax(scores))]
    x, y, width, height = cv2.boundingRect(contours[selected])
    left = x - (config.crop_size-width)//2
    top = y - (config.crop_size-height)//2
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(image.shape[1], left+config.crop_size), min(image.shape[0], top+config.crop_size)
    if x1 <= x0 or y1 <= y0:
        raise NoRegionError('ROI does not intersect the image')
    return {'crop_box': [x0, y0, x1, y1], 'contour_box': [x, y, width, height],
            'contour_index': int(selected), 'contour_count': len(contours),
            'candidate_indices': candidates, 'scores': scores.tolist() if scores is not None else None}


def extract_roi(image, config=None):
    info = locate_roi(image, config)
    x0, y0, x1, y1 = info['crop_box']
    return image[y0:y1, x0:x1].copy(), info
