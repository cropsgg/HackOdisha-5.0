#!/usr/bin/env python3
"""
Heuristic-based Obstacle Detection System
Fallback system for low-power scenarios using traditional computer vision
"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HeuristicObstacle:
    """Heuristic obstacle detection result"""
    x: float
    y: float
    obstacle_type: str
    confidence: float
    size: float
    bbox: List[float]

class HeuristicBoulderDetector:
    """Heuristic boulder detection using traditional computer vision"""
    
    def __init__(self):
        self.min_radius = 8
        self.max_radius = 60
        self.min_distance = 40
        self.quality_threshold = 0.3
    
    def detect_boulders(self, image: np.ndarray, start_x: int, start_y: int, end_x: int, end_y: int) -> List[HeuristicObstacle]:
        """Detect boulders using heuristic methods"""
        # Define region of interest
        padding = 100  # Smaller padding for heuristic processing
        min_x = max(0, int(min(start_x, end_x) - padding))
        max_x = min(image.shape[1], int(max(start_x, end_x) + padding))
        min_y = max(0, int(min(start_y, end_y) - padding))
        max_y = min(image.shape[0], int(max(start_y, end_y) + padding))
        
        roi = image[min_y:max_y, min_x:max_x]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding for better edge detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boulders = []
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < 50:  # Filter small contours
                continue
            
            # Calculate circularity (boulders should be roughly circular)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                continue
            
            # Filter based on circularity (boulders should be somewhat circular)
            if circularity < 0.3:
                continue
            
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Filter by radius
            if radius < self.min_radius or radius > self.max_radius:
                continue
            
            # Calculate confidence based on circularity and area
            confidence = min(0.9, circularity * 0.7 + (area / 1000) * 0.3)
            
            if confidence > self.quality_threshold:
                # Adjust coordinates back to full image
                full_x = x + min_x
                full_y = y + min_y
                
                bbox = [full_x - radius, full_y - radius, full_x + radius, full_y + radius]
                
                boulder = HeuristicObstacle(
                    x=full_x,
                    y=full_y,
                    obstacle_type='boulder',
                    confidence=confidence,
                    size=area,
                    bbox=bbox
                )
                boulders.append(boulder)
        
        # Remove overlapping detections
        boulders = self._remove_overlapping_boulders(boulders)
        
        # Limit number of boulders for efficiency
        if len(boulders) > 30:
            boulders.sort(key=lambda x: x.confidence, reverse=True)
            boulders = boulders[:30]
        
        logger.info(f"Heuristic boulder detection: Found {len(boulders)} boulders")
        return boulders
    
    def _remove_overlapping_boulders(self, boulders: List[HeuristicObstacle]) -> List[HeuristicObstacle]:
        """Remove overlapping boulder detections"""
        if len(boulders) <= 1:
            return boulders
        
        # Sort by confidence (keep higher confidence ones)
        boulders.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered_boulders = []
        for boulder in boulders:
            is_overlapping = False
            for existing in filtered_boulders:
                distance = np.sqrt((boulder.x - existing.x)**2 + (boulder.y - existing.y)**2)
                if distance < self.min_distance:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_boulders.append(boulder)
        
        return filtered_boulders

class HeuristicLandslideDetector:
    """Heuristic landslide detection using traditional computer vision"""
    
    def __init__(self):
        self.slope_threshold = 30.0
        self.area_threshold = 800
        self.roughness_threshold = 0.4
    
    def detect_landslides(self, image: np.ndarray, start_x: int, start_y: int, end_x: int, end_y: int) -> List[HeuristicObstacle]:
        """Detect landslides using heuristic methods"""
        # Define region of interest
        padding = 100
        min_x = max(0, int(min(start_x, end_x) - padding))
        max_x = min(image.shape[1], int(max(start_x, end_x) + padding))
        min_y = max(0, int(min(start_y, end_y) - padding))
        max_y = min(image.shape[0], int(max(start_y, end_y) + padding))
        
        roi = image[min_y:max_y, min_x:max_x]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients for slope detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate slope magnitude
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        slope_angle = np.arctan(slope_magnitude) * 180 / np.pi
        
        # Create slope mask
        slope_mask = (slope_angle > self.slope_threshold).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        slope_mask = cv2.morphologyEx(slope_mask, cv2.MORPH_CLOSE, kernel)
        slope_mask = cv2.morphologyEx(slope_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(slope_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        landslides = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.area_threshold:
                continue
            
            # Calculate roughness (standard deviation of gradients in the region)
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Get gradient values within the contour
            gradient_values = slope_magnitude[mask > 0]
            if len(gradient_values) > 0:
                roughness = np.std(gradient_values) / np.mean(gradient_values) if np.mean(gradient_values) > 0 else 0
            else:
                continue
            
            # Filter based on roughness
            if roughness < self.roughness_threshold:
                continue
            
            # Get contour center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x = M["m10"] / M["m00"]
                center_y = M["m01"] / M["m00"]
            else:
                continue
            
            # Calculate confidence based on slope and roughness
            slope_confidence = min(1.0, (np.mean(slope_angle[mask > 0]) - self.slope_threshold) / 30.0)
            roughness_confidence = min(1.0, roughness / 1.0)
            confidence = (slope_confidence * 0.6 + roughness_confidence * 0.4)
            
            if confidence > 0.4:
                # Adjust coordinates back to full image
                full_x = center_x + min_x
                full_y = center_y + min_y
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                bbox = [x + min_x, y + min_y, x + w + min_x, y + h + min_y]
                
                landslide = HeuristicObstacle(
                    x=full_x,
                    y=full_y,
                    obstacle_type='landslide',
                    confidence=confidence,
                    size=area,
                    bbox=bbox
                )
                landslides.append(landslide)
        
        # Limit number of landslides for efficiency
        if len(landslides) > 5:
            landslides.sort(key=lambda x: x.confidence, reverse=True)
            landslides = landslides[:5]
        
        logger.info(f"Heuristic landslide detection: Found {len(landslides)} landslides")
        return landslides

class HeuristicPathfindingSystem:
    """Complete heuristic-based pathfinding system"""
    
    def __init__(self):
        self.boulder_detector = HeuristicBoulderDetector()
        self.landslide_detector = HeuristicLandslideDetector()
    
    def detect_obstacles(self, image: np.ndarray, start_x: int, start_y: int, end_x: int, end_y: int) -> List[HeuristicObstacle]:
        """Detect all obstacles using heuristic methods"""
        logger.info("🔍 Starting heuristic obstacle detection...")
        
        # Detect boulders
        boulders = self.boulder_detector.detect_boulders(image, start_x, start_y, end_x, end_y)
        
        # Detect landslides
        landslides = self.landslide_detector.detect_landslides(image, start_x, start_y, end_x, end_y)
        
        # Combine all obstacles
        all_obstacles = boulders + landslides
        
        logger.info(f"🎯 Heuristic detection complete: {len(all_obstacles)} total obstacles")
        logger.info(f"   - Boulders: {len(boulders)}")
        logger.info(f"   - Landslides: {len(landslides)}")
        
        return all_obstacles
    
    def convert_to_selenographic(self, obstacles: List[HeuristicObstacle], coordinates: pd.DataFrame, image_shape: Tuple[int, int]) -> List[Dict]:
        """Convert heuristic obstacles to Selenographic coordinates"""
        from efficient_pathfinding_system import EfficientPathfindingSystem
        
        # Use the coordinate conversion from the main system
        system = EfficientPathfindingSystem()
        converted_obstacles = []
        
        for obstacle in obstacles:
            lon, lat = system._image_to_selenographic(obstacle.x, obstacle.y, coordinates, image_shape)
            
            if lon is not None and lat is not None:
                converted_obstacles.append({
                    'lon': lon,
                    'lat': lat,
                    'x': obstacle.x,
                    'y': obstacle.y,
                    'obstacle_type': obstacle.obstacle_type,
                    'confidence': obstacle.confidence,
                    'size': obstacle.size,
                    'bbox': obstacle.bbox
                })
        
        logger.info(f"📍 Converted {len(converted_obstacles)} obstacles to Selenographic coordinates")
        return converted_obstacles
