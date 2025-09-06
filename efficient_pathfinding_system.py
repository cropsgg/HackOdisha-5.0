#!/usr/bin/env python3
"""
Efficient Lunar Pathfinding System
Takes a single image and start/end coordinates, finds optimal path efficiently
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass
import folium
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ObstaclePoint:
    """Represents an obstacle point with coordinates and risk level"""
    lon: float
    lat: float
    x: float
    y: float
    obstacle_type: str  # 'boulder' or 'landslide'
    confidence: float
    risk_level: float  # 0-1 scale
    size: float  # area or diameter

@dataclass
class PathPoint:
    """Represents a point in the optimal path"""
    lon: float
    lat: float
    x: float
    y: float
    elevation: float
    slope: float
    risk_score: float
    is_safe: bool
    distance_from_start: float

def format_selenographic_coords(lon: float, lat: float) -> str:
    """Format Selenographic coordinates for display"""
    lon_dir = "E" if lon >= 0 else "W"
    lat_dir = "N" if lat >= 0 else "S"
    return f"{abs(lat):.6f}°{lat_dir}, {abs(lon):.6f}°{lon_dir}"

class EfficientPathfindingSystem:
    """Efficient pathfinding system for single image processing"""
    
    def __init__(self):
        """Initialize the efficient pathfinding system"""
        self.max_risk_score = 0.3  # 0-1 scale
        self.safety_margin = 50.0  # meters from obstacles
        
        logger.info("Efficient Pathfinding System initialized")
    
    def load_image_and_coordinates(self, image_path: str, csv_path: str) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Load a single image and its coordinate data"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                logger.info(f"Loaded image: {image.shape}")
            else:
                logger.error(f"Could not load image: {image_path}")
                return None, None
            
            # Load coordinates
            coordinates = pd.read_csv(csv_path)
            logger.info(f"Loaded coordinates: {len(coordinates)} points")
            
            return image, coordinates
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None, None
    
    def detect_boulders_in_region(self, image: np.ndarray, start_x: int, start_y: int, end_x: int, end_y: int) -> List[Dict]:
        """Detect boulders in the region of interest with full accuracy"""
        # Define region of interest with padding
        padding = 200  # Increased padding for better detection
        min_x = max(0, min(start_x, end_x) - padding)
        max_x = min(image.shape[1], max(start_x, end_x) + padding)
        min_y = max(0, min(start_y, end_y) - padding)
        max_y = min(image.shape[0], max(start_y, end_y) + padding)
        
        # Extract region of interest
        roi = image[min_y:max_y, min_x:max_x]
        logger.info(f"Processing ROI: {roi.shape} (from full image {image.shape})")
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect circles with balanced parameters for quality detection
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,  # Increased minDist to avoid overlapping
            param1=50, param2=50, minRadius=10, maxRadius=80  # Stricter parameters for quality
        )
        
        boulders = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Process detected circles with quality filtering
            for (x, y, r) in circles:
                # Adjust coordinates back to full image
                full_x = x + min_x
                full_y = y + min_y
                
                # Calculate bounding box
                bbox = [full_x - r, full_y - r, full_x + r, full_y + r]
                area = np.pi * r * r
                
                # Enhanced confidence calculation with quality checks
                confidence = min(0.95, 0.3 + (r / 100.0) + 0.2)  # Better confidence scoring
                
                # Quality filtering - only keep high-quality boulder detections
                if confidence >= 0.7 and r >= 20:  # Even higher threshold for quality
                    boulders.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'area': area,
                        'radius': r
                    })
        
        # Limit the number of boulders to prevent overwhelming the pathfinding
        if len(boulders) > 50:
            # Sort by confidence and keep only the top 50
            boulders.sort(key=lambda x: x['confidence'], reverse=True)
            boulders = boulders[:50]
        
        logger.info(f"Detected {len(boulders)} boulders in ROI (full accuracy)")
        return boulders
    
    def detect_landslides_in_region(self, image: np.ndarray, start_x: int, start_y: int, end_x: int, end_y: int) -> Dict[str, np.ndarray]:
        """Detect landslides in the region of interest with full accuracy"""
        # Define region of interest with padding
        padding = 200  # Increased padding for better detection
        min_x = max(0, min(start_x, end_x) - padding)
        max_x = min(image.shape[1], max(start_x, end_x) + padding)
        min_y = max(0, min(start_y, end_y) - padding)
        max_y = min(image.shape[0], max(start_y, end_y) + padding)
        
        # Extract region of interest
        roi = image[min_y:max_y, min_x:max_x]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Create synthetic terrain data
        dtm = gray.astype(np.float32)
        
        # Calculate slope from image gradients
        grad_x = cv2.Sobel(dtm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(dtm, cv2.CV_32F, 0, 1, ksize=3)
        slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * 180 / np.pi
        
        # Create landslide mask with balanced thresholds for quality detection
        landslide_mask = (slope > 35.0).astype(np.float32)  # Higher threshold for quality
        landslide_score = np.clip(slope / 90.0, 0, 1)
        
        # Apply moderate smoothing to preserve details
        if landslide_mask.shape[0] > 10 and landslide_mask.shape[1] > 10:
            landslide_mask = cv2.GaussianBlur(landslide_mask, (5, 5), 0)
            landslide_mask = (landslide_mask > 0.5).astype(np.float32)  # Higher threshold for quality
        else:
            # For small regions, use smaller kernel
            landslide_mask = cv2.GaussianBlur(landslide_mask, (3, 3), 0)
            landslide_mask = (landslide_mask > 0.5).astype(np.float32)
        
        # Calculate statistics
        high_risk_pixels = np.sum(landslide_mask > 0.5)
        total_pixels = landslide_mask.size
        landslide_percentage = (high_risk_pixels / total_pixels) * 100
        
        results = {
            'landslide_mask': landslide_mask,
            'landslide_score': landslide_score,
            'statistics': {
                'landslide_percentage': landslide_percentage,
                'high_risk_pixels': high_risk_pixels,
                'total_pixels': total_pixels
            }
        }
        
        logger.info(f"Landslide detection completed: {landslide_percentage:.2f}% high risk in ROI (full accuracy)")
        return results
    
    def convert_obstacles_to_coordinates(self, boulders: List[Dict], landslide_results: Dict, 
                                       coordinates: pd.DataFrame, image_shape: Tuple[int, int]) -> List[ObstaclePoint]:
        """Convert detected obstacles to Selenographic coordinates"""
        obstacles = []
        
        # Convert boulders to coordinate points
        for i, boulder in enumerate(boulders):
            bbox = boulder['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Convert image coordinates to Selenographic coordinates
            lon, lat = self._image_to_selenographic(center_x, center_y, coordinates, image_shape)
            
            if lon is not None and lat is not None:
                obstacle = ObstaclePoint(
                    lon=lon, lat=lat, x=center_x, y=center_y,
                    obstacle_type='boulder',
                    confidence=boulder['confidence'],
                    risk_level=min(boulder['confidence'] * 1.2, 1.0),
                    size=np.sqrt(boulder['area'])
                )
                obstacles.append(obstacle)
        
        # Convert landslide areas to coordinate points (full accuracy)
        if 'landslide_mask' in landslide_results:
            landslide_mask = landslide_results['landslide_mask']
            landslide_score = landslide_results.get('landslide_score', np.zeros_like(landslide_mask))
            
            # Find high-risk landslide areas with quality filtering
            high_risk_mask = (landslide_mask > 0) & (landslide_score > 0.7)  # Higher threshold for quality
            contours, _ = cv2.findContours(high_risk_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process significant landslide areas with quality filtering
            landslide_areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Much higher area threshold for quality
                    # Get contour center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = M["m10"] / M["m00"]
                        center_y = M["m01"] / M["m00"]
                    
                    # Convert to Selenographic coordinates
                    lon, lat = self._image_to_selenographic(center_x, center_y, coordinates, image_shape)
                    
                    if lon is not None and lat is not None:
                        # Calculate average risk score for this area
                        mask = np.zeros_like(landslide_mask, dtype=np.uint8)
                        cv2.fillPoly(mask, [contour], 1)
                        avg_risk = np.mean(landslide_score[mask > 0])
                        
                        landslide_areas.append({
                            'lon': lon, 'lat': lat, 'x': center_x, 'y': center_y,
                            'confidence': avg_risk, 'risk_level': avg_risk, 'size': area
                        })
            
            # Limit landslide areas to prevent overwhelming pathfinding
            if len(landslide_areas) > 10:
                # Sort by risk level and keep only the top 10
                landslide_areas.sort(key=lambda x: x['risk_level'], reverse=True)
                landslide_areas = landslide_areas[:10]
            
            # Add landslide obstacles
            for area in landslide_areas:
                obstacle = ObstaclePoint(
                    lon=area['lon'], lat=area['lat'], x=area['x'], y=area['y'],
                    obstacle_type='landslide',
                    confidence=area['confidence'],
                    risk_level=area['risk_level'],
                    size=area['size']
                )
                obstacles.append(obstacle)
        
        logger.info(f"Converted {len(obstacles)} obstacles to Selenographic coordinates")
        return obstacles
    
    def _image_to_selenographic(self, x: float, y: float, coordinates: pd.DataFrame, 
                               image_shape: Tuple[int, int]) -> Tuple[Optional[float], Optional[float]]:
        """Convert image coordinates to Selenographic coordinates using Pixel/Scan mapping"""
        try:
            # Ensure coordinates are within image bounds
            x = max(0, min(x, image_shape[1] - 1))
            y = max(0, min(y, image_shape[0] - 1))
            
            # Scale from browse image coordinates to original image coordinates
            # Original image dimensions from CSV data
            original_width = coordinates['Pixel'].max() + 1  # 12000
            original_height = coordinates['Scan'].max() + 1  # 90148
            
            # Scale to original image dimensions
            scale_x = original_width / image_shape[1]  # 12000 / 1200 = 10
            scale_y = original_height / image_shape[0]  # 90148 / 9015 ≈ 10
            
            original_pixel = x * scale_x
            original_scan = y * scale_y
            
            # Find the closest coordinate entry in original coordinates
            distances = ((coordinates['Pixel'] - original_pixel) ** 2 + (coordinates['Scan'] - original_scan) ** 2) ** 0.5
            closest_idx = distances.idxmin()
            closest_coord = coordinates.loc[closest_idx]
            
            return closest_coord['Longitude'], closest_coord['Latitude']
            
        except Exception as e:
            logger.error(f"Error converting coordinates: {e}")
            return None, None
    
    def _selenographic_to_image(self, lon: float, lat: float, coordinates: pd.DataFrame, 
                               image_shape: Tuple[int, int]) -> Tuple[Optional[float], Optional[float]]:
        """Convert Selenographic coordinates to image coordinates using Pixel/Scan mapping"""
        try:
            # Find the closest coordinate entry by longitude and latitude
            distances = ((coordinates['Longitude'] - lon) ** 2 + (coordinates['Latitude'] - lat) ** 2) ** 0.5
            closest_idx = distances.idxmin()
            closest_coord = coordinates.loc[closest_idx]
            
            # Get the original Pixel and Scan values
            original_pixel = float(closest_coord['Pixel'])
            original_scan = float(closest_coord['Scan'])
            
            # Scale from original image coordinates to browse image coordinates
            # Original image dimensions from CSV data
            original_width = coordinates['Pixel'].max() + 1  # 12000
            original_height = coordinates['Scan'].max() + 1  # 90148
            
            # Scale to browse image dimensions
            scale_x = image_shape[1] / original_width  # 1200 / 12000 = 0.1
            scale_y = image_shape[0] / original_height  # 9015 / 90148 ≈ 0.1
            
            x = original_pixel * scale_x
            y = original_scan * scale_y
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, image_shape[1] - 1))
            y = max(0, min(y, image_shape[0] - 1))
            
            return x, y
            
        except Exception as e:
            logger.error(f"Error converting coordinates: {e}")
            return None, None
    
    def find_optimal_path(self, start_lon: float, start_lat: float, end_lon: float, end_lat: float,
                         obstacles: List[ObstaclePoint], coordinates: pd.DataFrame, image_shape: Tuple[int, int]) -> List[PathPoint]:
        """Find optimal path using A* algorithm with obstacle avoidance"""
        
        # Convert start and end coordinates to image coordinates
        start_x, start_y = self._selenographic_to_image(start_lon, start_lat, coordinates, image_shape)
        end_x, end_y = self._selenographic_to_image(end_lon, end_lat, coordinates, image_shape)
        
        if start_x is None or start_y is None or end_x is None or end_y is None:
            logger.error("Could not convert start/end coordinates to image coordinates")
            return []
        
        # Create a small, focused grid around the path
        # Define path region with padding
        padding = 50
        min_x = max(0, int(min(start_x, end_x) - padding))
        max_x = min(image_shape[1], int(max(start_x, end_x) + padding))
        min_y = max(0, int(min(start_y, end_y) - padding))
        max_y = min(image_shape[0], int(max(start_y, end_y) + padding))
        
        # Create grid for accurate pathfinding
        grid_size = 30  # Further reduced resolution for better pathfinding success
        obstacle_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Map obstacles to grid
        for obstacle in obstacles:
            # Convert obstacle coordinates to image coordinates
            obs_x, obs_y = self._selenographic_to_image(obstacle.lon, obstacle.lat, coordinates, image_shape)
            if obs_x is None or obs_y is None:
                continue
            
            # Check if obstacle is in our region of interest
            if min_x <= obs_x <= max_x and min_y <= obs_y <= max_y:
                # Convert to grid coordinates
                grid_x = int((obs_x - min_x) / (max_x - min_x) * grid_size)
                grid_y = int((obs_y - min_y) / (max_y - min_y) * grid_size)
                
                # Ensure within bounds
                grid_x = np.clip(grid_x, 0, grid_size - 1)
                grid_y = np.clip(grid_y, 0, grid_size - 1)
                
                # Set obstacle value
                if obstacle.obstacle_type == 'boulder':
                    obstacle_value = obstacle.risk_level * 2.0
                else:  # landslide
                    obstacle_value = obstacle.risk_level * 1.5
                
                # Apply to grid with safety margin
                radius = 1  # Minimal radius for better pathfinding success
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        new_x, new_y = grid_x + dx, grid_y + dy
                        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
                            distance = np.sqrt(dx*dx + dy*dy)
                            if distance <= radius:
                                weight = np.exp(-distance**2 / (2 * (radius/2)**2))
                                obstacle_map[new_y, new_x] = max(obstacle_map[new_y, new_x], obstacle_value * weight)
        
        # Create graph for pathfinding
        pathfinding_graph = nx.Graph()
        
        # Add nodes to graph
        for y in range(grid_size):
            for x in range(grid_size):
                # Convert grid coordinates to Selenographic coordinates
                img_x = min_x + (x / grid_size) * (max_x - min_x)
                img_y = min_y + (y / grid_size) * (max_y - min_y)
                
                # Convert to Selenographic coordinates
                lon, lat = self._image_to_selenographic(img_x, img_y, coordinates, image_shape)
                
                # Check if location is safe
                risk_score = obstacle_map[y, x]
                is_safe = risk_score < self.max_risk_score
                
                if is_safe:
                    pathfinding_graph.add_node((x, y), lon=lon, lat=lat, risk=risk_score)
        
        # Add edges between nearby nodes
        for node in pathfinding_graph.nodes():
            x, y = node
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (x + dx, y + dy)
                    if neighbor in pathfinding_graph.nodes():
                        # Calculate cost based on distance and risk
                        start_risk = pathfinding_graph.nodes[node]['risk']
                        end_risk = pathfinding_graph.nodes[neighbor]['risk']
                        avg_risk = (start_risk + end_risk) / 2
                        
                        # Distance cost
                        distance_cost = np.sqrt(dx*dx + dy*dy)
                        
                        # Risk cost
                        risk_cost = avg_risk * 1  # Minimal risk cost for better pathfinding
                        
                        total_cost = distance_cost + risk_cost
                        
                        pathfinding_graph.add_edge(node, neighbor, weight=total_cost)
        
        # Find start and end grid coordinates
        start_grid_x = int((start_x - min_x) / (max_x - min_x) * grid_size)
        start_grid_y = int((start_y - min_y) / (max_y - min_y) * grid_size)
        end_grid_x = int((end_x - min_x) / (max_x - min_x) * grid_size)
        end_grid_y = int((end_y - min_y) / (max_y - min_y) * grid_size)
        
        # Ensure coordinates are within bounds
        start_grid_x = np.clip(start_grid_x, 0, grid_size - 1)
        start_grid_y = np.clip(start_grid_y, 0, grid_size - 1)
        end_grid_x = np.clip(end_grid_x, 0, grid_size - 1)
        end_grid_y = np.clip(end_grid_y, 0, grid_size - 1)
        
        start_node = (start_grid_x, start_grid_y)
        end_node = (end_grid_x, end_grid_y)
        
        # Add start and end nodes if not in graph
        if start_node not in pathfinding_graph.nodes():
            pathfinding_graph.add_node(start_node, lon=start_lon, lat=start_lat, risk=0)
        if end_node not in pathfinding_graph.nodes():
            pathfinding_graph.add_node(end_node, lon=end_lon, lat=end_lat, risk=0)
        
        # Find shortest path using A*
        try:
            path_nodes = nx.astar_path(
                pathfinding_graph, start_node, end_node,
                heuristic=lambda n1, n2: np.sqrt((n1[0]-n2[0])**2 + (n1[1]-n2[1])**2),
                weight='weight'
            )
            
            # Convert path nodes to PathPoint objects with smoothing
            path_points = []
            total_distance = 0.0
            
            # Smooth the path by taking every 2nd point to avoid too many close points
            smoothed_nodes = path_nodes[::2]  # Take every 2nd point
            if smoothed_nodes[-1] != path_nodes[-1]:  # Ensure end point is included
                smoothed_nodes.append(path_nodes[-1])
            
            for i, (x, y) in enumerate(smoothed_nodes):
                node_data = pathfinding_graph.nodes[(x, y)]
                lon, lat = node_data['lon'], node_data['lat']
                risk_score = node_data['risk']
                
                # Calculate distance from start
                if i > 0:
                    prev_lon, prev_lat = pathfinding_graph.nodes[smoothed_nodes[i-1]]['lon'], pathfinding_graph.nodes[smoothed_nodes[i-1]]['lat']
                    segment_distance = self._calculate_lunar_distance(prev_lon, prev_lat, lon, lat)
                    total_distance += segment_distance
                
                point = PathPoint(
                    lon=lon, lat=lat, x=x, y=y,
                    elevation=0.0,
                    slope=0.0,
                    risk_score=risk_score,
                    is_safe=risk_score < self.max_risk_score,
                    distance_from_start=total_distance
                )
                path_points.append(point)
            
            logger.info(f"Found optimal path with {len(path_points)} points, total distance: {total_distance:.2f} km")
            return path_points
            
        except nx.NetworkXNoPath:
            logger.error("No safe path found between start and end points")
            return []
        except Exception as e:
            logger.error(f"Error finding path: {e}")
            return []
    
    def _calculate_lunar_distance(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Calculate distance between two Selenographic coordinates on lunar surface"""
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        # Haversine formula for lunar surface
        moon_radius = 1737400  # meters
        a = (np.sin(delta_lat / 2) ** 2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return moon_radius * c / 1000.0  # Return in kilometers
    
    def create_folium_visualization(self, obstacles: List[ObstaclePoint], path_points: List[PathPoint],
                                  start_lon: float, start_lat: float, end_lon: float, end_lat: float,
                                  obs_id: str, image_path: str, coordinates: pd.DataFrame) -> folium.Map:
        """Create interactive folium map visualization with lunar image background"""
        
        # Calculate map center
        center_lat = (start_lat + end_lat) / 2
        center_lon = (start_lon + end_lon) / 2
        
        # Get coordinate bounds
        min_lon, max_lon = coordinates['Longitude'].min(), coordinates['Longitude'].max()
        min_lat, max_lat = coordinates['Latitude'].min(), coordinates['Latitude'].max()
        
        # Create base map with proper bounds
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles=None  # No default tiles
        )
        
        # Add the lunar image as overlay with proper bounds
        folium.raster_layers.ImageOverlay(
            image=image_path,
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],  # [southwest, northeast]
            opacity=0.8,
            interactive=True,
            cross_origin=False
        ).add_to(m)
        
        # Set map bounds to fit the image
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
        
        # Add start point
        folium.Marker(
            [start_lat, start_lon],
            popup=f"Start<br>Coordinates: {format_selenographic_coords(start_lon, start_lat)}",
            tooltip="Start Point",
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(m)
        
        # Add end point
        folium.Marker(
            [end_lat, end_lon],
            popup=f"Goal<br>Coordinates: {format_selenographic_coords(end_lon, end_lat)}",
            tooltip="Goal Point",
            icon=folium.Icon(color='red', icon='flag', prefix='fa')
        ).add_to(m)
        
        # Add boulder obstacles
        boulder_coords = []
        for obstacle in obstacles:
            if obstacle.obstacle_type == 'boulder':
                boulder_coords.append([obstacle.lat, obstacle.lon])
                folium.CircleMarker(
                    [obstacle.lat, obstacle.lon],
                    radius=max(3, int(obstacle.size / 50)),
                    popup=f"Boulder<br>Confidence: {obstacle.confidence:.2f}<br>Risk: {obstacle.risk_level:.2f}",
                    tooltip="Boulder",
                    color='blue',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.7
                ).add_to(m)
        
        # Add landslide obstacles
        landslide_coords = []
        for obstacle in obstacles:
            if obstacle.obstacle_type == 'landslide':
                landslide_coords.append([obstacle.lat, obstacle.lon])
                folium.Marker(
                    [obstacle.lat, obstacle.lon],
                    popup=f"Landslide<br>Risk Level: {obstacle.risk_level:.2f}<br>Size: {obstacle.size:.0f}",
                    tooltip="Landslide",
                    icon=folium.Icon(color='red', icon='warning', prefix='fa')
                ).add_to(m)
        
        # Add optimal path with enhanced visualization
        if path_points:
            path_coords = [[point.lat, point.lon] for point in path_points]
            
            # Ensure path connects start and end points
            full_path_coords = [[start_lat, start_lon]] + path_coords + [[end_lat, end_lon]]
            
            # Create a thick, prominent path line
            folium.PolyLine(
                full_path_coords,
                color='green',
                weight=6,  # Thicker line for better visibility
                opacity=0.9,
                popup=f"Optimal Path<br>Length: {path_points[-1].distance_from_start:.2f} km<br>Points: {len(path_points)}"
            ).add_to(m)
            
            # Add path waypoints as small markers
            for i, point in enumerate(path_points):
                if i % 2 == 0:  # Show every 2nd point to avoid clutter
                    folium.CircleMarker(
                        [point.lat, point.lon],
                        radius=3,
                        color='green',
                        fill=True,
                        fillColor='lightgreen',
                        opacity=0.8,
                        popup=f"Waypoint {i}<br>Risk: {point.risk_score:.2f}<br>Safe: {point.is_safe}"
                    ).add_to(m)
            
            # Add path statistics
            safe_segments = sum(1 for point in path_points if point.is_safe)
            safety_percentage = (safe_segments / len(path_points)) * 100
            
            # Add statistics popup
            stats_html = f"""
            <div style="font-family: Arial; padding: 10px;">
                <h3>Path Statistics</h3>
                <p><b>Total Distance:</b> {path_points[-1].distance_from_start:.2f} km</p>
                <p><b>Path Points:</b> {len(path_points)}</p>
                <p><b>Safety:</b> {safety_percentage:.1f}% safe segments</p>
                <p><b>Obstacles Avoided:</b> {len(obstacles)} total</p>
                <p><b>Boulders:</b> {len(boulder_coords)}</p>
                <p><b>Landslides:</b> {len(landslide_coords)}</p>
            </div>
            """
            folium.Marker(
                [center_lat, center_lon],
                popup=folium.Popup(stats_html, max_width=300),
                tooltip="Path Statistics",
                icon=folium.Icon(color='purple', icon='info', prefix='fa')
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Legend</b></p>
        <p><i class="fa fa-circle" style="color:blue"></i> Boulder</p>
        <p><i class="fa fa-warning" style="color:red"></i> Landslide</p>
        <p><i class="fa fa-play" style="color:green"></i> Start</p>
        <p><i class="fa fa-flag" style="color:red"></i> Goal</p>
        <p><span style="color:green; font-weight:bold">━</span> Optimal Path</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add title
        title_html = f'''
        <h3 style="position:fixed; 
                   top:10px; left:50px; width:500px; height:30px; 
                   background-color:white; z-index:9999; 
                   font-size:16px; padding:10px">
        Efficient Lunar Pathfinding - {obs_id}
        </h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def process_pathfinding(self, image_path: str, csv_path: str, start_lon: float, start_lat: float,
                          end_lon: float, end_lat: float) -> Dict[str, Any]:
        """Complete pathfinding pipeline for single image"""
        logger.info(f"Starting pathfinding from {format_selenographic_coords(start_lon, start_lat)} to {format_selenographic_coords(end_lon, end_lat)}")
        
        # Load image and coordinates
        image, coordinates = self.load_image_and_coordinates(image_path, csv_path)
        if image is None or coordinates is None:
            return {"error": "Failed to load image or coordinate data"}
        
        # Convert start and end coordinates to image coordinates
        start_x, start_y = self._selenographic_to_image(start_lon, start_lat, coordinates, image.shape[:2])
        end_x, end_y = self._selenographic_to_image(end_lon, end_lat, coordinates, image.shape[:2])
        
        if start_x is None or start_y is None or end_x is None or end_y is None:
            return {"error": "Could not convert start/end coordinates to image coordinates"}
        
        logger.info(f"Start image coordinates: ({start_x:.1f}, {start_y:.1f})")
        logger.info(f"End image coordinates: ({end_x:.1f}, {end_y:.1f})")
        
        # Detect obstacles in region of interest
        boulders = self.detect_boulders_in_region(image, int(start_x), int(start_y), int(end_x), int(end_y))
        landslide_results = self.detect_landslides_in_region(image, int(start_x), int(start_y), int(end_x), int(end_y))
        
        # Convert obstacles to coordinates
        obstacles = self.convert_obstacles_to_coordinates(boulders, landslide_results, coordinates, image.shape[:2])
        
        # Find optimal path
        path_points = self.find_optimal_path(start_lon, start_lat, end_lon, end_lat, obstacles, coordinates, image.shape[:2])
        
        # Create visualization
        obs_id = Path(image_path).stem
        folium_map = self.create_folium_visualization(obstacles, path_points, start_lon, start_lat, end_lon, end_lat, obs_id, image_path, coordinates)
        
        # Calculate statistics
        stats = {
            "total_obstacles": len(obstacles),
            "boulders": len([o for o in obstacles if o.obstacle_type == 'boulder']),
            "landslides": len([o for o in obstacles if o.obstacle_type == 'landslide']),
            "path_found": len(path_points) > 0,
            "path_length_km": path_points[-1].distance_from_start if path_points else 0,
            "path_points": len(path_points),
            "safety_percentage": (sum(1 for p in path_points if p.is_safe) / len(path_points) * 100) if path_points else 0
        }
        
        return {
            "observation_id": obs_id,
            "obstacles": obstacles,
            "path_points": path_points,
            "folium_map": folium_map,
            "statistics": stats,
            "image_shape": image.shape,
            "coordinate_bounds": {
                "min_lon": coordinates['Longitude'].min(),
                "max_lon": coordinates['Longitude'].max(),
                "min_lat": coordinates['Latitude'].min(),
                "max_lat": coordinates['Latitude'].max()
            }
        }

def main():
    """Main function to demonstrate the efficient pathfinding system"""
    print("=== Efficient Lunar Pathfinding System ===")
    
    # Initialize system
    system = EfficientPathfindingSystem()
    
    # Example usage with specific files and coordinates
    image_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_b_brw_d18.png"
    csv_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_g_grd_d18.csv"
    
    # Example coordinates (you can change these)
    start_lon, start_lat = 41.406, -19.694  # Start point
    end_lon, end_lat = 41.420, -19.690      # End point
    
    print(f"Image: {image_path}")
    print(f"CSV: {csv_path}")
    print(f"Start: {format_selenographic_coords(start_lon, start_lat)}")
    print(f"End: {format_selenographic_coords(end_lon, end_lat)}")
    
    # Process pathfinding
    results = system.process_pathfinding(image_path, csv_path, start_lon, start_lat, end_lon, end_lat)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Display results
    print("\n=== Results ===")
    print(f"Total obstacles detected: {results['statistics']['total_obstacles']}")
    print(f"  - Boulders: {results['statistics']['boulders']}")
    print(f"  - Landslides: {results['statistics']['landslides']}")
    print(f"Path found: {results['statistics']['path_found']}")
    print(f"Path length: {results['statistics']['path_length_km']:.2f} km")
    print(f"Safety: {results['statistics']['safety_percentage']:.1f}%")
    
    # Save folium map
    output_dir = Path("pathfinding_results")
    output_dir.mkdir(exist_ok=True)
    
    map_path = output_dir / f"efficient_lunar_pathfinding_{results['observation_id']}.html"
    results['folium_map'].save(str(map_path))
    print(f"Interactive map saved to: {map_path}")
    
    # Save results as JSON
    json_path = output_dir / f"efficient_results_{results['observation_id']}.json"
    
    # Convert results to JSON-serializable format
    json_results = {
        "observation_id": results['observation_id'],
        "statistics": results['statistics'],
        "coordinate_bounds": {
            "min_lon": float(results['coordinate_bounds']['min_lon']),
            "max_lon": float(results['coordinate_bounds']['max_lon']),
            "min_lat": float(results['coordinate_bounds']['min_lat']),
            "max_lat": float(results['coordinate_bounds']['max_lat'])
        },
        "obstacles": [
            {
                "lon": float(o.lon), "lat": float(o.lat), "type": o.obstacle_type,
                "confidence": float(o.confidence), "risk_level": float(o.risk_level), "size": float(o.size)
            } for o in results['obstacles']
        ],
        "path_points": [
            {
                "lon": float(p.lon), "lat": float(p.lat), "risk_score": float(p.risk_score),
                "is_safe": bool(p.is_safe), "distance_from_start": float(p.distance_from_start)
            } for p in results['path_points']
        ]
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    print("\n=== Pathfinding Complete ===")
    print("Open the HTML file in a web browser to view the interactive map!")

if __name__ == "__main__":
    main()
