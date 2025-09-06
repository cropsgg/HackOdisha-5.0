#!/usr/bin/env python3
"""
Simplified Integrated Lunar Pathfinding System
A working version that handles import issues gracefully
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
from folium import plugins
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

class SimpleChandrayaanDataLoader:
    """Simplified Chandrayaan-2 data loader"""
    
    def __init__(self, data_dir: str = "chandrayaan-2"):
        self.data_dir = Path(data_dir)
        self.observations = self._discover_observations()
    
    def _discover_observations(self) -> Dict[str, Dict]:
        """Discover all available Chandrayaan-2 observations"""
        observations = {}
        
        if not self.data_dir.exists():
            print(f"Warning: Chandrayaan-2 data directory not found: {self.data_dir}")
            return observations
        
        # Find all observation directories/files
        for file_path in self.data_dir.glob("*.img"):
            # Extract observation ID from filename
            filename = file_path.stem
            parts = filename.split('_')
            if len(parts) >= 4:
                obs_id = '_'.join(parts[:4])  # ch2_ohr_ncp_20240425T1603031918
                
                if obs_id not in observations:
                    observations[obs_id] = {}
                
                # Store file paths
                observations[obs_id]['img_file'] = file_path
                # Look for browse PNG with correct naming pattern
                browse_png_pattern = f"{obs_id}_b_brw_*.png"
                browse_png_files = list(self.data_dir.glob(browse_png_pattern))
                if browse_png_files:
                    observations[obs_id]['browse_png'] = browse_png_files[0]
                
                # Look for corresponding CSV file
                csv_pattern = f"{obs_id}_g_grd_*.csv"
                csv_files = list(self.data_dir.glob(csv_pattern))
                if csv_files:
                    observations[obs_id]['csv_file'] = csv_files[0]
        
        print(f"Discovered {len(observations)} Chandrayaan-2 observations")
        return observations
    
    def get_available_observations(self) -> List[str]:
        """Get list of available observation IDs"""
        return list(self.observations.keys())
    
    def load_observation_image(self, obs_id: str) -> Optional[np.ndarray]:
        """Load the main image data for an observation"""
        if obs_id not in self.observations:
            print(f"Observation {obs_id} not found")
            return None
        
        browse_png = self.observations[obs_id].get('browse_png')
        if not browse_png or not browse_png.exists():
            print(f"Browse PNG not found for observation {obs_id}")
            return None
        
        try:
            image = cv2.imread(str(browse_png))
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return None
        except Exception as e:
            print(f"Error loading image for observation {obs_id}: {e}")
            return None
    
    def load_coordinate_data(self, obs_id: str) -> Optional[pd.DataFrame]:
        """Load coordinate data from CSV file"""
        if obs_id not in self.observations:
            print(f"Observation {obs_id} not found")
            return None
        
        csv_file = self.observations[obs_id].get('csv_file')
        if not csv_file or not csv_file.exists():
            print(f"CSV file not found for observation {obs_id}")
            return None
        
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded coordinate data: {len(df)} points for observation {obs_id}")
            return df
        except Exception as e:
            print(f"Error loading CSV for observation {obs_id}: {e}")
            return None

class SimpleBoulderDetector:
    """Simplified boulder detector using basic computer vision"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
    
    def detect_boulders(self, image: np.ndarray) -> List[Dict]:
        """Detect boulders using simple computer vision techniques"""
        # Resize image for faster processing
        height, width = image.shape[:2]
        scale_factor = min(1.0, 1000.0 / max(height, width))  # Limit to 1000px max dimension
        
        if scale_factor < 1.0:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = cv2.resize(image, (new_width, new_height))
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height} for faster processing")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect circles (boulders are roughly circular) with stricter parameters
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,  # Increased minDist
            param1=50, param2=40, minRadius=15, maxRadius=80  # Stricter parameters
        )
        
        boulders = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Limit to top 50 most confident detections
            circle_data = []
            for (x, y, r) in circles:
                # Calculate bounding box
                bbox = [x - r, y - r, x + r, y + r]
                area = np.pi * r * r
                
                # Simple confidence based on circle detection quality
                confidence = min(0.9, 0.5 + (r / 100.0))
                
                if confidence >= self.confidence_threshold:
                    circle_data.append((confidence, {
                        'bbox': bbox,
                        'confidence': confidence,
                        'area': area,
                        'radius': r
                    }))
            
            # Sort by confidence and take top 50
            circle_data.sort(key=lambda x: x[0], reverse=True)
            boulders = [data[1] for data in circle_data[:50]]  # Limit to 50 boulders
        
        logger.info(f"Detected {len(boulders)} boulders using optimized CV (limited to top 50)")
        return boulders

class SimpleLandslideDetector:
    """Simplified landslide detector using basic image analysis"""
    
    def __init__(self, config: Dict):
        self.critical_slope = config.get('critical_slope', 30.0)
        self.detection_threshold = config.get('detection_threshold', 0.6)
    
    def detect_landslides(self, terrain_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Detect landslides using simple slope analysis"""
        slope = terrain_data.get('slope', np.zeros((100, 100)))
        
        # Create landslide mask based on slope with higher threshold
        landslide_mask = (slope > self.critical_slope * 1.5).astype(np.float32)  # Higher threshold
        landslide_score = np.clip(slope / 90.0, 0, 1)  # Normalize slope to 0-1
        
        # Apply heavy smoothing to reduce noise
        landslide_mask = cv2.GaussianBlur(landslide_mask, (15, 15), 0)
        
        # Only keep areas with very high risk
        landslide_mask = (landslide_mask > 0.8).astype(np.float32)
        
        # Calculate statistics
        high_risk_pixels = np.sum(landslide_mask > self.detection_threshold)
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
        
        logger.info(f"Landslide detection completed: {landslide_percentage:.2f}% high risk (optimized)")
        return results

class SimplifiedPathfindingSystem:
    """Simplified integrated pathfinding system"""
    
    def __init__(self, chandrayaan_data_dir: str = "chandrayaan-2"):
        """Initialize the simplified pathfinding system"""
        self.chandrayaan_data_dir = chandrayaan_data_dir
        self.data_loader = SimpleChandrayaanDataLoader(chandrayaan_data_dir)
        
        # Initialize detection systems
        self.boulder_detector = SimpleBoulderDetector(confidence_threshold=0.5)
        
        # Physics-based landslide detector config
        landslide_config = {
            'critical_slope': 30.0,
            'detection_threshold': 0.6
        }
        self.landslide_detector = SimpleLandslideDetector(landslide_config)
        
        # Pathfinding parameters
        self.max_risk_score = 0.3  # 0-1 scale
        self.safety_margin = 50.0  # meters from obstacles
        
        logger.info("Simplified Pathfinding System initialized")
        logger.info(f"Available Chandrayaan-2 observations: {len(self.data_loader.get_available_observations())}")
    
    def load_chandrayaan_observation(self, obs_id: str) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Load Chandrayaan-2 observation data"""
        try:
            image = self.data_loader.load_observation_image(obs_id)
            coordinates = self.data_loader.load_coordinate_data(obs_id)
            
            if image is not None and coordinates is not None:
                logger.info(f"Loaded observation {obs_id}: image shape {image.shape}, {len(coordinates)} coordinate points")
                return image, coordinates
            else:
                logger.warning(f"Failed to load complete data for observation {obs_id}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error loading observation {obs_id}: {e}")
            return None, None
    
    def detect_boulders(self, image: np.ndarray) -> List[Dict]:
        """Detect boulders in the image"""
        try:
            boulders = self.boulder_detector.detect_boulders(image)
            logger.info(f"Detected {len(boulders)} boulders")
            return boulders
        except Exception as e:
            logger.error(f"Error in boulder detection: {e}")
            return []
    
    def detect_landslides(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect landslides using physics-based criteria"""
        try:
            # Convert image to terrain data format
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Create synthetic terrain data
            terrain_data = self._create_synthetic_terrain_data(gray)
            
            results = self.landslide_detector.detect_landslides(terrain_data)
            logger.info(f"Landslide detection completed: {results['statistics']['landslide_percentage']:.2f}% high risk")
            return results
            
        except Exception as e:
            logger.error(f"Error in landslide detection: {e}")
            return {}
    
    def _create_synthetic_terrain_data(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Create synthetic terrain data from image"""
        # Use image intensity as elevation proxy
        dtm = image.astype(np.float32)
        
        # Calculate slope from image gradients
        grad_x = cv2.Sobel(dtm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(dtm, cv2.CV_32F, 0, 1, ksize=3)
        slope = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * 180 / np.pi
        
        return {
            'dtm': dtm,
            'slope': slope
        }
    
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
        
        # Convert landslide areas to coordinate points (limit to top 20)
        if 'landslide_mask' in landslide_results:
            landslide_mask = landslide_results['landslide_mask']
            landslide_score = landslide_results.get('landslide_score', np.zeros_like(landslide_mask))
            
            # Find high-risk landslide areas
            high_risk_mask = (landslide_mask > 0) & (landslide_score > 0.8)  # Higher threshold
            contours, _ = cv2.findContours(high_risk_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area and take only the largest ones
            contour_areas = [(cv2.contourArea(contour), contour) for contour in contours if cv2.contourArea(contour) > 500]  # Higher area threshold
            contour_areas.sort(key=lambda x: x[0], reverse=True)
            
            for area, contour in contour_areas[:20]:  # Limit to top 20 landslide areas
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
                        
                        obstacle = ObstaclePoint(
                            lon=lon, lat=lat, x=center_x, y=center_y,
                            obstacle_type='landslide',
                            confidence=avg_risk,
                            risk_level=avg_risk,
                            size=area
                        )
                        obstacles.append(obstacle)
        
        logger.info(f"Converted {len(obstacles)} obstacles to Selenographic coordinates")
        return obstacles
    
    def _image_to_selenographic(self, x: float, y: float, coordinates: pd.DataFrame, 
                               image_shape: Tuple[int, int]) -> Tuple[Optional[float], Optional[float]]:
        """Convert image coordinates to Selenographic coordinates"""
        try:
            # Normalize image coordinates to 0-1 range
            norm_x = x / image_shape[1]
            norm_y = y / image_shape[0]
            
            # Use coordinate bounds for interpolation
            min_lon, max_lon = coordinates['Longitude'].min(), coordinates['Longitude'].max()
            min_lat, max_lat = coordinates['Latitude'].min(), coordinates['Latitude'].max()
            
            lon = min_lon + norm_x * (max_lon - min_lon)
            lat = min_lat + norm_y * (max_lat - min_lat)
            
            return lon, lat
            
        except Exception as e:
            logger.error(f"Error converting coordinates: {e}")
            return None, None
    
    def create_obstacle_map(self, obstacles: List[ObstaclePoint], bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Create a grid-based obstacle map for pathfinding"""
        min_lon, max_lon, min_lat, max_lat = bounds
        
        # Create smaller grid for faster processing
        grid_size = 50  # 50x50 grid (reduced from 100x100)
        obstacle_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Map obstacles to grid
        for obstacle in obstacles:
            # Convert Selenographic coordinates to grid coordinates
            grid_x = int((obstacle.lon - min_lon) / (max_lon - min_lon) * grid_size)
            grid_y = int((obstacle.lat - min_lat) / (max_lat - min_lat) * grid_size)
            
            # Ensure within bounds
            grid_x = np.clip(grid_x, 0, grid_size - 1)
            grid_y = np.clip(grid_y, 0, grid_size - 1)
            
            # Set obstacle value based on risk level and type
            if obstacle.obstacle_type == 'boulder':
                obstacle_value = obstacle.risk_level * 2.0  # Boulders are high priority
            else:  # landslide
                obstacle_value = obstacle.risk_level * 1.5
            
            # Apply to grid with some spreading for safety margin
            radius = max(2, int(obstacle.size / 100))  # Safety radius
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        distance = np.sqrt(dx*dx + dy*dy)
                        if distance <= radius:
                            # Gaussian falloff
                            weight = np.exp(-distance**2 / (2 * (radius/2)**2))
                            obstacle_map[ny, nx] = max(obstacle_map[ny, nx], obstacle_value * weight)
        
        return obstacle_map
    
    def find_optimal_path(self, start_lon: float, start_lat: float, end_lon: float, end_lat: float,
                         obstacles: List[ObstaclePoint], coordinates: pd.DataFrame) -> List[PathPoint]:
        """Find optimal path using A* algorithm with obstacle avoidance"""
        
        # Get coordinate bounds
        min_lon, max_lon = coordinates['Longitude'].min(), coordinates['Longitude'].max()
        min_lat, max_lat = coordinates['Latitude'].min(), coordinates['Latitude'].max()
        bounds = (min_lon, max_lon, min_lat, max_lat)
        
        # Create obstacle map
        obstacle_map = self.create_obstacle_map(obstacles, bounds)
        
        # Create graph for pathfinding
        graph = nx.Graph()
        grid_size = obstacle_map.shape[0]
        
        # Add nodes to graph
        for y in range(grid_size):
            for x in range(grid_size):
                # Convert grid coordinates to Selenographic coordinates
                lon = min_lon + (x / grid_size) * (max_lon - min_lon)
                lat = min_lat + (y / grid_size) * (max_lat - min_lat)
                
                # Check if location is safe
                risk_score = obstacle_map[y, x]
                is_safe = risk_score < self.max_risk_score
                
                if is_safe or (x, y) in [(0, 0), (grid_size-1, grid_size-1)]:  # Allow start/end even if risky
                    graph.add_node((x, y), lon=lon, lat=lat, risk=risk_score)
        
        # Add edges between nearby nodes
        for node in graph.nodes():
            x, y = node
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    neighbor = (x + dx, y + dy)
                    if neighbor in graph.nodes():
                        # Calculate cost based on distance and risk
                        start_risk = graph.nodes[node]['risk']
                        end_risk = graph.nodes[neighbor]['risk']
                        avg_risk = (start_risk + end_risk) / 2
                        
                        # Distance cost (diagonal moves cost more)
                        distance_cost = np.sqrt(dx*dx + dy*dy)
                        
                        # Risk cost
                        risk_cost = avg_risk * 10  # Scale risk cost
                        
                        total_cost = distance_cost + risk_cost
                        
                        graph.add_edge(node, neighbor, weight=total_cost)
        
        # Find start and end grid coordinates
        start_x = int((start_lon - min_lon) / (max_lon - min_lon) * grid_size)
        start_y = int((start_lat - min_lat) / (max_lat - min_lat) * grid_size)
        end_x = int((end_lon - min_lon) / (max_lon - min_lon) * grid_size)
        end_y = int((end_lat - min_lat) / (max_lat - min_lat) * grid_size)
        
        # Ensure coordinates are within bounds
        start_x = np.clip(start_x, 0, grid_size - 1)
        start_y = np.clip(start_y, 0, grid_size - 1)
        end_x = np.clip(end_x, 0, grid_size - 1)
        end_y = np.clip(end_y, 0, grid_size - 1)
        
        start_node = (start_x, start_y)
        end_node = (end_x, end_y)
        
        # Add start and end nodes if not in graph
        if start_node not in graph.nodes():
            graph.add_node(start_node, lon=start_lon, lat=start_lat, risk=0)
        if end_node not in graph.nodes():
            graph.add_node(end_node, lon=end_lon, lat=end_lat, risk=0)
        
        # Find shortest path using A*
        try:
            path_nodes = nx.astar_path(
                graph, start_node, end_node,
                heuristic=lambda n1, n2: np.sqrt((n1[0]-n2[0])**2 + (n1[1]-n2[1])**2),
                weight='weight'
            )
            
            # Convert path nodes to PathPoint objects
            path_points = []
            total_distance = 0.0
            
            for i, (x, y) in enumerate(path_nodes):
                node_data = graph.nodes[(x, y)]
                lon, lat = node_data['lon'], node_data['lat']
                risk_score = node_data['risk']
                
                # Calculate distance from start
                if i > 0:
                    prev_lon, prev_lat = graph.nodes[path_nodes[i-1]]['lon'], graph.nodes[path_nodes[i-1]]['lat']
                    segment_distance = self._calculate_lunar_distance(prev_lon, prev_lat, lon, lat)
                    total_distance += segment_distance
                
                point = PathPoint(
                    lon=lon, lat=lat, x=x, y=y,
                    elevation=0.0,  # Would need actual elevation data
                    slope=0.0,      # Would need actual slope data
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
                                  obs_id: str) -> folium.Map:
        """Create interactive folium map visualization"""
        
        # Calculate map center
        center_lat = (start_lat + end_lat) / 2
        center_lon = (start_lon + end_lon) / 2
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
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
                    radius=max(3, int(obstacle.size / 50)),  # Scale marker size
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
        
        # Add optimal path
        if path_points:
            path_coords = [[point.lat, point.lon] for point in path_points]
            folium.PolyLine(
                path_coords,
                color='green',
                weight=4,
                opacity=0.8,
                popup=f"Optimal Path<br>Length: {path_points[-1].distance_from_start:.2f} km<br>Points: {len(path_points)}"
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
        Lunar Pathfinding - Chandrayaan-2 Observation: {obs_id}
        </h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def process_complete_pathfinding(self, obs_id: str, start_lon: float, start_lat: float,
                                   end_lon: float, end_lat: float) -> Dict[str, Any]:
        """Complete pathfinding pipeline"""
        logger.info(f"Starting complete pathfinding for observation {obs_id}")
        
        # Load Chandrayaan-2 data
        image, coordinates = self.load_chandrayaan_observation(obs_id)
        if image is None or coordinates is None:
            return {"error": "Failed to load Chandrayaan-2 data"}
        
        # Detect obstacles
        boulders = self.detect_boulders(image)
        landslide_results = self.detect_landslides(image)
        
        # Convert obstacles to coordinates
        obstacles = self.convert_obstacles_to_coordinates(boulders, landslide_results, coordinates, image.shape[:2])
        
        # Find optimal path
        path_points = self.find_optimal_path(start_lon, start_lat, end_lon, end_lat, obstacles, coordinates)
        
        # Create visualization
        folium_map = self.create_folium_visualization(obstacles, path_points, start_lon, start_lat, end_lon, end_lat, obs_id)
        
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
    """Main function to demonstrate the simplified pathfinding system"""
    print("=== Simplified Lunar Pathfinding System ===")
    
    # Initialize system
    system = SimplifiedPathfindingSystem()
    
    # Get available observations
    observations = system.data_loader.get_available_observations()
    if not observations:
        print("No Chandrayaan-2 observations available")
        return
    
    print(f"Available observations: {len(observations)}")
    
    # Use first observation for demonstration
    obs_id = observations[0]
    print(f"Using observation: {obs_id}")
    
    # Define start and end points (example coordinates from the CSV data)
    start_lon, start_lat = 41.4, -19.7  # Example start point
    end_lon, end_lat = 41.5, -19.6      # Example end point
    
    print(f"Start: {format_selenographic_coords(start_lon, start_lat)}")
    print(f"End: {format_selenographic_coords(end_lon, end_lat)}")
    
    # Process complete pathfinding
    results = system.process_complete_pathfinding(obs_id, start_lon, start_lat, end_lon, end_lat)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Display results
    print("\n=== Results ===")
    print(f"Observation: {results['observation_id']}")
    print(f"Total obstacles detected: {results['statistics']['total_obstacles']}")
    print(f"  - Boulders: {results['statistics']['boulders']}")
    print(f"  - Landslides: {results['statistics']['landslides']}")
    print(f"Path found: {results['statistics']['path_found']}")
    print(f"Path length: {results['statistics']['path_length_km']:.2f} km")
    print(f"Safety: {results['statistics']['safety_percentage']:.1f}%")
    
    # Save folium map
    output_dir = Path("pathfinding_results")
    output_dir.mkdir(exist_ok=True)
    
    map_path = output_dir / f"simplified_lunar_pathfinding_{obs_id}.html"
    results['folium_map'].save(str(map_path))
    print(f"Interactive map saved to: {map_path}")
    
    # Save results as JSON
    json_path = output_dir / f"simplified_results_{obs_id}.json"
    
    # Convert results to JSON-serializable format
    json_results = {
        "observation_id": results['observation_id'],
        "statistics": results['statistics'],
        "coordinate_bounds": results['coordinate_bounds'],
        "obstacles": [
            {
                "lon": o.lon, "lat": o.lat, "type": o.obstacle_type,
                "confidence": o.confidence, "risk_level": o.risk_level, "size": o.size
            } for o in results['obstacles']
        ],
        "path_points": [
            {
                "lon": p.lon, "lat": p.lat, "risk_score": p.risk_score,
                "is_safe": p.is_safe, "distance_from_start": p.distance_from_start
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
