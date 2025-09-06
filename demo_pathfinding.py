#!/usr/bin/env python3
"""
Demonstration Script for Integrated Lunar Pathfinding System
Shows how to use the system with specific Chandrayaan-2 coordinates
"""

import sys
from pathlib import Path
import json

# Add the current directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "Landslide" / "src" / "coordinate_utils"))

from integrated_pathfinding_system import IntegratedPathfindingSystem
from coordinate_utils.selenographic_converter import format_selenographic_coords

def demo_with_specific_coordinates():
    """Demonstrate pathfinding with specific coordinates from Chandrayaan-2 data"""
    
    print("=== Lunar Pathfinding Demonstration ===")
    print("Using real Chandrayaan-2 coordinates for optimal path planning")
    
    # Initialize the system
    system = IntegratedPathfindingSystem()
    
    # Get available observations
    observations = system.data_loader.get_available_observations()
    if not observations:
        print("❌ No Chandrayaan-2 observations available")
        return
    
    print(f"✅ Found {len(observations)} Chandrayaan-2 observations")
    
    # Use a specific observation (you can change this)
    obs_id = observations[0]  # Use first observation
    print(f"📍 Using observation: {obs_id}")
    
    # Load coordinate data to get valid coordinate ranges
    coordinates = system.data_loader.load_coordinate_data(obs_id)
    if coordinates is None:
        print("❌ Failed to load coordinate data")
        return
    
    # Get coordinate bounds
    min_lon, max_lon = coordinates['Longitude'].min(), coordinates['Longitude'].max()
    min_lat, max_lat = coordinates['Latitude'].min(), coordinates['Latitude'].max()
    
    print(f"🗺️  Coordinate bounds:")
    print(f"   Longitude: {min_lon:.6f}° to {max_lon:.6f}°")
    print(f"   Latitude: {min_lat:.6f}° to {max_lat:.6f}°")
    
    # Define start and end points within the coordinate bounds
    # These are example coordinates - you can modify them
    start_lon = min_lon + (max_lon - min_lon) * 0.2  # 20% from left edge
    start_lat = min_lat + (max_lat - min_lat) * 0.3  # 30% from bottom edge
    
    end_lon = min_lon + (max_lon - min_lon) * 0.8    # 80% from left edge
    end_lat = min_lat + (max_lat - min_lat) * 0.7    # 70% from bottom edge
    
    print(f"\n🎯 Path Planning:")
    print(f"   Start: {format_selenographic_coords(start_lon, start_lat)}")
    print(f"   End: {format_selenographic_coords(end_lon, end_lat)}")
    
    # Process complete pathfinding
    print(f"\n🔍 Processing pathfinding...")
    results = system.process_complete_pathfinding(obs_id, start_lon, start_lat, end_lon, end_lat)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Display results
    print(f"\n✅ Pathfinding Results:")
    print(f"   📊 Total obstacles: {results['statistics']['total_obstacles']}")
    print(f"   🪨 Boulders: {results['statistics']['boulders']}")
    print(f"   🏔️  Landslides: {results['statistics']['landslides']}")
    print(f"   🛤️  Path found: {'Yes' if results['statistics']['path_found'] else 'No'}")
    print(f"   📏 Path length: {results['statistics']['path_length_km']:.2f} km")
    print(f"   🛡️  Safety: {results['statistics']['safety_percentage']:.1f}%")
    
    # Save results
    output_dir = Path("pathfinding_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save interactive map
    map_path = output_dir / f"lunar_pathfinding_demo_{obs_id}.html"
    results['folium_map'].save(str(map_path))
    print(f"\n🗺️  Interactive map saved: {map_path}")
    
    # Save detailed results
    json_path = output_dir / f"pathfinding_demo_{obs_id}.json"
    with open(json_path, 'w') as f:
        json.dump(results['statistics'], f, indent=2)
    print(f"📄 Results saved: {json_path}")
    
    print(f"\n🎉 Demonstration complete!")
    print(f"   Open {map_path} in your web browser to see the interactive map")
    print(f"   The map shows boulders (blue circles), landslides (red markers),")
    print(f"   start point (green), goal point (red), and optimal path (green line)")

def demo_with_custom_coordinates():
    """Demonstrate pathfinding with user-specified coordinates"""
    
    print("\n=== Custom Coordinate Demo ===")
    print("Enter your own start and end coordinates")
    
    system = IntegratedPathfindingSystem()
    observations = system.data_loader.get_available_observations()
    
    if not observations:
        print("❌ No observations available")
        return
    
    # Show available observations
    print(f"\nAvailable observations:")
    for i, obs_id in enumerate(observations[:5]):  # Show first 5
        print(f"   {i+1}. {obs_id}")
    
    # Use first observation for demo
    obs_id = observations[0]
    coordinates = system.data_loader.load_coordinate_data(obs_id)
    
    if coordinates is None:
        print("❌ Failed to load coordinates")
        return
    
    # Get bounds
    min_lon, max_lon = coordinates['Longitude'].min(), coordinates['Longitude'].max()
    min_lat, max_lat = coordinates['Latitude'].min(), coordinates['Latitude'].max()
    
    print(f"\nCoordinate bounds for {obs_id}:")
    print(f"   Longitude: {min_lon:.6f}° to {max_lon:.6f}°")
    print(f"   Latitude: {min_lat:.6f}° to {max_lat:.6f}°")
    
    # Example coordinates (you can modify these)
    start_lon, start_lat = 41.406, -19.694  # Example from CSV data
    end_lon, end_lat = 41.420, -19.690      # Example from CSV data
    
    print(f"\nUsing example coordinates:")
    print(f"   Start: {format_selenographic_coords(start_lon, start_lat)}")
    print(f"   End: {format_selenographic_coords(end_lon, end_lat)}")
    
    # Process pathfinding
    results = system.process_complete_pathfinding(obs_id, start_lon, start_lat, end_lon, end_lat)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Save results
    output_dir = Path("pathfinding_results")
    output_dir.mkdir(exist_ok=True)
    
    map_path = output_dir / f"custom_pathfinding_{obs_id}.html"
    results['folium_map'].save(str(map_path))
    
    print(f"\n✅ Custom pathfinding complete!")
    print(f"   Map saved: {map_path}")
    print(f"   Obstacles: {results['statistics']['total_obstacles']}")
    print(f"   Path length: {results['statistics']['path_length_km']:.2f} km")

def show_available_data():
    """Show information about available Chandrayaan-2 data"""
    
    print("\n=== Available Chandrayaan-2 Data ===")
    
    system = IntegratedPathfindingSystem()
    observations = system.data_loader.get_available_observations()
    
    print(f"Total observations: {len(observations)}")
    
    for i, obs_id in enumerate(observations[:3]):  # Show first 3
        print(f"\n{i+1}. {obs_id}")
        
        # Load coordinate data
        coordinates = system.data_loader.load_coordinate_data(obs_id)
        if coordinates is not None:
            min_lon, max_lon = coordinates['Longitude'].min(), coordinates['Longitude'].max()
            min_lat, max_lat = coordinates['Latitude'].min(), coordinates['Latitude'].max()
            
            print(f"   📍 Coordinate bounds:")
            print(f"      Longitude: {min_lon:.6f}° to {max_lon:.6f}°")
            print(f"      Latitude: {min_lat:.6f}° to {max_lat:.6f}°")
            print(f"   📊 Data points: {len(coordinates)}")
            
            # Show sample coordinates
            print(f"   📋 Sample coordinates:")
            for j in range(min(3, len(coordinates))):
                row = coordinates.iloc[j]
                print(f"      {format_selenographic_coords(row['Longitude'], row['Latitude'])}")
        else:
            print("   ❌ No coordinate data available")

if __name__ == "__main__":
    print("🚀 Lunar Pathfinding System Demo")
    print("=" * 50)
    
    # Show available data
    show_available_data()
    
    # Run demonstrations
    demo_with_specific_coordinates()
    demo_with_custom_coordinates()
    
    print("\n" + "=" * 50)
    print("🎯 Demo complete! Check the 'pathfinding_results' folder for outputs.")
    print("   Open the HTML files in your web browser to see interactive maps.")
