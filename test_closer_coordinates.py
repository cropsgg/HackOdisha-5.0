#!/usr/bin/env python3
"""
Test script with closer coordinates for better pathfinding
"""

import sys
from pathlib import Path
from efficient_pathfinding_system import EfficientPathfindingSystem, format_selenographic_coords

def test_closer_coordinates():
    """Test with closer coordinates that should have fewer obstacles"""
    
    print("=== Testing with Closer Coordinates ===")
    
    # Initialize system
    system = EfficientPathfindingSystem()
    
    # Test with coordinates that are closer together
    image_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_b_brw_d18.png"
    csv_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_g_grd_d18.csv"
    
    # Test case with closer coordinates
    start_lon, start_lat = 41.406, -19.694
    end_lon, end_lat = 41.450, -19.720  # Closer together
    
    print(f"Start: {format_selenographic_coords(start_lon, start_lat)}")
    print(f"End: {format_selenographic_coords(end_lon, end_lat)}")
    
    # Process pathfinding
    results = system.process_pathfinding(
        image_path, csv_path, 
        start_lon, start_lat,
        end_lon, end_lat
    )
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    # Display results
    print(f"✅ Path found: {results['statistics']['path_found']}")
    print(f"📏 Path length: {results['statistics']['path_length_km']:.3f} km")
    print(f"🛡️ Safety: {results['statistics']['safety_percentage']:.1f}%")
    print(f"🚧 Obstacles: {results['statistics']['total_obstacles']} total")
    print(f"   - Boulders: {results['statistics']['boulders']}")
    print(f"   - Landslides: {results['statistics']['landslides']}")
    print(f"📍 Path points: {len(results['path_points'])}")
    
    # Save results
    output_dir = Path("pathfinding_results")
    output_dir.mkdir(exist_ok=True)
    
    map_path = output_dir / "closer_coordinates_test.html"
    results['folium_map'].save(str(map_path))
    print(f"\n🗺️ Map saved: {map_path}")
    print("Open this file in a web browser to check the path!")

if __name__ == "__main__":
    test_closer_coordinates()
