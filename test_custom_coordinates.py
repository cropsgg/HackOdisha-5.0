#!/usr/bin/env python3
"""
Test custom coordinates with direct HTML generation
"""

import sys
from pathlib import Path
from efficient_pathfinding_system import EfficientPathfindingSystem

def test_custom_coordinates():
    """Test custom coordinates and generate HTML directly"""
    
    print("🌙 Testing Custom Coordinates with Direct HTML Generation")
    print("=" * 60)
    
    # Initialize system
    system = EfficientPathfindingSystem()
    
    # Test coordinates - these should be well within the lunar image bounds
    start_lon, start_lat = 41.440, -19.700  # Custom start position
    end_lon, end_lat = 41.470, -19.730      # Custom end position
    
    print(f"📍 Start: {start_lat:.3f}°N, {start_lon:.3f}°E")
    print(f"📍 End:   {end_lat:.3f}°N, {end_lon:.3f}°E")
    print()
    
    # Mission parameters
    image_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_b_brw_d18.png"
    csv_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_g_grd_d18.csv"
    
    # Process pathfinding
    results = system.process_pathfinding(
        image_path, csv_path, start_lon, start_lat, end_lon, end_lat
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
    
    # Save results with custom filename
    output_dir = Path("pathfinding_results")
    output_dir.mkdir(exist_ok=True)
    
    map_path = output_dir / "custom_coordinates_test.html"
    results['folium_map'].save(str(map_path))
    print(f"\n🗺️ Map saved: {map_path}")
    print("Open this file in a web browser to see the custom coordinates visualization!")

if __name__ == "__main__":
    test_custom_coordinates()
