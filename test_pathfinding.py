#!/usr/bin/env python3
"""
Test script for the efficient pathfinding system with different coordinates
"""

import sys
from pathlib import Path
from efficient_pathfinding_system import EfficientPathfindingSystem, format_selenographic_coords

def test_pathfinding():
    """Test pathfinding with different coordinate scenarios"""
    
    print("=== Testing Efficient Lunar Pathfinding System ===")
    
    # Initialize system
    system = EfficientPathfindingSystem()
    
    # Test scenarios
    scenarios = [
        {
            "name": "Short Distance Test",
            "start": (41.406, -19.694),
            "end": (41.410, -19.692)
        },
        {
            "name": "Medium Distance Test", 
            "start": (41.406, -19.694),
            "end": (41.420, -19.690)
        },
        {
            "name": "Long Distance Test",
            "start": (41.400, -19.700),
            "end": (41.430, -19.680)
        }
    ]
    
    # Use the same image and CSV for all tests
    image_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_b_brw_d18.png"
    csv_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_g_grd_d18.csv"
    
    for i, scenario in enumerate(scenarios):
        print(f"\n--- {scenario['name']} ---")
        print(f"Start: {format_selenographic_coords(scenario['start'][0], scenario['start'][1])}")
        print(f"End: {format_selenographic_coords(scenario['end'][0], scenario['end'][1])}")
        
        # Process pathfinding
        results = system.process_pathfinding(
            image_path, csv_path, 
            scenario['start'][0], scenario['start'][1],
            scenario['end'][0], scenario['end'][1]
        )
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            continue
        
        # Display results
        print(f"✅ Path found: {results['statistics']['path_found']}")
        print(f"📏 Path length: {results['statistics']['path_length_km']:.2f} km")
        print(f"🛡️ Safety: {results['statistics']['safety_percentage']:.1f}%")
        print(f"🚧 Obstacles: {results['statistics']['total_obstacles']} total")
        print(f"   - Boulders: {results['statistics']['boulders']}")
        print(f"   - Landslides: {results['statistics']['landslides']}")
        
        # Save results
        output_dir = Path("pathfinding_results")
        output_dir.mkdir(exist_ok=True)
        
        map_path = output_dir / f"test_{i+1}_{scenario['name'].replace(' ', '_').lower()}.html"
        results['folium_map'].save(str(map_path))
        print(f"🗺️ Map saved: {map_path}")

if __name__ == "__main__":
    test_pathfinding()
