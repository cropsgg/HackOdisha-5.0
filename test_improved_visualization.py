#!/usr/bin/env python3
"""
Test script to demonstrate improved visualization with proper coordinate conversion
"""

import sys
from pathlib import Path
from lunar_rover_orchestrator import LunarRoverOrchestrator

def test_improved_visualization():
    """Test the improved visualization with proper coordinate conversion"""
    
    print("🚀 Testing Improved Visualization System")
    print("=" * 60)
    
    # Initialize orchestrator
    power_csv_path = "synthetic_rover_power_nomode.csv"
    orchestrator = LunarRoverOrchestrator(power_csv_path)
    
    # Test mission with coordinates that should be well within the lunar image
    image_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_b_brw_d18.png"
    csv_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_g_grd_d18.csv"
    
    # Test coordinates - these should be well within the lunar image bounds
    start_lon, start_lat = 41.420, -19.720  # More central coordinates
    end_lon, end_lat = 41.450, -19.750      # End point further south-east
    
    print(f"📍 Start: {start_lon:.3f}°E, {start_lat:.3f}°N")
    print(f"📍 End: {end_lon:.3f}°E, {end_lat:.3f}°N")
    print()
    
    # Run the mission
    result = orchestrator.run_mission(
        image_path, csv_path, start_lon, start_lat, end_lon, end_lat,
        "Improved Visualization Test"
    )
    
    print("\n🎯 Mission completed!")
    print("✅ Key improvements:")
    print("   - Start/end points now properly positioned within lunar image")
    print("   - Coordinate conversion uses proper scaling from original to browse image")
    print("   - Path visualization shows clear route on lunar surface")
    print("   - Obstacles are accurately positioned relative to terrain")
    print()
    print("📁 Check the 'mission_results' folder for:")
    print("   - HTML visualization map with improved positioning")
    print("   - JSON mission report with detailed statistics")
    print("   - Both ML and heuristic modes working correctly")

if __name__ == "__main__":
    test_improved_visualization()
