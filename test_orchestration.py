#!/usr/bin/env python3
"""
Test script for the Lunar Rover Orchestration System
"""

import sys
from pathlib import Path
from lunar_rover_orchestrator import LunarRoverOrchestrator

def test_orchestration():
    """Test the orchestration system with different scenarios"""
    
    print("🚀 Testing Lunar Rover Orchestration System")
    print("=" * 60)
    
    # Initialize orchestrator
    power_csv_path = "synthetic_rover_power_nomode.csv"
    orchestrator = LunarRoverOrchestrator(power_csv_path)
    
    # Test mission
    image_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_b_brw_d18.png"
    csv_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_g_grd_d18.csv"
    
    # Test coordinates
    start_lon, start_lat = 41.406, -19.694
    end_lon, end_lat = 41.450, -19.720
    
    print(f"📍 Start: {start_lon:.3f}°E, {start_lat:.3f}°N")
    print(f"📍 End: {end_lon:.3f}°E, {end_lat:.3f}°N")
    print()
    
    # Run the mission
    result = orchestrator.run_mission(
        image_path, csv_path, start_lon, start_lat, end_lon, end_lat,
        "Test Mission"
    )
    
    print("\n🎯 Mission completed!")
    print("Check the 'mission_results' folder for:")
    print("- HTML visualization map")
    print("- JSON mission report")
    print("- Detailed statistics")

if __name__ == "__main__":
    test_orchestration()
