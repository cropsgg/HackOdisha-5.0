#!/usr/bin/env python3
"""
Custom Coordinates Demo
Allows you to specify start and end coordinates as command line arguments
"""

import sys
import argparse
from pathlib import Path
from lunar_rover_orchestrator import LunarRoverOrchestrator

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Lunar Pathfinding with Custom Coordinates')
    
    parser.add_argument('--start-lon', type=float, required=True,
                       help='Start longitude (e.g., 41.420)')
    parser.add_argument('--start-lat', type=float, required=True,
                       help='Start latitude (e.g., -19.720)')
    parser.add_argument('--end-lon', type=float, required=True,
                       help='End longitude (e.g., 41.450)')
    parser.add_argument('--end-lat', type=float, required=True,
                       help='End latitude (e.g., -19.750)')
    parser.add_argument('--mission-name', type=str, default='Custom Mission',
                       help='Mission name (default: Custom Mission)')
    
    return parser.parse_args()

def validate_coordinates(start_lon, start_lat, end_lon, end_lat):
    """Validate that coordinates are within valid ranges"""
    import pandas as pd
    
    try:
        coordinates = pd.read_csv("chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_g_grd_d18.csv")
        min_lon = coordinates['Longitude'].min()
        max_lon = coordinates['Longitude'].max()
        min_lat = coordinates['Latitude'].min()
        max_lat = coordinates['Latitude'].max()
        
        print(f"📍 Valid coordinate ranges:")
        print(f"   Longitude: {min_lon:.3f}° to {max_lon:.3f}°")
        print(f"   Latitude: {min_lat:.3f}° to {max_lat:.3f}°")
        print()
        
        # Validate coordinates
        if not (min_lon <= start_lon <= max_lon):
            raise ValueError(f"Start longitude {start_lon:.3f} is outside valid range ({min_lon:.3f} to {max_lon:.3f})")
        if not (min_lon <= end_lon <= max_lon):
            raise ValueError(f"End longitude {end_lon:.3f} is outside valid range ({min_lon:.3f} to {max_lon:.3f})")
        if not (min_lat <= start_lat <= max_lat):
            raise ValueError(f"Start latitude {start_lat:.3f} is outside valid range ({min_lat:.3f} to {max_lat:.3f})")
        if not (min_lat <= end_lat <= max_lat):
            raise ValueError(f"End latitude {end_lat:.3f} is outside valid range ({min_lat:.3f} to {max_lat:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Coordinate validation error: {e}")
        return False

def run_custom_mission(start_lon, start_lat, end_lon, end_lat, mission_name):
    """Run mission with custom coordinates"""
    
    print("🌙 Custom Coordinates Lunar Pathfinding")
    print("=" * 50)
    print(f"📍 Start: {start_lat:.3f}°N, {start_lon:.3f}°E")
    print(f"📍 End:   {end_lat:.3f}°N, {end_lon:.3f}°E")
    print(f"📝 Mission: {mission_name}")
    print()
    
    # Calculate approximate distance
    import numpy as np
    dx = end_lon - start_lon
    dy = end_lat - start_lat
    distance = np.sqrt(dx*dx + dy*dy) * 111.32  # Rough conversion to km
    print(f"📏 Approximate distance: ~{distance:.2f} km")
    print()
    
    # Initialize orchestrator
    power_csv_path = "synthetic_rover_power_nomode.csv"
    orchestrator = LunarRoverOrchestrator(power_csv_path)
    
    # Mission parameters
    image_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_b_brw_d18.png"
    csv_path = "chandrayaan-2/ch2_ohr_ncp_20210331T2033243734_g_grd_d18.csv"
    
    # Run the mission
    result = orchestrator.run_mission(
        image_path, csv_path, start_lon, start_lat, end_lon, end_lat,
        mission_name
    )
    
    # Show results
    print()
    print("🎯 MISSION COMPLETED!")
    print("=" * 50)
    
    if 'results' in result and 'statistics' in result['results']:
        stats = result['results']['statistics']
        print(f"✅ Status: {result['path_status']}")
        print(f"🧠 Model: {result['model_type']}")
        print(f"⏱️ Processing Time: {result['processing_time']:.2f} seconds")
        print(f"📏 Path Length: {stats.get('path_length_km', 0):.3f} km")
        print(f"🚧 Obstacles: {stats.get('total_obstacles', 0)} total")
        print(f"   - Boulders: {stats.get('boulders', 0)}")
        print(f"   - Landslides: {stats.get('landslides', 0)}")
        
        # Show battery status
        if 'battery_status' in result:
            battery = result['battery_status']
            print(f"🔋 Battery: {battery.get('battery_percent', 0):.1f}%")
            print(f"⚡ Power Efficiency: {battery.get('power_efficiency_score', 0):.2f}")
    
    print()
    print("📁 Check the 'mission_results' folder for:")
    print("   - HTML visualization map")
    print("   - JSON mission report")
    print("   - Detailed statistics")

def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate coordinates
        if not validate_coordinates(args.start_lon, args.start_lat, args.end_lon, args.end_lat):
            return
        
        # Run the mission
        run_custom_mission(args.start_lon, args.start_lat, args.end_lon, args.end_lat, args.mission_name)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Usage examples:")
        print("python custom_coordinates_demo.py --start-lon 41.420 --start-lat -19.720 --end-lon 41.450 --end-lat -19.750")
        print("python custom_coordinates_demo.py --start-lon 41.420 --start-lat -19.720 --end-lon 41.430 --end-lat -19.730 --mission-name 'Short Test'")

if __name__ == "__main__":
    main()
