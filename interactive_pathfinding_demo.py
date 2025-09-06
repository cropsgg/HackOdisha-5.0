#!/usr/bin/env python3
"""
Interactive Pathfinding Demo
Allows users to input custom start and end coordinates for lunar pathfinding
"""

import sys
from pathlib import Path
from lunar_rover_orchestrator import LunarRoverOrchestrator
import pandas as pd

def get_user_coordinates():
    """Get start and end coordinates from user input"""
    
    print("🌙 Interactive Lunar Pathfinding Demo")
    print("=" * 50)
    print("Enter coordinates for your lunar mission:")
    print()
    
    # Load coordinate data to show valid ranges
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
        
    except Exception as e:
        print("⚠️ Could not load coordinate ranges, using defaults")
        min_lon, max_lon = 41.4, 41.5
        min_lat, max_lat = -19.8, -19.6
    
    # Get start coordinates
    print("🚀 START POSITION:")
    while True:
        try:
            start_lon = float(input(f"   Enter start longitude ({min_lon:.3f} to {max_lon:.3f}): "))
            if min_lon <= start_lon <= max_lon:
                break
            else:
                print(f"   ❌ Longitude must be between {min_lon:.3f} and {max_lon:.3f}")
        except ValueError:
            print("   ❌ Please enter a valid number")
    
    while True:
        try:
            start_lat = float(input(f"   Enter start latitude ({min_lat:.3f} to {max_lat:.3f}): "))
            if min_lat <= start_lat <= max_lat:
                break
            else:
                print(f"   ❌ Latitude must be between {min_lat:.3f} and {max_lat:.3f}")
        except ValueError:
            print("   ❌ Please enter a valid number")
    
    print()
    
    # Get end coordinates
    print("🎯 END POSITION:")
    while True:
        try:
            end_lon = float(input(f"   Enter end longitude ({min_lon:.3f} to {max_lon:.3f}): "))
            if min_lon <= end_lon <= max_lon:
                break
            else:
                print(f"   ❌ Longitude must be between {min_lon:.3f} and {max_lon:.3f}")
        except ValueError:
            print("   ❌ Please enter a valid number")
    
    while True:
        try:
            end_lat = float(input(f"   Enter end latitude ({min_lat:.3f} to {max_lat:.3f}): "))
            if min_lat <= end_lat <= max_lat:
                break
            else:
                print(f"   ❌ Latitude must be between {min_lat:.3f} and {max_lat:.3f}")
        except ValueError:
            print("   ❌ Please enter a valid number")
    
    return start_lon, start_lat, end_lon, end_lat

def get_mission_name():
    """Get mission name from user"""
    print()
    mission_name = input("📝 Enter mission name (or press Enter for 'Custom Mission'): ").strip()
    if not mission_name:
        mission_name = "Custom Mission"
    return mission_name

def show_coordinate_preview(start_lon, start_lat, end_lon, end_lat):
    """Show a preview of the selected coordinates"""
    print()
    print("📍 COORDINATE PREVIEW:")
    print("=" * 30)
    print(f"Start: {start_lat:.3f}°N, {start_lon:.3f}°E")
    print(f"End:   {end_lat:.3f}°N, {end_lon:.3f}°E")
    
    # Calculate approximate distance
    import numpy as np
    dx = end_lon - start_lon
    dy = end_lat - start_lat
    distance = np.sqrt(dx*dx + dy*dy) * 111.32  # Rough conversion to km
    print(f"Distance: ~{distance:.2f} km")
    print()

def run_interactive_mission():
    """Run the interactive pathfinding mission"""
    
    # Get user input
    start_lon, start_lat, end_lon, end_lat = get_user_coordinates()
    mission_name = get_mission_name()
    
    # Show preview
    show_coordinate_preview(start_lon, start_lat, end_lon, end_lat)
    
    # Confirm mission
    confirm = input("🚀 Proceed with this mission? (y/n): ").lower().strip()
    if confirm not in ['y', 'yes']:
        print("❌ Mission cancelled.")
        return
    
    print()
    print("🚀 Starting mission...")
    print("=" * 50)
    
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

def show_example_coordinates():
    """Show some example coordinates for reference"""
    print()
    print("💡 EXAMPLE COORDINATES:")
    print("=" * 30)
    print("Short distance examples:")
    print("  Start: 41.420°E, -19.720°N")
    print("  End:   41.430°E, -19.730°N")
    print()
    print("Medium distance examples:")
    print("  Start: 41.420°E, -19.720°N")
    print("  End:   41.450°E, -19.750°N")
    print()
    print("Long distance examples:")
    print("  Start: 41.420°E, -19.720°N")
    print("  End:   41.500°E, -19.800°N")
    print()

def main():
    """Main function"""
    try:
        print("🌙 Welcome to the Interactive Lunar Pathfinding System!")
        print()
        
        # Show example coordinates
        show_example_coordinates()
        
        # Ask if user wants to see examples or proceed
        choice = input("Would you like to see more examples or proceed with custom coordinates? (e/proceed): ").lower().strip()
        
        if choice == 'e':
            show_example_coordinates()
        
        # Run the interactive mission
        run_interactive_mission()
        
    except KeyboardInterrupt:
        print("\n❌ Mission cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check that all required files are present.")

if __name__ == "__main__":
    main()
