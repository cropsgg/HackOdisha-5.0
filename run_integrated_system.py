#!/usr/bin/env python3
"""
Complete Integrated Lunar Pathfinding System Runner
Demonstrates the full pipeline from detection to visualization
"""

import os
import sys
from pathlib import Path
import json
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "Landslide" / "src" / "coordinate_utils"))

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'numpy', 'cv2', 'pandas', 'matplotlib', 'folium', 
        'networkx', 'torch', 'transformers'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements_integrated.txt")
        return False
    
    print("✅ All required packages are available")
    return True

def check_chandrayaan_data():
    """Check if Chandrayaan-2 data is available"""
    data_dir = Path("chandrayaan-2")
    if not data_dir.exists():
        print("❌ Chandrayaan-2 data directory not found")
        print("   Expected: chandrayaan-2/ directory with satellite data")
        return False
    
    # Check for data files
    img_files = list(data_dir.glob("*.img"))
    csv_files = list(data_dir.glob("*.csv"))
    png_files = list(data_dir.glob("*.png"))
    
    if not img_files or not csv_files or not png_files:
        print("❌ Incomplete Chandrayaan-2 data")
        print(f"   Found: {len(img_files)} .img, {len(csv_files)} .csv, {len(png_files)} .png files")
        return False
    
    print(f"✅ Chandrayaan-2 data available: {len(img_files)} observations")
    return True

def run_system_demo():
    """Run the complete integrated system demonstration"""
    
    print("🚀 Starting Integrated Lunar Pathfinding System")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check data
    if not check_chandrayaan_data():
        return False
    
    try:
        # Import the integrated system
        from integrated_pathfinding_system import IntegratedPathfindingSystem
        from coordinate_utils.selenographic_converter import format_selenographic_coords
        
        print("\n🔧 Initializing Integrated Pathfinding System...")
        system = IntegratedPathfindingSystem()
        
        # Get available observations
        observations = system.data_loader.get_available_observations()
        if not observations:
            print("❌ No Chandrayaan-2 observations available")
            return False
        
        print(f"📡 Found {len(observations)} Chandrayaan-2 observations")
        
        # Use first observation for demonstration
        obs_id = observations[0]
        print(f"🎯 Using observation: {obs_id}")
        
        # Load coordinate data to get valid ranges
        coordinates = system.data_loader.load_coordinate_data(obs_id)
        if coordinates is None:
            print("❌ Failed to load coordinate data")
            return False
        
        # Get coordinate bounds
        min_lon, max_lon = coordinates['Longitude'].min(), coordinates['Longitude'].max()
        min_lat, max_lat = coordinates['Latitude'].min(), coordinates['Latitude'].max()
        
        print(f"\n🗺️  Coordinate bounds:")
        print(f"   Longitude: {min_lon:.6f}° to {max_lon:.6f}°")
        print(f"   Latitude: {min_lat:.6f}° to {max_lat:.6f}°")
        
        # Define start and end points
        start_lon = min_lon + (max_lon - min_lon) * 0.25  # 25% from left
        start_lat = min_lat + (max_lat - min_lat) * 0.25  # 25% from bottom
        
        end_lon = min_lon + (max_lon - min_lon) * 0.75    # 75% from left
        end_lat = min_lat + (max_lat - min_lat) * 0.75    # 75% from bottom
        
        print(f"\n🎯 Path Planning:")
        print(f"   Start: {format_selenographic_coords(start_lon, start_lat)}")
        print(f"   End: {format_selenographic_coords(end_lon, end_lat)}")
        
        # Process complete pathfinding
        print(f"\n🔍 Processing pathfinding...")
        start_time = time.time()
        
        results = system.process_complete_pathfinding(
            obs_id, start_lon, start_lat, end_lon, end_lat
        )
        
        processing_time = time.time() - start_time
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return False
        
        # Display results
        print(f"\n✅ Pathfinding Results (processed in {processing_time:.2f}s):")
        print(f"   📊 Total obstacles: {results['statistics']['total_obstacles']}")
        print(f"   🪨 Boulders: {results['statistics']['boulders']}")
        print(f"   🏔️  Landslides: {results['statistics']['landslides']}")
        print(f"   🛤️  Path found: {'Yes' if results['statistics']['path_found'] else 'No'}")
        print(f"   📏 Path length: {results['statistics']['path_length_km']:.2f} km")
        print(f"   🛡️  Safety: {results['statistics']['safety_percentage']:.1f}%")
        print(f"   📍 Path points: {results['statistics']['path_points']}")
        
        # Save results
        output_dir = Path("pathfinding_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save interactive map
        map_path = output_dir / f"integrated_lunar_pathfinding_{obs_id}.html"
        results['folium_map'].save(str(map_path))
        print(f"\n🗺️  Interactive map saved: {map_path}")
        
        # Save detailed results
        json_path = output_dir / f"integrated_results_{obs_id}.json"
        
        # Prepare JSON-serializable results
        json_results = {
            "observation_id": results['observation_id'],
            "statistics": results['statistics'],
            "coordinate_bounds": results['coordinate_bounds'],
            "processing_time_seconds": processing_time,
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
        print(f"📄 Detailed results saved: {json_path}")
        
        # Create summary report
        summary_path = output_dir / f"summary_report_{obs_id}.txt"
        with open(summary_path, 'w') as f:
            f.write("Integrated Lunar Pathfinding System - Summary Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Observation ID: {obs_id}\n")
            f.write(f"Processing Time: {processing_time:.2f} seconds\n\n")
            f.write("Path Planning:\n")
            f.write(f"  Start: {format_selenographic_coords(start_lon, start_lat)}\n")
            f.write(f"  End: {format_selenographic_coords(end_lon, end_lat)}\n\n")
            f.write("Detection Results:\n")
            f.write(f"  Total Obstacles: {results['statistics']['total_obstacles']}\n")
            f.write(f"  Boulders: {results['statistics']['boulders']}\n")
            f.write(f"  Landslides: {results['statistics']['landslides']}\n\n")
            f.write("Path Results:\n")
            f.write(f"  Path Found: {'Yes' if results['statistics']['path_found'] else 'No'}\n")
            f.write(f"  Path Length: {results['statistics']['path_length_km']:.2f} km\n")
            f.write(f"  Safety: {results['statistics']['safety_percentage']:.1f}%\n")
            f.write(f"  Path Points: {results['statistics']['path_points']}\n\n")
            f.write("Output Files:\n")
            f.write(f"  Interactive Map: {map_path}\n")
            f.write(f"  Detailed Results: {json_path}\n")
            f.write(f"  Summary Report: {summary_path}\n")
        
        print(f"📋 Summary report saved: {summary_path}")
        
        print(f"\n🎉 Integrated System Demo Complete!")
        print(f"   📁 All outputs saved to: {output_dir}")
        print(f"   🌐 Open {map_path} in your web browser to see the interactive map")
        print(f"   📊 The map shows:")
        print(f"      • 🟢 Start point (green marker)")
        print(f"      • 🔴 Goal point (red marker)")
        print(f"      • 🔵 Boulders (blue circles)")
        print(f"      • 🔴 Landslides (red warning markers)")
        print(f"      • 🟢 Optimal path (green line)")
        print(f"      • 📊 Statistics (purple info marker)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all required packages are installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_system_info():
    """Show system information and capabilities"""
    print("🌙 Integrated Lunar Pathfinding System")
    print("=" * 50)
    print("This system combines:")
    print("  🔍 Boulder Detection (Mask2Former instance segmentation)")
    print("  🏔️  Landslide Detection (physics-based criteria)")
    print("  🗺️  Optimal Pathfinding (A* algorithm with risk weights)")
    print("  📊 Interactive Visualization (Folium maps)")
    print("  🛰️  Chandrayaan-2 Integration (real satellite data)")
    print()
    print("Key Features:")
    print("  • Dual obstacle detection for comprehensive safety")
    print("  • Risk-aware pathfinding with weighted cost functions")
    print("  • Selenographic coordinate system for lunar navigation")
    print("  • Interactive web-based visualization")
    print("  • Real Chandrayaan-2 satellite data integration")
    print()

if __name__ == "__main__":
    show_system_info()
    
    success = run_system_demo()
    
    if success:
        print("\n" + "=" * 60)
        print("🎯 System demonstration completed successfully!")
        print("   Check the 'pathfinding_results' folder for all outputs.")
        print("   Open the HTML file in your web browser to see the interactive map.")
    else:
        print("\n" + "=" * 60)
        print("❌ System demonstration failed.")
        print("   Please check the error messages above and ensure:")
        print("   • All dependencies are installed")
        print("   • Chandrayaan-2 data is available")
        print("   • Python environment is properly configured")
