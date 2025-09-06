#!/usr/bin/env python3
"""
Complete Integration Test for Lunar Rover System
Tests the full flow from orchestrator to frontend
"""

import requests
import json
import time
import subprocess
import sys
from pathlib import Path

def test_api_server_startup():
    """Test if API server can start and respond"""
    print("🧪 Testing API Server Startup...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            return True
        else:
            print(f"❌ API server health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API server: {e}")
        print("   Please start the API server first: python start_integrated_system.py")
        return False

def test_status_endpoint():
    """Test the status endpoint and verify data structure"""
    print("\n🧪 Testing Status Endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/api/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status endpoint working")
            print(f"   - Success: {data.get('success')}")
            
            if data.get('success') and 'status' in data:
                status = data['status']
                print(f"   - Is Running: {status.get('is_running')}")
                print(f"   - Total Missions: {status.get('system_stats', {}).get('total_missions', 0)}")
                print(f"   - Available Observations: {len(status.get('available_observations', []))}")
                
                # Check battery status structure
                if 'battery_status' in status:
                    battery = status['battery_status']
                    required_fields = ['battery_percent', 'solar_input_wm2', 'power_consumption_w', 'can_run_ml', 'reason', 'power_efficiency_score']
                    missing_fields = [field for field in required_fields if field not in battery]
                    if missing_fields:
                        print(f"   ⚠️  Missing battery fields: {missing_fields}")
                    else:
                        print(f"   ✅ Battery status structure correct")
                        print(f"      - Battery: {battery['battery_percent']:.1f}%")
                        print(f"      - ML Enabled: {battery['can_run_ml']}")
                
                return True
            else:
                print(f"   ❌ Status data structure invalid: {data}")
                return False
        else:
            print(f"❌ Status request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status test failed: {e}")
        return False

def test_mission_execution():
    """Test a complete mission execution"""
    print("\n🧪 Testing Mission Execution...")
    
    try:
        # Test mission data
        mission_data = {
            "name": "Integration Test Mission",
            "image_path": "chandrayaan-2/test.png",
            "csv_path": "chandrayaan-2/test.csv",
            "start_lon": 41.406,
            "start_lat": -19.694,
            "end_lon": 41.420,
            "end_lat": -19.700
        }
        
        print(f"   📋 Mission: {mission_data['name']}")
        print(f"   📍 Start: {mission_data['start_lon']}, {mission_data['start_lat']}")
        print(f"   🎯 End: {mission_data['end_lon']}, {mission_data['end_lat']}")
        
        # Run integrated mission (more likely to work without data files)
        response = requests.post(
            "http://localhost:8000/api/run_integrated_mission",
            json=mission_data,
            timeout=60  # Allow more time for processing
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                result = data.get('result', {})
                print(f"✅ Mission executed successfully")
                print(f"   - Mission Name: {result.get('mission_name')}")
                print(f"   - Model Type: {result.get('model_type')}")
                print(f"   - Path Status: {result.get('path_status')}")
                print(f"   - Processing Time: {result.get('processing_time'):.2f}s")
                
                # Check statistics
                stats = result.get('statistics', {})
                print(f"   - Path Found: {stats.get('path_found')}")
                print(f"   - Path Length: {stats.get('path_length_km', 0):.2f} km")
                print(f"   - Total Obstacles: {stats.get('total_obstacles', 0)}")
                print(f"   - Boulders: {stats.get('boulders', 0)}")
                print(f"   - Landslides: {stats.get('landslides', 0)}")
                
                # Check battery status
                battery = result.get('battery_status', {})
                print(f"   - Battery Level: {battery.get('battery_percent', 0):.1f}%")
                print(f"   - ML Processing: {battery.get('can_run_ml', False)}")
                
                # Check if map was generated
                if result.get('folium_map_html'):
                    print(f"   ✅ Interactive map generated: {result['folium_map_html']}")
                else:
                    print(f"   ⚠️  No interactive map generated")
                
                return True
            else:
                print(f"❌ Mission failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Mission request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Mission execution failed: {e}")
        return False

def test_output_files():
    """Test if output files are generated"""
    print("\n🧪 Testing Output File Generation...")
    
    mission_results_dir = Path("mission_results")
    if not mission_results_dir.exists():
        print("❌ Mission results directory not found")
        return False
    
    # Look for recent files
    html_files = list(mission_results_dir.glob("*.html"))
    json_files = list(mission_results_dir.glob("*.json"))
    
    print(f"   📁 Found {len(html_files)} HTML files")
    print(f"   📄 Found {len(json_files)} JSON files")
    
    if html_files:
        latest_html = max(html_files, key=lambda f: f.stat().st_mtime)
        print(f"   ✅ Latest HTML map: {latest_html.name}")
        
        # Check if file is readable
        try:
            with open(latest_html, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'folium' in content.lower() or 'leaflet' in content.lower():
                    print(f"   ✅ HTML map appears to be valid")
                else:
                    print(f"   ⚠️  HTML map may be corrupted")
        except Exception as e:
            print(f"   ❌ Cannot read HTML file: {e}")
    
    if json_files:
        latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
        print(f"   ✅ Latest JSON report: {latest_json.name}")
        
        # Check if JSON is valid
        try:
            with open(latest_json, 'r') as f:
                data = json.load(f)
                print(f"   ✅ JSON report is valid")
                print(f"      - Mission: {data.get('mission_name', 'Unknown')}")
                print(f"      - Model: {data.get('model_type', 'Unknown')}")
        except Exception as e:
            print(f"   ❌ Cannot read JSON file: {e}")
    
    return len(html_files) > 0 and len(json_files) > 0

def test_frontend_compatibility():
    """Test if the API responses are compatible with frontend expectations"""
    print("\n🧪 Testing Frontend Compatibility...")
    
    try:
        # Test status endpoint for frontend compatibility
        response = requests.get("http://localhost:8000/api/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            
            # Check required fields for frontend
            required_status_fields = ['success', 'status']
            if all(field in data for field in required_status_fields):
                print("✅ Status response has required fields")
                
                status = data['status']
                required_status_data = ['is_running', 'system_stats', 'available_observations']
                if all(field in status for field in required_status_data):
                    print("✅ Status data has required fields")
                    
                    # Check system stats
                    stats = status['system_stats']
                    required_stats = ['total_missions', 'ml_missions', 'heuristic_missions', 'successful_paths', 'failed_paths']
                    if all(field in stats for field in required_stats):
                        print("✅ System stats have required fields")
                    else:
                        missing = [f for f in required_stats if f not in stats]
                        print(f"❌ Missing system stats fields: {missing}")
                        return False
                else:
                    missing = [f for f in required_status_data if f not in status]
                    print(f"❌ Missing status data fields: {missing}")
                    return False
            else:
                missing = [f for f in required_status_fields if f not in data]
                print(f"❌ Missing status response fields: {missing}")
                return False
        else:
            print(f"❌ Status request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Frontend compatibility test failed: {e}")
        return False

def main():
    """Run complete integration test"""
    print("🚀 Lunar Rover Complete Integration Test")
    print("=" * 50)
    
    tests = [
        ("API Server Startup", test_api_server_startup),
        ("Status Endpoint", test_status_endpoint),
        ("Mission Execution", test_mission_execution),
        ("Output File Generation", test_output_files),
        ("Frontend Compatibility", test_frontend_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration is working correctly.")
        print("\n✅ The system is ready for frontend use:")
        print("   1. Start frontend: cd frontend && npm run dev")
        print("   2. Open http://localhost:8081")
        print("   3. Navigate to Path Finder page")
        print("   4. Configure and run missions")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
