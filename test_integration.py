#!/usr/bin/env python3
"""
Test script for the integrated lunar rover system
Verifies that the API server can start and respond to requests
"""

import requests
import time
import subprocess
import sys
import json
from pathlib import Path

def test_api_health():
    """Test if the API server is responding"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API health check passed")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API health check failed: {e}")
        return False

def test_api_status():
    """Test the status endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ API status endpoint working")
                print(f"   - Connected: {data.get('status', {}).get('is_running', False)}")
                print(f"   - Total missions: {data.get('status', {}).get('system_stats', {}).get('total_missions', 0)}")
                return True
            else:
                print(f"❌ API status failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ API status request failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API status request failed: {e}")
        return False

def test_observations():
    """Test the observations endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/observations", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                observations = data.get("observations", [])
                print(f"✅ Observations endpoint working - {len(observations)} observations available")
                return True
            else:
                print(f"❌ Observations failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Observations request failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Observations request failed: {e}")
        return False

def test_mission_endpoint():
    """Test a simple mission request"""
    try:
        mission_data = {
            "name": "Test Mission",
            "image_path": "chandrayaan-2/test.png",
            "csv_path": "chandrayaan-2/test.csv",
            "start_lon": 41.406,
            "start_lat": -19.694,
            "end_lon": 41.420,
            "end_lat": -19.700
        }
        
        response = requests.post(
            "http://localhost:8000/api/run_integrated_mission",
            json=mission_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Mission endpoint working")
                result = data.get("result", {})
                print(f"   - Mission: {result.get('mission_name', 'Unknown')}")
                print(f"   - Model: {result.get('model_type', 'Unknown')}")
                print(f"   - Path found: {result.get('statistics', {}).get('path_found', False)}")
                return True
            else:
                print(f"❌ Mission failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Mission request failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Mission request failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🧪 Testing Lunar Rover Integration")
    print("=" * 40)
    
    # Check if API server is running
    print("1. Testing API health...")
    if not test_api_health():
        print("\n❌ API server is not running!")
        print("   Please start the API server first:")
        print("   python start_integrated_system.py")
        return False
    
    print("\n2. Testing API status...")
    if not test_api_status():
        return False
    
    print("\n3. Testing observations...")
    if not test_observations():
        return False
    
    print("\n4. Testing mission endpoint...")
    if not test_mission_endpoint():
        return False
    
    print("\n" + "=" * 40)
    print("🎉 All integration tests passed!")
    print("✅ The system is ready for frontend integration")
    print("\n📋 Next steps:")
    print("1. Start the frontend: cd frontend && npm run dev")
    print("2. Open http://localhost:8081 in your browser")
    print("3. Navigate to the Path Finder page")
    print("4. Configure and run a mission")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
