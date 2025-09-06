#!/usr/bin/env python3
"""
Start Integrated Lunar Rover System
Runs both the Python API server and provides instructions for the frontend
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI dependencies found")
        return True
    except ImportError:
        print("❌ FastAPI dependencies not found")
        print("Installing required dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_api.txt"], check=True)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False

def start_api_server():
    """Start the Python API server"""
    print("🚀 Starting Lunar Rover Orchestrator API Server...")
    
    # Check if orchestrator files exist
    if not Path("lunar_rover_orchestrator.py").exists():
        print("❌ lunar_rover_orchestrator.py not found")
        return False
    
    if not Path("orchestrator_api_server.py").exists():
        print("❌ orchestrator_api_server.py not found")
        return False
    
    try:
        # Start the API server
        process = subprocess.Popen([
            sys.executable, "orchestrator_api_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ API server started successfully")
        print("📡 API Server running on: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        
        return process
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def main():
    """Main function"""
    print("🌙 Lunar Rover Integrated System Startup")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Cannot start system without dependencies")
        return
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        print("❌ Failed to start API server")
        return
    
    print("\n" + "=" * 50)
    print("🎯 SYSTEM READY!")
    print("=" * 50)
    print("📡 Python API Server: http://localhost:8000")
    print("🌐 Frontend Server: http://localhost:8081")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("\n📋 Next Steps:")
    print("1. Open a new terminal")
    print("2. Navigate to the frontend directory: cd frontend")
    print("3. Start the frontend: npm run dev")
    print("4. Open http://localhost:8081 in your browser")
    print("\n🛑 Press Ctrl+C to stop the API server")
    
    try:
        # Keep the API server running
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down API server...")
        api_process.terminate()
        print("✅ API server stopped")

if __name__ == "__main__":
    main()
