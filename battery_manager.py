#!/usr/bin/env python3
"""
Battery Management System for Lunar Rover
Monitors battery levels and determines if ML processing is feasible
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BatteryManager:
    """Manages rover battery levels and power consumption decisions"""
    
    def __init__(self, csv_path: str):
        """Initialize battery manager with power data"""
        self.csv_path = csv_path
        self.power_data = None
        self.current_battery = 0.0
        self.current_solar_input = 0.0
        self.current_power_consumption = 0.0
        self.load_power_data()
    
    def load_power_data(self):
        """Load power data from CSV file"""
        try:
            self.power_data = pd.read_csv(self.csv_path)
            self.power_data['timestamp'] = pd.to_datetime(self.power_data['timestamp'])
            logger.info(f"Loaded {len(self.power_data)} power data points")
        except Exception as e:
            logger.error(f"Error loading power data: {e}")
            self.power_data = None
    
    def get_random_battery_status(self) -> Dict[str, float]:
        """Get a random battery status from the dataset"""
        if self.power_data is None or len(self.power_data) == 0:
            # Fallback to synthetic data
            return {
                'battery_percent': random.uniform(20, 80),
                'solar_input_wm2': random.uniform(200, 400),
                'power_consumption_w': random.uniform(80, 150)
            }
        
        # Select random row
        random_idx = random.randint(0, len(self.power_data) - 1)
        row = self.power_data.iloc[random_idx]
        
        return {
            'battery_percent': float(row['battery_percent']),
            'solar_input_wm2': float(row['solar_input_wm2']),
            'power_consumption_w': float(row['power_consumption_w'])
        }
    
    def can_run_ml_processing(self, battery_status: Dict[str, float]) -> Tuple[bool, str]:
        """
        Determine if ML processing is feasible based on battery status
        
        Returns:
            Tuple[bool, str]: (can_run_ml, reason)
        """
        battery_percent = battery_status['battery_percent']
        solar_input = battery_status['solar_input_wm2']
        power_consumption = battery_status['power_consumption_w']
        
        # ML processing requires significant power
        ml_power_requirement = 200  # Watts for ML processing
        current_power_available = solar_input - power_consumption
        
        # Decision criteria
        if battery_percent < 30:
            return False, f"Battery too low: {battery_percent:.1f}% < 30%"
        
        if current_power_available < ml_power_requirement:
            return False, f"Insufficient power: {current_power_available:.1f}W < {ml_power_requirement}W required"
        
        if battery_percent < 50 and current_power_available < ml_power_requirement * 1.5:
            return False, f"Low battery with insufficient power buffer: {battery_percent:.1f}% battery, {current_power_available:.1f}W available"
        
        return True, f"ML processing feasible: {battery_percent:.1f}% battery, {current_power_available:.1f}W available"
    
    def estimate_processing_time(self, battery_status: Dict[str, float], use_ml: bool) -> float:
        """Estimate processing time based on battery and model type"""
        base_time = 2.0  # Base processing time in seconds
        
        if use_ml:
            # ML processing takes longer
            return base_time * 3.0
        else:
            # Heuristic processing is faster
            return base_time * 0.5
    
    def get_power_efficiency_score(self, battery_status: Dict[str, float]) -> float:
        """Calculate power efficiency score (0-1)"""
        battery_percent = battery_status['battery_percent']
        solar_input = battery_status['solar_input_wm2']
        power_consumption = battery_status['power_consumption_w']
        
        # Normalize battery level (0-1)
        battery_score = battery_percent / 100.0
        
        # Normalize power efficiency (0-1)
        power_efficiency = max(0, (solar_input - power_consumption) / solar_input) if solar_input > 0 else 0
        
        # Combined score
        return (battery_score * 0.6 + power_efficiency * 0.4)
    
    def log_battery_status(self, battery_status: Dict[str, float], decision: Tuple[bool, str]):
        """Log current battery status and decision"""
        can_run_ml, reason = decision
        efficiency_score = self.get_power_efficiency_score(battery_status)
        
        logger.info("=" * 50)
        logger.info("🔋 BATTERY STATUS REPORT")
        logger.info("=" * 50)
        logger.info(f"Battery Level: {battery_status['battery_percent']:.1f}%")
        logger.info(f"Solar Input: {battery_status['solar_input_wm2']:.1f} W/m²")
        logger.info(f"Power Consumption: {battery_status['power_consumption_w']:.1f} W")
        logger.info(f"Power Efficiency Score: {efficiency_score:.2f}")
        logger.info(f"ML Processing: {'✅ ENABLED' if can_run_ml else '❌ DISABLED'}")
        logger.info(f"Reason: {reason}")
        logger.info("=" * 50)
        
        return {
            'battery_percent': battery_status['battery_percent'],
            'solar_input_wm2': battery_status['solar_input_wm2'],
            'power_consumption_w': battery_status['power_consumption_w'],
            'power_efficiency_score': efficiency_score,
            'can_run_ml': can_run_ml,
            'reason': reason
        }
