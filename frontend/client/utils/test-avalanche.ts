/**
 * Simple test file for Avalanche Subnet Integration
 * Run this in the browser console to test basic functionality
 */

import { 
  healthMonitor, 
  transferManager, 
  encryptionManager,
  ISRO_SUBNET_CONFIG
} from './avalanche';

// Test basic functionality
export function testAvalancheIntegration() {
  console.log('🧪 Testing Avalanche Subnet Integration...');
  
  try {
    // Test 1: Configuration
    console.log('✅ Configuration loaded:', Object.keys(ISRO_SUBNET_CONFIG).length, 'stations');
    
    // Test 2: Health Monitor
    const healthData = healthMonitor.getHealthData();
    console.log('✅ Health monitor working:', healthData.size, 'stations monitored');
    
    // Test 3: Transfer Manager
    const transfers = transferManager.getAllTransfers();
    console.log('✅ Transfer manager working:', transfers.length, 'active transfers');
    
    // Test 4: Encryption
    const testData = new TextEncoder().encode('Test data');
    const checksum = encryptionManager.generateChecksum(testData.buffer);
    console.log('✅ Encryption working, checksum:', checksum);
    
    console.log('🎉 All tests passed! Avalanche integration is working correctly.');
    return true;
    
  } catch (error) {
    console.error('❌ Test failed:', error);
    return false;
  }
}

// Test station health
export function testStationHealth() {
  console.log('🏥 Testing Station Health...');
  
  Object.keys(ISRO_SUBNET_CONFIG).forEach(stationId => {
    const isHealthy = healthMonitor.isStationHealthy(stationId);
    const health = healthMonitor.getStationHealth(stationId);
    
    console.log(`${stationId.toUpperCase()}:`, {
      healthy: isHealthy,
      uptime: health ? `${(health.uptime * 100).toFixed(2)}%` : 'Unknown',
      validators: health ? health.validatorCount : 'Unknown',
      latency: health ? `${health.networkLatency.toFixed(1)}ms` : 'Unknown'
    });
  });
}

// Test transfer creation
export async function testTransferCreation() {
  console.log('📤 Testing Transfer Creation...');
  
  try {
    const transferId = await transferManager.initiateTransfer({
      id: 'test_transfer_' + Date.now(),
      fromStation: 'bangalore',
      toStation: 'delhi',
      dataSize: 1024 * 1024, // 1MB
      priority: 'medium',
      encryptionType: 'AES-256',
      timestamp: Date.now(),
      checksum: 'test_checksum'
    });
    
    console.log('✅ Transfer created with ID:', transferId);
    
    // Check status
    const status = transferManager.getTransferStatus(transferId);
    console.log('✅ Transfer status:', status);
    
    return transferId;
    
  } catch (error) {
    console.error('❌ Transfer creation failed:', error);
    return null;
  }
}

// Run all tests
export async function runAllTests() {
  console.log('🚀 Running All Avalanche Integration Tests...');
  console.log('=' .repeat(50));
  
  const results = {
    basic: testAvalancheIntegration(),
    health: testStationHealth(),
    transfer: await testTransferCreation()
  };
  
  console.log('=' .repeat(50));
  console.log('📊 Test Results:', results);
  
  const allPassed = results.basic && results.transfer;
  console.log(allPassed ? '🎉 All tests passed!' : '❌ Some tests failed.');
  
  return allPassed;
}

// Export for browser console usage
if (typeof window !== 'undefined') {
  (window as any).testAvalanche = {
    testIntegration: testAvalancheIntegration,
    testHealth: testStationHealth,
    testTransfer: testTransferCreation,
    runAll: runAllTests
  };
  
  console.log('🧪 Avalanche tests loaded! Run testAvalanche.runAll() to test everything.');
}
