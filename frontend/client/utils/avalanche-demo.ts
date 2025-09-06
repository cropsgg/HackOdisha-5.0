/**
 * Avalanche Subnet Demo for ISRO Data Transfer System
 * This file demonstrates the functionality of the subnet system
 */

import { 
  healthMonitor, 
  transferManager, 
  encryptionManager,
  type TransferRequest,
  ISRO_SUBNET_CONFIG
} from './avalanche';

// Demo function to showcase subnet operations
export async function runAvalancheDemo() {
  console.log('🚀 Starting Avalanche Subnet Demo for ISRO Stations');
  console.log('=' .repeat(60));

  // 1. Display ISRO Station Configuration
  console.log('\n📡 ISRO Station Configuration:');
  Object.entries(ISRO_SUBNET_CONFIG).forEach(([stationId, config]) => {
    console.log(`\n  ${stationId.toUpperCase()}:`);
    console.log(`    Subnet ID: ${config.subnetId.substring(0, 20)}...`);
    console.log(`    Chain ID: ${config.chainId}`);
    console.log(`    RPC URL: ${config.rpcUrl}`);
    console.log(`    Validators: ${config.validators.length}`);
    console.log(`    Min Stake: ${parseInt(config.minStake) / 1e18} AVAX`);
    console.log(`    Uptime Requirement: ${config.uptimeRequirement * 100}%`);
  });

  // 2. Check Station Health
  console.log('\n🏥 Station Health Status:');
  const healthData = healthMonitor.getHealthData();
  healthData.forEach((health, stationId) => {
    const isHealthy = healthMonitor.isStationHealthy(stationId);
    console.log(`\n  ${stationId.toUpperCase()}:`);
    console.log(`    Status: ${isHealthy ? '✅ Healthy' : '❌ Unhealthy'}`);
    console.log(`    Uptime: ${(health.uptime * 100).toFixed(2)}%`);
    console.log(`    Last Block: ${health.lastBlock}`);
    console.log(`    Validators: ${health.validatorCount}`);
    console.log(`    Consensus Rate: ${(health.consensusRate * 100).toFixed(2)}%`);
    console.log(`    Network Latency: ${health.networkLatency.toFixed(1)}ms`);
    console.log(`    Data Throughput: ${health.dataThroughput.toFixed(1)} MB/s`);
  });

  // 3. Simulate Data Transfer
  console.log('\n📤 Simulating Data Transfer:');
  
  const demoTransfer: TransferRequest = {
    id: 'demo_transfer_001',
    fromStation: 'bangalore',
    toStation: 'delhi',
    dataSize: 2.4 * 1024 * 1024 * 1024, // 2.4 GB
    priority: 'high',
    encryptionType: 'AES-256',
    timestamp: Date.now(),
    checksum: encryptionManager.generateChecksum(new ArrayBuffer(1024))
  };

  console.log(`  Transfer Request: ${demoTransfer.id}`);
  console.log(`  From: ${demoTransfer.fromStation.toUpperCase()}`);
  console.log(`  To: ${demoTransfer.toStation.toUpperCase()}`);
  console.log(`  Data Size: ${(demoTransfer.dataSize / (1024 * 1024 * 1024)).toFixed(1)} GB`);
  console.log(`  Priority: ${demoTransfer.priority.toUpperCase()}`);
  console.log(`  Encryption: ${demoTransfer.encryptionType}`);
  console.log(`  Checksum: ${demoTransfer.checksum}`);

  try {
    console.log('\n  🔄 Initiating transfer...');
    const transferId = await transferManager.initiateTransfer(demoTransfer);
    console.log(`  ✅ Transfer initiated with ID: ${transferId}`);

    // Monitor transfer progress
    let progress = 0;
    const progressInterval = setInterval(() => {
      const status = transferManager.getTransferStatus(transferId);
      if (status) {
        if (status.progress > progress) {
          progress = status.progress;
          console.log(`  📊 Progress: ${progress}%`);
        }
        
        if (status.status === 'completed') {
          console.log('  🎉 Transfer completed successfully!');
          clearInterval(progressInterval);
        } else if (status.status === 'failed') {
          console.log(`  ❌ Transfer failed: ${status.error}`);
          clearInterval(progressInterval);
        }
      }
    }, 1000);

    // Wait for transfer to complete or timeout after 30 seconds
    setTimeout(() => {
      clearInterval(progressInterval);
      const finalStatus = transferManager.getTransferStatus(transferId);
      if (finalStatus && finalStatus.status === 'processing') {
        console.log('  ⏰ Demo timeout - transfer still in progress');
      }
    }, 30000);

  } catch (error) {
    console.error('  ❌ Failed to initiate transfer:', error);
  }

  // 4. Display Active Transfers
  console.log('\n📋 Active Transfers:');
  const activeTransfers = transferManager.getAllTransfers();
  if (activeTransfers.length === 0) {
    console.log('  No active transfers');
  } else {
    activeTransfers.forEach((transfer, index) => {
      console.log(`\n  Transfer ${index + 1}:`);
      console.log(`    ID: ${transfer.requestId}`);
      console.log(`    Status: ${transfer.status}`);
      console.log(`    Progress: ${transfer.progress}%`);
      console.log(`    Current Station: ${transfer.currentStation}`);
      console.log(`    Next Station: ${transfer.nextStation}`);
      console.log(`    Estimated Time: ${transfer.estimatedTime}s`);
      if (transfer.error) {
        console.log(`    Error: ${transfer.error}`);
      }
    });
  }

  // 5. Encryption Demo
  console.log('\n🔐 Encryption Demo:');
  const sampleData = new TextEncoder().encode('ISRO Lunar Mission Data - Top Secret');
  const sampleBuffer = sampleData.buffer;
  
  console.log(`  Original Data: ${sampleData}`);
  console.log(`  Data Size: ${sampleBuffer.byteLength} bytes`);
  
  try {
    const encryptedData = await encryptionManager.encryptData(sampleBuffer, 'AES-256');
    console.log(`  Encrypted Size: ${encryptedData.byteLength} bytes`);
    
    const decryptedData = await encryptionManager.decryptData(encryptedData, 'AES-256');
    const decryptedText = new TextDecoder().decode(decryptedData);
    console.log(`  Decrypted Data: ${decryptedText}`);
    
    const checksum = encryptionManager.generateChecksum(sampleBuffer);
    console.log(`  Checksum: ${checksum}`);
    
    console.log('  ✅ Encryption/Decryption successful');
  } catch (error) {
    console.error('  ❌ Encryption demo failed:', error);
  }

  console.log('\n' + '=' .repeat(60));
  console.log('🎯 Avalanche Subnet Demo Complete!');
  console.log('\nKey Features Demonstrated:');
  console.log('• ISRO Station Configuration & Health Monitoring');
  console.log('• Secure Data Transfer with Priority Levels');
  console.log('• Real-time Progress Tracking');
  console.log('• Military-grade Encryption (AES-256)');
  console.log('• Private Subnet Network Isolation');
  console.log('• Validator Consensus & Uptime Monitoring');
}

// Export demo function for use in components
export { runAvalancheDemo };
