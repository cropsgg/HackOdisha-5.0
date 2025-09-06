#!/usr/bin/env node

/**
 * Check AVAX Balance on Avalanche Fuji Testnet
 */

import { ethers } from 'ethers';
import fs from 'fs';

// Load configuration
const config = JSON.parse(fs.readFileSync('avalanche-subnet-config.json', 'utf8'));

async function checkBalance() {
  console.log("🔍 Checking AVAX Balance on Fuji Testnet...");
  console.log("=" .repeat(50));

  // Fuji testnet configuration
  const fujiRpcUrl = "https://api.avax-test.network/ext/bc/C/rpc";
  const provider = new ethers.JsonRpcProvider(fujiRpcUrl);

  console.log(`Network: Avalanche Fuji Testnet`);
  console.log(`RPC URL: ${fujiRpcUrl}`);
  console.log(`Chain ID: 43113`);
  console.log("");

  // Check balance for all generated accounts
  for (let i = 0; i < config.accounts.length; i++) {
    const account = config.accounts[i];
    const wallet = new ethers.Wallet(account.privateKey, provider);
    
    try {
      const balance = await provider.getBalance(wallet.address);
      const balanceInAvax = ethers.formatEther(balance);
      
      console.log(`Account ${i + 1}: ${account.address}`);
      console.log(`Balance: ${balanceInAvax} AVAX`);
      
      if (parseFloat(balanceInAvax) < 1) {
        console.log(`⚠️  Low balance! You need at least 1 AVAX for deployment.`);
        console.log(`💡 Get testnet AVAX from: https://faucet.avax.network/`);
      } else {
        console.log(`✅ Sufficient balance for deployment`);
      }
      console.log("");
      
    } catch (error) {
      console.error(`❌ Error checking balance for account ${i + 1}:`, error.message);
    }
  }

  // Check if we can connect to the network
  try {
    const network = await provider.getNetwork();
    console.log(`✅ Successfully connected to Fuji testnet`);
    console.log(`Network: ${network.name} (Chain ID: ${network.chainId})`);
  } catch (error) {
    console.error(`❌ Failed to connect to Fuji testnet:`, error.message);
  }

  console.log("\n📋 Next Steps:");
  console.log("1. If balance is low, get testnet AVAX from the faucet");
  console.log("2. Run the deployment script: node deploy-contracts-testnet.js");
  console.log("3. Verify contracts on Fuji testnet explorer");
}

checkBalance().catch(console.error);
