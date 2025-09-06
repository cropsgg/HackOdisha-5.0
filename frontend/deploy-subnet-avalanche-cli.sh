#!/bin/bash

# ISRO Avalanche Subnet Deployment Script
# This script deploys the subnet using Avalanche CLI

set -e

echo "🚀 Deploying ISRO Subnet on Avalanche Mainnet..."
echo "=================================================="

# Check if Avalanche CLI is installed
if ! command -v avalanche &> /dev/null; then
    echo "❌ Avalanche CLI not found. Please install it first."
    echo "Installation: https://docs.avax.network/quickstart/upgrade-your-avalanchego-installation"
    exit 1
fi

# Check if configuration file exists
if [ ! -f "avalanche-subnet-config.json" ]; then
    echo "❌ Configuration file not found. Run the Node.js script first."
    exit 1
fi

# Load configuration
CONFIG=$(cat avalanche-subnet-config.json)
ACCOUNTS=$(echo $CONFIG | jq -r '.accounts')

echo "📋 Loaded configuration:"
echo "  Network: Avalanche Mainnet"
echo "  Validators: $(echo $ACCOUNTS | jq length)"

# Deploy subnet
echo ""
echo "🔧 Deploying subnet..."

# Create subnet configuration file
cat > subnet-config.json << EOF
{
  "subnetName": "ISRO Lunar Mission Subnet",
  "projectDir": "./isro-subnet",
  "tokenName": "ISRO",
  "tokenSymbol": "ISRO",
  "vm": "subnet-evm",
  "genesis": {
    "config": {
      "chainId": 12345,
      "homesteadBlock": 0,
      "eip150Block": 0,
      "eip150Hash": "0x2086799aeebeae135c246c65021c82b4e15a2c451340993aacfd2751886514f0",
      "eip155Block": 0,
      "eip158Block": 0,
      "byzantiumBlock": 0,
      "constantinopleBlock": 0,
      "petersburgBlock": 0,
      "istanbulBlock": 0,
      "muirGlacierBlock": 0,
      "subnetEVMTimestamp": 0,
      "feeConfig": {
        "gasLimit": 20000000,
        "minBaseFee": 1000000000,
        "targetGas": 100000000,
        "baseFeeChangeDenominator": 1,
        "minBlockGasCost": 0,
        "maxBlockGasCost": 10000000,
        "targetBlockRate": 2,
        "blockGasCostStep": 500000
      }
    },
    "alloc": {},
    "nonce": "0x0",
    "timestamp": "0x0",
    "extraData": "0x00",
    "gasLimit": "0x1312D00",
    "difficulty": "0x0",
    "mixHash": "0x0000000000000000000000000000000000000000000000000000000000000000",
    "coinbase": "0x0000000000000000000000000000000000000000",
    "number": "0x0",
    "gasUsed": "0x0",
    "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000000"
  }
}
EOF

echo "✅ Subnet configuration created"

# Deploy subnet
echo ""
echo "🚀 Deploying subnet to Avalanche Mainnet..."
avalanche subnet deploy mainnet

echo ""
echo "✅ Subnet deployed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Add validators to the subnet"
echo "2. Update frontend configuration with new subnet ID"
echo "3. Test the subnet functionality"
echo ""
echo "📋 Subnet information saved to:"
echo "  - subnet-config.json"
echo "  - avalanche-subnet-config.json"
