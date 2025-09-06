import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Shield,
  Network, 
  Server, 
  Building2, 
  Rocket, 
  Hash, 
  Key, 
  Users, 
  Eye, 
  EyeOff, 
  Play, 
  Square, 
  X, 
  AlertTriangle, 
  TrendingUp, 
  BarChart3,
  FileText,
  Smartphone,
  Globe,
  CheckCircle
} from 'lucide-react';
import { useAvalancheSubnet } from '@/hooks/useAvalancheSubnet';
import { DEPLOYED_CONTRACTS } from '@/utils/avalanche';

const DataTransfer: React.FC = () => {
  const {
    stationHealth,
    selectedStation,
    activeTransfers,
    transferProgress,
    isTransferring,
    showSensitiveData,
    startTransfer,
    stopTransfer,
    cancelTransfer,
    isStationHealthy,
    getTransferRoute,
    formatDataSize,
    formatTime,
    stations,
    setSelectedStation,
    setShowSensitiveData
  } = useAvalancheSubnet();

  const [transferForm, setTransferForm] = useState({
    fromStation: 'bangalore',
    toStation: 'chennai',
    dataSize: 1024 * 1024 * 100, // 100MB
    priority: 'medium' as const,
    encryptionType: 'AES-256' as const
  });

  const handleTransfer = () => {
    startTransfer(
      transferForm.fromStation,
      transferForm.toStation,
      transferForm.dataSize,
      transferForm.priority
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <Shield className="w-12 h-12 text-purple-400 mr-4" />
            <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Avalanche Subnet Data Transfer
            </h1>
          </div>
          <p className="text-xl text-purple-200 max-w-4xl mx-auto">
            Secure, immutable data transmission through ISRO's private Avalanche subnet network.
            Military-grade encryption ensures data remains confidential within the closed network.
          </p>
          
          {/* Deployment Status */}
          <div className="mt-6 flex justify-center">
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-green-900/30 border border-green-500/30">
              <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></div>
              <span className="text-green-400 text-sm font-medium">
                Subnet Deployed on Avalanche Mainnet
              </span>
            </div>
          </div>

          {/* Contract Status */}
          <div className="mt-4 flex justify-center space-x-4">
            <div className="inline-flex items-center px-3 py-1 rounded-full bg-blue-900/30 border border-blue-500/30">
              <CheckCircle className="w-4 h-4 text-blue-400 mr-2" />
              <span className="text-blue-400 text-xs">Smart Contracts Deployed</span>
            </div>
            <div className="inline-flex items-center px-3 py-1 rounded-full bg-purple-900/30 border border-purple-500/30">
              <CheckCircle className="w-4 h-4 text-purple-400 mr-2" />
              <span className="text-purple-400 text-xs">4 Validators Active</span>
            </div>
          </div>
        </div>

        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 bg-purple-900/30 border border-purple-500/30">
            <TabsTrigger value="overview" className="data-[state=active]:bg-purple-600">Overview</TabsTrigger>
            <TabsTrigger value="transfer" className="data-[state=active]:bg-purple-600">Data Transfer</TabsTrigger>
            <TabsTrigger value="contracts" className="data-[state=active]:bg-purple-600">Smart Contracts</TabsTrigger>
            <TabsTrigger value="monitoring" className="data-[state=active]:bg-purple-600">Monitoring</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* ISRO Stations Overview */}
            <Card className="bg-purple-900/20 border-purple-500/30">
            <CardHeader>
                <CardTitle className="flex items-center text-purple-200">
                  <Building2 className="w-5 h-5 mr-2" />
                  ISRO Ground Stations
              </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {Object.entries(stations).map(([id, station]) => (
                    <div
                      key={id}
                      className={`p-4 rounded-lg border cursor-pointer transition-all ${
                        selectedStation === id
                          ? 'border-purple-400 bg-purple-800/30'
                          : 'border-purple-500/30 bg-purple-800/20 hover:border-purple-400/50'
                      }`}
                      onClick={() => setSelectedStation(id)}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-semibold text-purple-200">{station.name}</h3>
                  <Badge
                          variant={isStationHealthy(id) ? 'default' : 'destructive'}
                          className={isStationHealthy(id) ? 'bg-green-600' : 'bg-red-600'}
                  >
                          {isStationHealthy(id) ? 'Healthy' : 'Unhealthy'}
                  </Badge>
                </div>
                      <div className="text-sm text-purple-300 space-y-1">
                        <div>Subnet ID: {station.subnetId}</div>
                        <div>Validators: {station.validators.length}</div>
                        <div>Min Stake: {parseInt(station.minStake) / 1e18} AVAX</div>
                      </div>
                      {selectedStation === id && (
                        <div className="mt-4 p-3 bg-purple-800/30 rounded border border-purple-500/30">
                          <div className="text-xs font-mono text-purple-200 break-all">
                            Subnet ID: {station.subnetId}
                          </div>
                          <div className="text-xs text-purple-300 mt-1">
                            Chain ID: 0x3039
                          </div>
                          <div className="text-xs text-purple-300 mt-1">
                            Last Block: {station.lastBlock}
                          </div>
                          <div className="text-xs text-purple-300 mt-1">
                            Network: Avalanche Mainnet
                  </div>
                </div>
                    )}
                  </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Avalanche Subnet Architecture */}
            <Card className="bg-purple-900/20 border-purple-500/30">
              <CardHeader>
                <CardTitle className="flex items-center text-purple-200">
                  <Network className="w-5 h-5 mr-2" />
                  Subnet Architecture
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-3 gap-6">
                  {/* Data Flow */}
                  <div className="space-y-4">
                    <h4 className="text-lg font-semibold text-purple-200">Data Flow</h4>
                    <div className="space-y-3">
                      <div className="flex items-center space-x-3">
                        <Rocket className="w-5 h-5 text-blue-400" />
                        <span className="text-purple-300">Lunar Rover</span>
                        <span className="text-white">→</span>
                        <span className="text-purple-300">Satellite Relay</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <Server className="w-5 h-5 text-green-400" />
                        <span className="text-purple-300">Satellite Relay</span>
                        <span className="text-white">→</span>
                        <span className="text-purple-300">ISRO Subnet</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <Hash className="w-5 h-5 text-purple-400" />
                        <span className="text-purple-300">ISRO Subnet</span>
                        <span className="text-white">→</span>
                        <span className="text-purple-300">Immutable Storage</span>
                  </div>
                </div>
              </div>

                  {/* ISRO Network */}
                  <div className="space-y-4">
                    <h4 className="text-lg font-semibold text-purple-200">ISRO Network</h4>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-purple-300">Network Type:</span>
                        <Badge variant="outline" className="border-purple-500/30 text-purple-200">
                          Private Subnet
                        </Badge>
                  </div>
                      <div className="flex items-center justify-between">
                        <span className="text-purple-300">Validators:</span>
                        <Badge variant="outline" className="border-purple-500/30 text-purple-200">
                          4 Active
                  </Badge>
                </div>
                      <div className="flex items-center justify-between">
                        <span className="text-purple-300">Consensus:</span>
                        <Badge variant="outline" className="border-purple-500/30 text-purple-200">
                          Avalanche
                        </Badge>
                  </div>
                </div>
                  </div>

                  {/* Subnet Details */}
                  <div className="space-y-4">
                    <h4 className="text-lg font-semibold text-purple-200">Subnet Configuration</h4>
                    <div className="grid md:grid-cols-3 gap-6 text-sm">
                      <div>
                        <span className="text-purple-300">Network Type:</span>
                        <span className="text-white ml-2">Private Subnet</span>
                      </div>
                      <div>
                        <span className="text-purple-300">Consensus:</span>
                        <span className="text-white ml-2">Avalanche</span>
                      </div>
                      <div>
                        <span className="text-purple-300">Validators:</span>
                        <span className="text-white ml-2">4 Active</span>
                      </div>
                      <div>
                        <span className="text-purple-300">Block Time:</span>
                        <span className="text-white ml-2">~2 seconds</span>
                      </div>
                      <div>
                        <span className="text-purple-300">Finality:</span>
                        <span className="text-white ml-2">~3 seconds</span>
                      </div>
                      <div>
                        <span className="text-purple-300">Status:</span>
                        <span className="text-green-400 ml-2">Deployed</span>
                </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
          </TabsContent>

          {/* Data Transfer Tab */}
          <TabsContent value="transfer" className="space-y-6">
            {/* Transfer Control */}
            <Card className="bg-purple-900/20 border-purple-500/30">
            <CardHeader>
                <CardTitle className="flex items-center justify-between text-purple-200">
                <div className="flex items-center">
                    <Rocket className="w-5 h-5 mr-2" />
                    Secure Data Transfer
                </div>
                  <div className="flex space-x-2">
                <Button
                      onClick={() => setShowSensitiveData(!showSensitiveData)}
                  size="sm"
                      variant="outline"
                      className="border-purple-500/30 text-purple-200"
                >
                      {showSensitiveData ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </Button>
                  </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
                  <div>
                    <Label htmlFor="fromStation" className="text-purple-300">From Station</Label>
                    <Select value={transferForm.fromStation} onValueChange={(value) => setTransferForm({...transferForm, fromStation: value})}>
                      <SelectTrigger className="bg-purple-800/30 border-purple-500/30 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-purple-800 border-purple-500">
                        {Object.entries(stations).map(([id, station]) => (
                          <SelectItem key={id} value={id} className="text-white">
                            {station.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="toStation" className="text-purple-300">To Station</Label>
                    <Select value={transferForm.toStation} onValueChange={(value) => setTransferForm({...transferForm, toStation: value})}>
                      <SelectTrigger className="bg-purple-800/30 border-purple-500/30 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-purple-800 border-purple-500">
                        {Object.entries(stations).map(([id, station]) => (
                          <SelectItem key={id} value={id} className="text-white">
                            {station.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                <div>
                    <Label htmlFor="dataSize" className="text-purple-300">Data Size (bytes)</Label>
                    <Input
                      id="dataSize"
                      type="number"
                      value={transferForm.dataSize}
                      onChange={(e) => setTransferForm({...transferForm, dataSize: parseInt(e.target.value)})}
                      className="bg-purple-800/30 border-purple-500/30 text-white"
                    />
                  </div>
                  <div>
                    <Label htmlFor="priority" className="text-purple-300">Priority</Label>
                    <Select value={transferForm.priority} onValueChange={(value: any) => setTransferForm({...transferForm, priority: value})}>
                      <SelectTrigger className="bg-purple-800/30 border-purple-500/30 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-purple-800 border-purple-500">
                        <SelectItem value="low" className="text-white">Low</SelectItem>
                        <SelectItem value="medium" className="text-white">Medium</SelectItem>
                        <SelectItem value="high" className="text-white">High</SelectItem>
                        <SelectItem value="critical" className="text-white">Critical</SelectItem>
                      </SelectContent>
                    </Select>
                </div>
                  <div>
                    <Label htmlFor="encryptionType" className="text-purple-300">Encryption</Label>
                    <Select value={transferForm.encryptionType} onValueChange={(value: any) => setTransferForm({...transferForm, encryptionType: value})}>
                      <SelectTrigger className="bg-purple-800/30 border-purple-500/30 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-purple-800 border-purple-500">
                        <SelectItem value="AES-256" className="text-white">AES-256</SelectItem>
                        <SelectItem value="ChaCha20-Poly1305" className="text-white">ChaCha20-Poly1305</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="route" className="text-purple-300">Route</Label>
                    <Input
                      id="route"
                      value={getTransferRoute(transferForm.fromStation, transferForm.toStation).join(" → ")}
                      readOnly
                      className="bg-purple-800/30 border-purple-500/30 text-white"
                    />
                  </div>
                </div>

                <div className="flex space-x-4">
                  {!isTransferring ? (
                    <Button onClick={handleTransfer} className="bg-purple-600 hover:bg-purple-700">
                      <Play className="w-4 h-4 mr-2" />
                      Start Transfer
                    </Button>
                  ) : (
                    <>
                      <Button onClick={() => stopTransfer("current")} variant="outline" className="border-purple-500/30 text-purple-200">
                        <Square className="w-4 h-4 mr-2" />
                        Pause
                      </Button>
                      <Button onClick={() => cancelTransfer("current")} variant="outline" className="border-red-500/30 text-red-200">
                        <X className="w-4 h-4 mr-2" />
                        Cancel
                      </Button>
                    </>
                  )}
                </div>

                {/* Transfer Progress */}
                {isTransferring && (
                  <>
                    <div className="mt-6">
                      <div className="flex justify-between text-sm text-purple-200 mb-2">
                        <span>Mission Data Package #{Math.floor(Math.random() * 10000)}</span>
                        <span>{transferProgress}%</span>
                      </div>
                      <Progress value={transferProgress} className="h-2" />
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 text-sm mt-4">
                      <div>
                        <span className="text-purple-300">Size:</span>
                        <span className="text-white ml-2">{formatDataSize(transferForm.dataSize)}</span>
                      </div>
                      <div>
                        <span className="text-purple-300">Route:</span>
                        <span className="text-white ml-2">
                          {getTransferRoute(transferForm.fromStation, transferForm.toStation).join(" → ")}
                        </span>
                      </div>
                      <div>
                        <span className="text-purple-300">ETA:</span>
                        <span className="text-white ml-2">Calculating...</span>
                      </div>
                      <div>
                        <span className="text-purple-300">Encryption:</span>
                        <span className="text-white ml-2">{transferForm.encryptionType}</span>
                </div>
              </div>
                  </>
                )}
            </CardContent>
          </Card>

            {/* Security Status */}
            <Card className="bg-purple-900/20 border-purple-500/30">
            <CardHeader>
                <CardTitle className="flex items-center text-purple-200">
                  <Shield className="w-5 h-5 mr-2" />
                Security Status
              </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="p-4 bg-green-900/30 border border-green-500/30 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                      <span className="text-green-200 font-semibold">Avalanche Subnet Verification</span>
                    </div>
                    <p className="text-sm text-green-300">All validators confirmed</p>
                </div>
                  <div className="p-4 bg-green-900/30 border border-green-500/30 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                      <span className="text-green-200 font-semibold">Private Network Access</span>
                </div>
                    <p className="text-sm text-green-300">ISRO-only subnet</p>
                </div>
                  <div className="p-4 bg-green-900/30 border border-green-500/30 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                      <span className="text-green-200 font-semibold">Validator Consensus</span>
                    </div>
                    <p className="text-sm text-green-300">4/4 validators active</p>
                  </div>
                  <div className="p-4 bg-yellow-900/30 border border-yellow-500/30 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <AlertTriangle className="w-5 h-5 text-yellow-400" />
                      <span className="text-yellow-200 font-semibold">Zero-Knowledge Proofs</span>
                    </div>
                    <p className="text-sm text-yellow-300">Implementation pending</p>
                </div>
              </div>
            </CardContent>
          </Card>
          </TabsContent>

          {/* Smart Contracts Tab */}
          <TabsContent value="contracts" className="space-y-6">
            {/* Contract Overview */}
            <Card className="bg-purple-900/20 border-purple-500/30">
          <CardHeader>
                <CardTitle className="flex items-center text-purple-200">
                  <FileText className="w-5 h-5 mr-2" />
                  Deployed Smart Contracts
            </CardTitle>
          </CardHeader>
          <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  {/* ISRO Token Contract */}
                  <div className="p-4 bg-purple-800/30 border border-purple-500/30 rounded-lg">
                    <div className="flex items-center space-x-2 mb-3">
                      <Hash className="w-5 h-5 text-blue-400" />
                      <h3 className="text-lg font-semibold text-blue-200">ISRO Token Contract</h3>
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-purple-300">Address:</span>
                        <span className="text-white font-mono text-xs break-all">{DEPLOYED_CONTRACTS.isroToken}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-purple-300">Name:</span>
                        <span className="text-white">ISRO Lunar Mission Token</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-purple-300">Symbol:</span>
                        <span className="text-white">ISRO</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-purple-300">Supply:</span>
                        <span className="text-white">1,000,000,000 ISRO</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-purple-300">Status:</span>
                        <Badge className="bg-green-600">Deployed</Badge>
                      </div>
                    </div>
                  </div>

                  {/* Data Transfer Contract */}
                  <div className="p-4 bg-purple-800/30 border border-purple-500/30 rounded-lg">
                    <div className="flex items-center space-x-2 mb-3">
                      <Server className="w-5 h-5 text-green-400" />
                      <h3 className="text-lg font-semibold text-green-200">Data Transfer Contract</h3>
                    </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                        <span className="text-purple-300">Address:</span>
                        <span className="text-white font-mono text-xs break-all">{DEPLOYED_CONTRACTS.isroDataTransfer}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-purple-300">Min Stake:</span>
                        <span className="text-white">2,000,000 AVAX</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-purple-300">Timeout:</span>
                        <span className="text-white">1 hour</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-purple-300">Max Data:</span>
                        <span className="text-white">100 GB</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-purple-300">Status:</span>
                        <Badge className="bg-green-600">Deployed</Badge>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Contract Functions */}
            <Card className="bg-purple-900/20 border-purple-500/30">
              <CardHeader>
                <CardTitle className="flex items-center text-purple-200">
                  <Smartphone className="w-5 h-5 mr-2" />
                  Contract Functions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-lg font-semibold text-purple-200 mb-3">Token Functions</h4>
                    <div className="space-y-2 text-sm">
                      <div className="p-2 bg-purple-800/30 rounded border border-purple-500/30">
                        <span className="text-purple-300">stake(amount)</span>
                        <span className="text-white ml-2">- Stake ISRO tokens</span>
                      </div>
                      <div className="p-2 bg-purple-800/30 rounded border border-purple-500/30">
                        <span className="text-purple-300">stakeAsValidator(amount)</span>
                        <span className="text-white ml-2">- Stake as validator</span>
                      </div>
                      <div className="p-2 bg-purple-800/30 rounded border border-purple-500/30">
                        <span className="text-purple-300">claimRewards()</span>
                        <span className="text-white ml-2">- Claim staking rewards</span>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold text-purple-200 mb-3">Transfer Functions</h4>
                    <div className="space-y-2 text-sm">
                      <div className="p-2 bg-purple-800/30 rounded border border-purple-500/30">
                        <span className="text-purple-300">initiateTransfer(to, size, checksum)</span>
                        <span className="text-white ml-2">- Start data transfer</span>
                      </div>
                      <div className="p-2 bg-purple-800/30 rounded border border-purple-500/30">
                        <span className="text-purple-300">completeTransfer(id, checksum)</span>
                        <span className="text-white ml-2">- Complete transfer</span>
                      </div>
                      <div className="p-2 bg-purple-800/30 rounded border border-purple-500/30">
                        <span className="text-purple-300">registerStation(address, name)</span>
                        <span className="text-white ml-2">- Register new station</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Monitoring Tab */}
          <TabsContent value="monitoring" className="space-y-6">
            {/* Network Health */}
            <Card className="bg-purple-900/20 border-purple-500/30">
              <CardHeader>
                <CardTitle className="flex items-center text-purple-200">
                  <TrendingUp className="w-5 h-5 mr-2" />
                  Network Health
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {Object.entries(stationHealth).map(([stationId, health]) => (
                    <div key={stationId} className="p-4 bg-purple-800/30 border border-purple-500/30 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-semibold text-purple-200">{stations[stationId].name}</h4>
                        <Badge
                          variant={health.uptime > 0.95 ? 'default' : 'destructive'}
                          className={health.uptime > 0.95 ? 'bg-green-600' : 'bg-red-600'}
                        >
                          {Math.round(health.uptime * 100)}%
                        </Badge>
                      </div>
                      <div className="space-y-1 text-sm text-purple-300">
                        <div>Consensus: {Math.round(health.consensusRate * 100)}%</div>
                        <div>Latency: {Math.round(health.networkLatency)}ms</div>
                        <div>Throughput: {Math.round(health.dataThroughput)} MB/s</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

            {/* Active Transfers */}
            <Card className="bg-purple-900/20 border-purple-500/30">
            <CardHeader>
                <CardTitle className="flex items-center text-purple-200">
                  <BarChart3 className="w-5 h-5 mr-2" />
                  Active Transfers
              </CardTitle>
            </CardHeader>
            <CardContent>
                {activeTransfers.length === 0 ? (
                  <div className="text-center py-8 text-purple-300">
                    <Rocket className="w-12 h-12 mx-auto mb-4 text-purple-400" />
                    <p>No active transfers</p>
                    <p className="text-sm">Start a new transfer to see it here</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {activeTransfers.map((transfer) => (
                      <div key={transfer.requestId} className="p-4 bg-purple-800/30 border border-purple-500/30 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-purple-200 font-semibold">Transfer #{transfer.requestId.slice(0, 8)}</span>
                          <Badge
                            variant={transfer.status === 'completed' ? 'default' : 'secondary'}
                            className={transfer.status === 'completed' ? 'bg-green-600' : 'bg-purple-600'}
                          >
                            {transfer.status}
                          </Badge>
                        </div>
                        <div className="grid md:grid-cols-3 gap-4 text-sm">
                          <div>
                            <span className="text-purple-300">Route:</span>
                            <span className="text-white ml-2">{transfer.currentStation} → {transfer.nextStation}</span>
                </div>
                  <div>
                            <span className="text-purple-300">Progress:</span>
                            <span className="text-white ml-2">{transfer.progress}%</span>
                  </div>
                  <div>
                            <span className="text-purple-300">ETA:</span>
                            <span className="text-white ml-2">{formatTime(transfer.estimatedTime)}</span>
                          </div>
                        </div>
                        {transfer.status === 'failed' && transfer.error && (
                          <div className="mt-2 p-2 bg-red-900/30 border border-red-500/30 rounded text-red-200 text-sm">
                            Error: {transfer.error}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Subnet Connection Details */}
        <Card className="mt-8 bg-purple-900/20 border-purple-500/30">
          <CardHeader>
            <CardTitle className="flex items-center text-purple-200">
              <Globe className="w-5 h-5 mr-2" />
              Subnet Connection Details
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <span className="text-purple-300">Network:</span>
                <span className="text-white ml-2">Avalanche Mainnet</span>
              </div>
              <div>
                <span className="text-purple-300">Chain ID:</span>
                <span className="text-white ml-2">43114</span>
              </div>
              <div>
                <span className="text-purple-300">RPC URL:</span>
                <span className="text-white ml-2">api.avax.network</span>
              </div>
              <div>
                <span className="text-purple-300">Explorer:</span>
                <span className="text-white ml-2">snowtrace.io</span>
                </div>
              </div>
            </CardContent>
          </Card>

        {/* Avalanche Subnet Details */}
        <Card className="mt-6 bg-purple-900/20 border-purple-500/30">
            <CardHeader>
            <CardTitle className="flex items-center text-purple-200">
              <Key className="w-5 h-5 mr-2" />
              Subnet Security Features
              </CardTitle>
            </CardHeader>
            <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-lg font-semibold text-purple-200 mb-3">Network Isolation</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <span className="text-purple-300">Private validator set</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <span className="text-purple-300">No external node access</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <span className="text-purple-300">ISRO-only subnet</span>
                  </div>
                </div>
              </div>
                  <div>
                <h4 className="text-lg font-semibold text-purple-200 mb-3">Immutable Storage</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <span className="text-purple-300">Avalanche consensus</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <span className="text-purple-300">Cryptographic verification</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-green-400" />
                    <span className="text-purple-300">Tamper-proof records</span>
                  </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
      </div>
    </div>
  );
};

export default DataTransfer;