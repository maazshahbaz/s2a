# Server Specifications & Performance Configuration

## Hardware Specifications

### CPU - Intel Xeon Platinum 8462Y+
- **Total vCPUs**: 128 (64 physical cores with hyperthreading)
- **Architecture**: 2 sockets × 32 cores each
- **Base Frequency**: 2.8GHz
- **Max Boost Frequency**: 4.1GHz
- **Cache**: L1d: 3MB, L2 & L3 optimized for server workloads
- **Features**: AVX-512, Intel AMX, hardware acceleration for AI/ML workloads

### Memory (RAM)
- **Total Capacity**: 503GB (512GB with ~9GB system reserved)
- **Available**: 470GB free
- **Current Usage**: 29GB used
- **Type**: High-performance server RAM
- **Swap**: 2GB (minimal, as expected with abundant RAM)

### GPU - NVIDIA H100 PCIe
- **Model**: NVIDIA H100 PCIe 80GB
- **Total VRAM**: 81,559MB (80GB)
- **Available VRAM**: ~78GB (3GB currently used by Python process)
- **CUDA Version**: 12.8 (latest)
- **Driver Version**: 570.169
- **Power**: 103W / 350W (under light load)
- **Temperature**: 68°C (normal operating range)

### Storage
- **Total Space**: 439GB
- **Used**: 374GB (90% utilization)
- **Available**: 43GB free
- ⚠️ **Warning**: Disk space approaching capacity - monitor and cleanup recommended

## Performance Configuration Recommendations

### Current Configuration (Conservative)
```python
# batch_processor.py
num_workers: int = 8
max_batch_size: int = 128  
max_queue_size: int = 500
```

**Capacity**: ~40-50 concurrent jobs
**Resource Utilization**: 
- CPU: 6% (8/128 cores)
- RAM: 3% (~16GB/470GB)
- GPU: 15% (~12GB/80GB VRAM)

### Recommended Configuration (Optimized)
```python
# batch_processor.py  
num_workers: int = 32
max_batch_size: int = 128
max_queue_size: int = 2000
```

**Capacity**: ~200-300 concurrent jobs
**Resource Utilization**:
- CPU: 25% (32/128 cores) 
- RAM: 7% (~32GB/470GB)
- GPU: 60% (~48GB/80GB VRAM)

### Maximum Configuration (Aggressive)
```python
# batch_processor.py
num_workers: int = 64
max_batch_size: int = 128
max_queue_size: int = 5000  
```

**Capacity**: ~500-1000 concurrent jobs
**Resource Utilization**:
- CPU: 50% (64/128 cores)
- RAM: 14% (~64GB/470GB) 
- GPU: 90% (~72GB/80GB VRAM)

## Bottleneck Analysis

### Primary Bottleneck: GPU Memory
- **Single H100** shared across all workers
- **VRAM limit**: ~80GB total capacity
- **Per worker estimate**: 1-2GB during inference
- **Safe maximum**: 40-50 workers before memory pressure

### Secondary Bottleneck: GPU Compute
- **Sequential processing**: Workers compete for GPU cycles
- **Inference time**: ~2-5 seconds per short audio job
- **Chunked processing**: Long audios processed sequentially within job

### Non-Bottlenecks
- **CPU**: Massively over-provisioned (128 cores available)
- **RAM**: Abundant capacity (470GB available)
- **Network I/O**: Not tested but server-grade expected

## Monitoring Commands

### Real-time GPU Monitoring
```bash
# GPU memory and utilization
nvidia-smi -l 1

# Continuous monitoring with timestamps
watch -n 1 nvidia-smi
```

### System Resource Monitoring
```bash
# CPU and memory usage
htop

# Memory breakdown
free -h

# Disk usage
df -h

# CPU specifications
lscpu
```

### Application Performance Monitoring
```bash
# S2A service stats
curl http://localhost:8001/v1/statistics/stats | jq

# Queue and processing metrics
curl http://localhost:8001/v1/statistics/stats | jq .batch_processor

# Health check
curl http://localhost:8001/v1/statistics/health
```

## Scaling Guidelines

### Incremental Scaling Approach
1. **Start**: 8 workers → monitor for 24 hours
2. **Step 1**: Increase to 16 workers → monitor GPU memory
3. **Step 2**: Increase to 32 workers → monitor performance metrics
4. **Step 3**: Test 48-64 workers → watch for degradation

### Warning Signs (Scale Down)
- GPU memory usage > 95%
- Queue consistently > 80% full
- RTF (Real-time Factor) increasing significantly
- High error rates in transcription
- System memory pressure warnings

### Performance Indicators (Scale Up)
- GPU memory usage < 70% 
- Queue rarely exceeds 50% capacity
- RTF remains < 0.2 for most jobs
- CPU utilization < 40%
- Error rates < 1%

## Expected Performance at Scale

### With 32 Workers
- **Short audios** (2-24min): 200-300 concurrent jobs
- **Long audios** (24min-2hr): 32 concurrent jobs  
- **Mixed workload**: ~250 concurrent jobs
- **Queue throughput**: 1000+ jobs/hour

### With 64 Workers (Maximum)
- **Short audios**: 400-600 concurrent jobs
- **Long audios**: 64 concurrent jobs
- **Mixed workload**: ~500 concurrent jobs  
- **Queue throughput**: 2000+ jobs/hour

## Maintenance Notes

### Disk Space Management
- **Current usage**: 90% (374GB/439GB) - **CRITICAL**
- **Available space**: 43GB remaining

#### Major Space Consumers (Disk Usage Analysis)
1. **Audio Data Corpus: 71GB** 🎵
   ```
   /home/sj/data/s2a/corpus/
   ├── customer_support: 54GB
   └── sales: 17GB
   ```
   *Status: Training/test data - likely needed*

2. **Docker Images: 96GB** 🐳 ⚠️ **CLEANUP TARGET**
   ```
   Docker Images: 96.51GB (99% reclaimable)
   Build Cache: 16.69GB (100% reclaimable)
   ```
   *Recovery potential: ~113GB*

3. **HuggingFace Model Cache: 38GB** 🤗
   ```
   /home/sj/.cache/huggingface/
   ├── hub: 28GB (downloaded models)
   ├── xet: 9.3GB
   └── pip cache: 12GB
   ```
   *Recovery potential: 10-20GB (review unused models)*

4. **S2A Project: 8.1GB**
   ```
   /home/sj/work_space/bytepulse-ai/s2a/
   ```
   *Status: Current project - keep*

#### Immediate Cleanup Commands (Safe Recovery: ~130GB)

**Priority 1: Docker Cleanup** (112GB recovery)
```bash
# Remove unused Docker images (96GB recovery)
docker image prune -a

# Remove build cache (17GB recovery)  
docker builder prune

# Check space recovered
docker system df
```

**Priority 2: Python Caches** (12GB recovery)
```bash
# Clear pip cache
pip cache purge

# Verify cache cleared
du -sh ~/.cache/pip
```

**Priority 3: HuggingFace Models** (10-20GB potential)
```bash
# List cached models to review
ls ~/.cache/huggingface/hub/

# Remove specific unused models (careful!)
huggingface-cli delete-cache --dir ~/.cache/huggingface/hub/[model-name]

# Or clear entire cache (DANGEROUS - will re-download models)
# rm -rf ~/.cache/huggingface/hub/*
```

**Priority 4: System Cleanup** (5-10GB potential)
```bash
# Clean package manager cache
sudo apt clean
sudo apt autoremove

# Clean temporary files
sudo rm -rf /tmp/*
sudo rm -rf /var/tmp/*

# Clean journal logs older than 7 days
sudo journalctl --vacuum-time=7d
```

#### Automated Cleanup Script
```bash
#!/bin/bash
# cleanup_disk.sh - Safe disk space recovery

echo "Starting disk cleanup..."
echo "Current usage: $(df -h / | grep -v Filesystem)"

# Docker cleanup (safest, biggest impact)
echo "Cleaning Docker images and cache..."
docker image prune -a -f
docker builder prune -f
docker volume prune -f

# Python caches
echo "Cleaning Python caches..."
pip cache purge

# System cleanup  
echo "System cleanup..."
sudo apt clean
sudo apt autoremove -y
sudo journalctl --vacuum-time=7d

echo "Cleanup complete!"
echo "New usage: $(df -h / | grep -v Filesystem)"
```

#### Expected Recovery Results
- **Before cleanup**: 374GB used (90%)
- **After Docker cleanup**: 262GB used (60%) 
- **After full cleanup**: 244GB used (55%)
- **Available space**: 195GB (vs current 43GB)

#### Long-term Storage Strategy
- **Move audio corpus** to external storage or dedicated data partition
- **Implement log rotation** for application logs
- **Regular cleanup schedule** (weekly Docker cleanup)
- **Monitor disk usage** alerts at 80% capacity

### Regular Monitoring
- **Daily**: Check disk space and GPU memory
- **Weekly**: Review performance metrics and error rates
- **Monthly**: Analyze scaling opportunities and resource trends

## Hardware Upgrade Path

### Next Bottleneck (if needed)
- **Multi-GPU setup**: Add second H100 for 2x GPU capacity
- **GPU memory**: Upgrade to H100 SXM (94GB VRAM) if available
- **Storage**: Add NVMe SSD for faster I/O if needed

### Current Headroom
- **CPU**: 4x over-provisioned (can handle 4x current load)
- **RAM**: 15x over-provisioned (massive headroom)
- **GPU**: 4-8x scaling potential with proper configuration

---

*Last updated: September 4, 2025*  
*Server: Intel Xeon Platinum 8462Y+ / 503GB RAM / NVIDIA H100 PCIe 80GB*