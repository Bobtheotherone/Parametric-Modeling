# openEMS Docker Environment for Formula Foundry M2

This directory contains the Docker configuration for running openEMS FDTD simulations
as part of the Formula Foundry M2 milestone.

## Pinned Versions

| Component  | Version | Source                                    |
|------------|---------|-------------------------------------------|
| openEMS    | 0.0.35  | https://github.com/thliebig/openEMS       |
| CSXCAD     | 0.6.3   | https://github.com/thliebig/CSXCAD        |
| AppCSXCAD  | 0.2.3   | https://github.com/thliebig/AppCSXCAD     |
| Base Image | Ubuntu 22.04 | `ubuntu:22.04@sha256:77906da8...`    |

## Quick Start

### Build the Image

```bash
cd tools/m2/docker
docker build -t formula-foundry-openems:0.0.35 .
```

### Run openEMS (CPU-only)

```bash
# Interactive shell
docker compose up openems

# Run a simulation
docker compose run --rm openems openEMS /workspace/data/sim.xml
```

### Run openEMS with GPU Acceleration

```bash
docker compose --profile gpu up openems-gpu
```

## GPU Passthrough Requirements

openEMS can leverage GPU acceleration for certain operations. To enable GPU support,
you must configure NVIDIA Container Toolkit on your host system.

### Prerequisites

1. **NVIDIA GPU** with CUDA compute capability 3.5 or higher
2. **NVIDIA Driver** version 525.60.13 or later (for CUDA 12.x support)
3. **Docker** version 19.03 or later (with native GPU support)
4. **NVIDIA Container Toolkit** (nvidia-docker2)

### Installation (Ubuntu/Debian)

```bash
# 1. Install NVIDIA drivers (if not already installed)
sudo apt-get update
sudo apt-get install -y nvidia-driver-535

# 2. Configure the NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 4. Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 5. Verify installation
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Installation (RHEL/CentOS/Fedora)

```bash
# 1. Install NVIDIA drivers via the CUDA repository
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install -y cuda-drivers

# 2. Configure NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | \
    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# 3. Install NVIDIA Container Toolkit
sudo dnf install -y nvidia-container-toolkit

# 4. Configure Docker and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 5. Verify
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### WSL2 (Windows Subsystem for Linux)

GPU passthrough in WSL2 requires additional setup:

1. **Windows Requirements**:
   - Windows 11 (or Windows 10 version 21H2+)
   - NVIDIA GPU driver for Windows with WSL support (GameReady or Studio driver 510.06+)
   - WSL2 with a supported Linux distribution

2. **WSL2 Setup**:
   ```bash
   # Inside WSL2, the NVIDIA driver is automatically available
   # Just install the container toolkit
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo service docker restart
   ```

3. **Verify GPU access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

### Troubleshooting GPU Passthrough

| Issue | Solution |
|-------|----------|
| `nvidia-smi` not found in container | Ensure NVIDIA Container Toolkit is installed and Docker is restarted |
| "no NVIDIA GPU detected" | Check that `nvidia-smi` works on the host first |
| Permission denied on `/dev/nvidia*` | Add user to `docker` group or run with `sudo` |
| "Failed to initialize NVML" | Driver mismatch - update host driver or container base image |
| WSL2: GPU not visible | Update Windows NVIDIA driver to latest version with WSL support |

### Resource Configuration

The `docker-compose.yml` includes resource limits to prevent runaway simulations:

```yaml
# CPU-only service
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G

# GPU service
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Adjust these limits based on your hardware and simulation requirements.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | 4 | Number of OpenMP threads for parallel computation |
| `OPENEMS_OPTS` | (empty) | Additional openEMS CLI options |
| `NVIDIA_VISIBLE_DEVICES` | all | GPU devices to expose (e.g., `0,1` for first two GPUs) |

## Volume Mounts

The compose file mounts the following directories:

- `../../../data` → `/workspace/data` - Input simulation files
- `../../../artifacts` → `/workspace/artifacts` - Output artifacts
- `simulation-cache` → `/workspace/.cache` - Persistent cache volume

## AppCSXCAD Visualization

For visualization with AppCSXCAD, you need X11 forwarding:

```bash
# Allow X11 connections
xhost +local:docker

# Run AppCSXCAD
docker compose --profile gui up appcsxcad

# Clean up X11 permissions after
xhost -local:docker
```

On macOS, install XQuartz and enable "Allow connections from network clients".

## CI Integration

For CI/CD pipelines, use the `openems-run` service:

```bash
docker compose --profile ci run --rm openems-run /workspace/data/sim.xml --engine=multithreaded
```

This service runs non-interactively and exits after the simulation completes.

## Building a Custom Image

If you need to modify the openEMS build (e.g., different compile flags):

```bash
# Build with custom options
docker build \
  --build-arg CMAKE_BUILD_TYPE=Debug \
  -t formula-foundry-openems:0.0.35-debug .
```

## Security Notes

- The default container runs as a non-root user (`openems`)
- GPU passthrough requires `--privileged` or specific device permissions
- X11 forwarding should be disabled in production environments
