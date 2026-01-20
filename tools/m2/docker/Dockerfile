# openEMS + AppCSXCAD Docker Image for Formula Foundry M2
#
# Provides a reproducible FDTD simulation environment with:
# - openEMS 0.0.35 (FDTD solver)
# - CSXCAD 0.6.3 (geometry library)
# - AppCSXCAD (optional visualization)
# - Python bindings for scripted simulations
#
# Build context: tools/m2/docker/
# Usage: docker build -t formula-foundry-openems:0.0.35 .

# Pin to Ubuntu 22.04 for build reproducibility
FROM ubuntu:22.04@sha256:77906da86b60585ce12215807090eb327e7386c8fafb5402369e421f44eff17e AS builder

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libcgal-dev \
    libfftw3-dev \
    libhdf5-dev \
    libopenmpi-dev \
    libtinyxml-dev \
    libvtk9-dev \
    libvtk9-qt-dev \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-setuptools \
    qtbase5-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone and build CSXCAD (geometry library) - pinned to v0.6.3
WORKDIR /build
RUN git clone --depth 1 --branch v0.6.3 https://github.com/thliebig/CSXCAD.git
WORKDIR /build/CSXCAD/build
RUN cmake .. \
    -DCMAKE_INSTALL_PREFIX=/opt/openems \
    -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && make install

# Clone and build openEMS - pinned to v0.0.35
WORKDIR /build
RUN git clone --depth 1 --branch v0.0.35 https://github.com/thliebig/openEMS.git
WORKDIR /build/openEMS/build
RUN cmake .. \
    -DCMAKE_INSTALL_PREFIX=/opt/openems \
    -DCSXCAD_ROOT_DIR=/opt/openems \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_MPI=ON \
    && make -j$(nproc) \
    && make install

# Clone and build AppCSXCAD (visualization tool, optional)
WORKDIR /build
RUN git clone --depth 1 --branch v0.2.3 https://github.com/thliebig/AppCSXCAD.git
WORKDIR /build/AppCSXCAD/build
RUN cmake .. \
    -DCMAKE_INSTALL_PREFIX=/opt/openems \
    -DCSXCAD_ROOT_DIR=/opt/openems \
    -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && make install

# Build Python bindings for CSXCAD
WORKDIR /build/CSXCAD/python
RUN CSXCAD_ROOT_DIR=/opt/openems python3 setup.py build_ext --include-dirs=/opt/openems/include --library-dirs=/opt/openems/lib
RUN CSXCAD_ROOT_DIR=/opt/openems python3 setup.py install --prefix=/opt/openems

# Build Python bindings for openEMS
WORKDIR /build/openEMS/python
RUN OPENEMS_ROOT_DIR=/opt/openems CSXCAD_ROOT_DIR=/opt/openems python3 setup.py build_ext --include-dirs=/opt/openems/include --library-dirs=/opt/openems/lib
RUN OPENEMS_ROOT_DIR=/opt/openems CSXCAD_ROOT_DIR=/opt/openems python3 setup.py install --prefix=/opt/openems


# Runtime image - smaller footprint
FROM ubuntu:22.04@sha256:77906da86b60585ce12215807090eb327e7386c8fafb5402369e421f44eff17e AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-system1.74.0 \
    libboost-thread1.74.0 \
    libcgal13 \
    libfftw3-double3 \
    libgomp1 \
    libhdf5-103-1 \
    libopenmpi3 \
    libtinyxml2.6.2v5 \
    libvtk9.1 \
    libvtk9.1-qt \
    python3 \
    python3-h5py \
    python3-matplotlib \
    python3-numpy \
    python3-scipy \
    qt5-gtk-platformtheme \
    && rm -rf /var/lib/apt/lists/*

# Copy built artifacts from builder stage
COPY --from=builder /opt/openems /opt/openems

# Set environment for openEMS
ENV PATH="/opt/openems/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/openems/lib:$LD_LIBRARY_PATH" \
    PYTHONPATH="/opt/openems/lib/python3/dist-packages:$PYTHONPATH"

# Create non-root user for running simulations
RUN useradd -m -s /bin/bash openems
USER openems
WORKDIR /workspace

# Default entrypoint runs openEMS CLI
ENTRYPOINT ["openEMS"]
CMD ["--help"]

# Labels for traceability
LABEL org.opencontainers.image.title="formula-foundry-openems" \
      org.opencontainers.image.description="openEMS FDTD solver for Formula Foundry M2" \
      org.opencontainers.image.version="0.0.35" \
      org.opencontainers.image.vendor="Formula Foundry" \
      org.opencontainers.image.source="https://github.com/thliebig/openEMS" \
      formula-foundry.openems.version="0.0.35" \
      formula-foundry.csxcad.version="0.6.3" \
      formula-foundry.appcsxcad.version="0.2.3"
