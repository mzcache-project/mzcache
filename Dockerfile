# Dockerfile — reproducible host build environment for the mzCache AE package.
# See EVALUATION.md Section 1.1 (Option A). It bakes in the pinned toolchain, adb, the
# Python plotting env, and the Android NDK r29 so the CLI-experiment builds are
# reproducible from one image. It deliberately does NOT contain:
#   * the models / prefill-state files  -> downloaded into the mounted repo (Section 1.2)
#   * the phone                         -> adb talks to the HOST's adb server,
#                                          so run the container with --network host
#   * the Android Studio / Gradle stack -> the Section 7 app is built natively on the host
#   * CUDA                              -> the Section 6.3 accuracy tools build on the GPU
#                                          server (build_server_cuda.sh; use an
#                                          nvidia/cuda base image there)
#
# Pull & enter (or build it yourself with `docker build -t mzcache-ae .`):
#   adb start-server
#   docker pull appleyu1/mzcache-ae:v0.9
#   docker run --rm -it --network host -v "$PWD:/workspace" \
#       -e ADB_SERIAL=$ADB_SERIAL appleyu1/mzcache-ae:v0.9
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates \
        curl unzip file adb \
        python3 python3-matplotlib python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Android NDK r29 — same pin + checksum as scripts/setup/get_ndk.sh.
COPY scripts/setup/get_ndk.sh /tmp/get_ndk.sh
RUN /tmp/get_ndk.sh /opt >/dev/null && rm /tmp/get_ndk.sh
ENV ANDROID_NDK=/opt/android-ndk-r29

# Avoid git "dubious ownership" on the bind-mounted repo (different uid).
RUN git config --global --add safe.directory '*'

WORKDIR /workspace
CMD ["/bin/bash"]
