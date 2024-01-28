## Usage

How to validate real time principles, currently only supported on macOS / Linux / WSL: 


Get docker image:
```sh
docker pull realtimesanitizer/radsan-clang
```
Mount docker image to repo:
```sh
sudo docker run -v  $(pwd):/nn-inferenced-template -it realtimesanitizer/radsan-clang /bin/bash
```
**Inside docker:**

Install important packages:
```sh
apt-get update && apt-get install -y git cmake vim
```
Install juce dependencies:

```sh
apt install libasound2-dev libjack-jackd2-dev \
    ladspa-sdk \
    libcurl4-openssl-dev  \
    libfreetype6-dev \
    libx11-dev libxcomposite-dev libxcursor-dev libxcursor-dev libxext-dev libxinerama-dev libxrandr-dev libxrender-dev \
    libwebkit2gtk-4.0-dev \
    libglu1-mesa-dev mesa-common-dev
```

Build:
```sh
cd /nn-inference-template/nn-inference-template/

cmake . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release

# Build and test tflite
cmake --build cmake-build-release --config Release --target validate-real-time-principles-tflite
RADSAN_ERROR_MODE=continue ./cmake-build-release/validate-real-time-principles/tensorflow-lite/validate-real-time-principles-tflite 2>&1 | tee validate-real-time-principles-tflite.txt

# Build onnx
cmake --build cmake-build-release --config Release --target validate-real-time-principles-onnxruntime
RADSAN_ERROR_MODE=continue ./cmake-build-release/validate-real-time-principles/onnxruntime/validate-real-time-principles-onnxruntime 2>&1 | tee validate-real-time-principles-onnxruntime.txt

# Build libtorch
cmake --build cmake-build-release --config Release --target validate-real-time-principles-libtorch
RADSAN_ERROR_MODE=continue ./cmake-build-release/validate-real-time-principles/libtorch/validate-real-time-principles-libtorch 2>&1 | tee validate-real-time-principles-libtorch.txt

```
