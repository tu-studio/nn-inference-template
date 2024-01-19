#include <benchmark/benchmark.h>
#include "../source/PluginProcessor.h"
#include "../source/PluginEditor.h"
#include "utils/TestThread.h"

// TODO Make sure that benchmarks also work when HOST_BUFFER_SIZE % MODEL_INPUT_SIZE != 0

/* ============================================================ *
 * ========================= Configs ========================== *
 * ============================================================ */

#define NUM_ITERATIONS 50
#define NUM_REPETITIONS 10
#define PERCENTILE 0.999
#define STARTING_BUFFER_SIZE 2048
#define STOPPING_BUFFER_SIZE 8192

/* ============================================================ *
 * ===================== Helper functions ===================== *
 * ============================================================ */

static float randomSample () {
    return -1.f + (float) (std::rand()) / ((float) (RAND_MAX/2.f));
}

static double calculatePercentile(const std::vector<double>& v, double percentile) {
    // Make sure the data is not empty
    if (v.empty()) {
        throw std::invalid_argument("Input vector is empty.");
    }

    // Sort the data in ascending order
    std::vector<double> sortedData = v;
    std::sort(sortedData.begin(), sortedData.end());

    // Calculate the index for the 99th percentile
    size_t n = sortedData.size();
    size_t percentileIndex = (size_t) (percentile * (n - 1));

    // Check if the index is an integer
    if (percentileIndex == static_cast<size_t>(percentileIndex)) {
        // The index is an integer, return the value at that index
        return sortedData[static_cast<size_t>(percentileIndex)];
    } else {
        // Interpolate between the two nearest values
        size_t lowerIndex = static_cast<size_t>(percentileIndex);
        size_t upperIndex = lowerIndex + 1;
        double fraction = percentileIndex - lowerIndex;
        return (1.0 - fraction) * sortedData[lowerIndex] + fraction * sortedData[upperIndex];
    }
}

const auto calculateMin = [](const std::vector<double>& v) -> double {
    return *(std::min_element(std::begin(v), std::end(v)));
};

const auto calculateMax = [](const std::vector<double>& v) -> double {
    return *(std::max_element(std::begin(v), std::end(v)));
};

/* ============================================================ *
 * =================== BENCHMARK FIXTURES ===================== *
 * ============================================================ */

class ProcessBlockFixture : public benchmark::Fixture {
public:
    inline static std::unique_ptr<int> bufferSize = nullptr;
    inline static std::unique_ptr<AudioPluginAudioProcessor> plugin = nullptr;
    inline static std::unique_ptr<juce::AudioBuffer<float>> buffer = nullptr;
    inline static std::unique_ptr<juce::MidiBuffer> midiBuffer = nullptr;
    inline static std::unique_ptr<int> repetition = nullptr;

    void pushSamplesInBuffer() {
        for (int channel = 0; channel < plugin->getTotalNumInputChannels(); channel++) {
            for (int sample = 0; sample < plugin->getBlockSize(); sample++) {
                buffer->setSample(channel, sample, randomSample());
            }
        }
    }

    ProcessBlockFixture() {
        bufferSize = std::make_unique<int>(0);
        repetition = std::make_unique<int>(0);
    }
    ~ProcessBlockFixture() {
        bufferSize.reset(); // buffersize and repetetion don't need to be reset when the plugin is reset
        repetition.reset();
    }

    void SetUp(const ::benchmark::State& state);

    void TearDown(const ::benchmark::State& state);
};

class SingletonSetup {
public:

    ProcessBlockFixture& fixture;
    inline static std::unique_ptr<SingletonSetup> setup = nullptr;

    SingletonSetup(ProcessBlockFixture& thisFixture, const ::benchmark::State& state) : fixture(thisFixture) {
        auto gui = juce::ScopedJuceInitialiser_GUI {};
        fixture.buffer = std::make_unique<juce::AudioBuffer<float>>(2, *fixture.bufferSize);
        fixture.midiBuffer = std::make_unique<juce::MidiBuffer>();
        fixture.plugin = std::make_unique<AudioPluginAudioProcessor>();
        std::ignore = state;
    }

    ~SingletonSetup() {
        fixture.midiBuffer.reset();
        fixture.buffer.reset();
        fixture.plugin.reset();
    }

    static void PerformSetup(ProcessBlockFixture& fixture, const ::benchmark::State& state) {
        if (setup == nullptr) {
            setup = std::make_unique<SingletonSetup>(fixture, state);
        }
        if (*fixture.bufferSize != (int) state.range(0)) {
            *fixture.bufferSize = (int) state.range(0);
            std::cout << "\n------------------------------------------------------------------------------------------------" << std::endl;
            std::cout << "Sample Rate 44100 Hz | Buffer Size " << *fixture.bufferSize << " = " << (float) * fixture.bufferSize * 1000.f/44100.f << " ms" << std::endl;
            std::cout << "------------------------------------------------------------------------------------------------\n" << std::endl;
            fixture.buffer->setSize(2, (int) *fixture.bufferSize);
            fixture.repetition.reset(new int(0));
        }
        fixture.plugin->setPlayConfigDetails(2, 2, 44100, *fixture.bufferSize);
        fixture.plugin->prepareToPlay (44100, (int) *fixture.bufferSize);
    }

    static void PerformTearDown(ProcessBlockFixture& fixture, const ::benchmark::State& state) {
        setup.reset();
        std::ignore = fixture;
        std::ignore = state;
    }
};

void ProcessBlockFixture::SetUp(const ::benchmark::State& state) {
    SingletonSetup::PerformSetup(*this, state);
}

void ProcessBlockFixture::TearDown(const ::benchmark::State& state) {
    SingletonSetup::PerformTearDown(*this, state);
}

/* ============================================================ *
 * ================== BENCHMARK DEFINITIONS =================== *
 * ============================================================ */

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_LIBTORCH_BACKEND)(benchmark::State& state) {
    auto sessions = plugin->getInferenceManager().getInferenceThreadPool().getSessions();
    for (size_t i = 0; i < sessions.size(); i++) {
        sessions[i]->currentBackend = LIBTORCH;
    }

    int iteration = 0;

    for (auto _ : state) {
        pushSamplesInBuffer();

        bool init = plugin->getInferenceManager().isInitializing();
        int prevNumReceivedSamples = plugin->getInferenceManager().getNumReceivedSamples();

        auto start = std::chrono::high_resolution_clock::now();
        
        plugin->processBlock(*buffer, *midiBuffer);

        if (init) {
            while (plugin->getInferenceManager().getNumReceivedSamples() < prevNumReceivedSamples + plugin->getBlockSize()){
                std::this_thread::sleep_for(std::chrono::nanoseconds (10));
            }
        }
        else {
            while (plugin->getInferenceManager().getNumReceivedSamples() < prevNumReceivedSamples){
                std::this_thread::sleep_for(std::chrono::nanoseconds (10));
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());

        auto elapsedTimeMS = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

        std::cout << state.name() << "/" << state.range(0) << "/iteration:" << iteration << "/repetition:" << *repetition.get() << "\t\t\t" << elapsedTimeMS.count() << std::endl;
        iteration++;
    }
    *repetition.get() += 1;

    std::cout << "\n------------------------------------------------------------------------------------------------\n" << std::endl;
}

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_TFLITE_BACKEND)(benchmark::State& state) {
    auto sessions = plugin->getInferenceManager().getInferenceThreadPool().getSessions();
    for (size_t i = 0; i < sessions.size(); i++) {
        sessions[i]->currentBackend = TFLITE;
    }

    int iteration = 0;

    for (auto _ : state) {
        pushSamplesInBuffer();

        bool init = plugin->getInferenceManager().isInitializing();
        int prevNumReceivedSamples = plugin->getInferenceManager().getNumReceivedSamples();

        auto start = std::chrono::high_resolution_clock::now();
        
        plugin->processBlock(*buffer, *midiBuffer);

        if (init) {
            while (plugin->getInferenceManager().getNumReceivedSamples() < prevNumReceivedSamples + plugin->getBlockSize()){
                std::this_thread::sleep_for(std::chrono::nanoseconds (10));
            }
        }
        else {
            while (plugin->getInferenceManager().getNumReceivedSamples() < prevNumReceivedSamples){
                std::this_thread::sleep_for(std::chrono::nanoseconds (10));
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());

        auto elapsedTimeMS = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

        std::cout << state.name() << "/" << state.range(0) << "/iteration:" << iteration << "/repetition:" << *repetition.get() << "\t\t\t" << elapsedTimeMS.count() << std::endl;
        iteration++;
    }
    *repetition.get() += 1;

    std::cout << "\n------------------------------------------------------------------------------------------------\n" << std::endl;
}

BENCHMARK_DEFINE_F(ProcessBlockFixture, BM_ONNX_BACKEND)(benchmark::State& state) {
    auto& sessions = plugin->getInferenceManager().getInferenceThreadPool().getSessions();
    for (size_t i = 0; i < sessions.size(); i++) {
        sessions[i]->currentBackend.store(ONNX);
    }

    int iteration = 0;

    for (auto _ : state) {
        pushSamplesInBuffer();

        bool init = plugin->getInferenceManager().isInitializing();
        int prevNumReceivedSamples = plugin->getInferenceManager().getNumReceivedSamples();

        auto start = std::chrono::high_resolution_clock::now();
        
        plugin->processBlock(*buffer, *midiBuffer);

        if (init) {
            while (plugin->getInferenceManager().getNumReceivedSamples() < prevNumReceivedSamples + plugin->getBlockSize()){
                std::this_thread::sleep_for(std::chrono::nanoseconds (10));
            }
        }
        else {
            while (plugin->getInferenceManager().getNumReceivedSamples() < prevNumReceivedSamples){
                std::this_thread::sleep_for(std::chrono::nanoseconds (10));
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());

        auto elapsedTimeMS = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);

        std::cout << state.name() << "/" << state.range(0) << "/iteration:" << iteration << "/repetition:" << *repetition.get() << "\t\t\t" << elapsedTimeMS.count() << std::endl;
        iteration++;
    }
    *repetition.get() += 1;

    std::cout << "\n------------------------------------------------------------------------------------------------\n" << std::endl;
}

/* ============================================================ *
 * ================== BENCHMARK REGISTRATION ================== *
 * ============================================================ */

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_LIBTORCH_BACKEND)
->Unit(benchmark::kMillisecond)
->Iterations(NUM_ITERATIONS)->Repetitions(NUM_REPETITIONS)
->RangeMultiplier(2)->Range(STARTING_BUFFER_SIZE, STOPPING_BUFFER_SIZE)
->ComputeStatistics("min", calculateMin)
->ComputeStatistics("max", calculateMax)
->ComputeStatistics("percentile", [](const std::vector<double>& v) -> double {
    return calculatePercentile(v, PERCENTILE);
  })
->DisplayAggregatesOnly(false)
->UseManualTime();

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_TFLITE_BACKEND)
->Unit(benchmark::kMillisecond)
->Iterations(NUM_ITERATIONS)->Repetitions(NUM_REPETITIONS)
->RangeMultiplier(2)->Range(STARTING_BUFFER_SIZE, STOPPING_BUFFER_SIZE)
->ComputeStatistics("min", calculateMin)
->ComputeStatistics("max", calculateMax)
->ComputeStatistics("percentile", [](const std::vector<double>& v) -> double {
    return calculatePercentile(v, PERCENTILE);
  })
->DisplayAggregatesOnly(false)
->UseManualTime();

BENCHMARK_REGISTER_F(ProcessBlockFixture, BM_ONNX_BACKEND)
->Unit(benchmark::kMillisecond)
->Iterations(NUM_ITERATIONS)->Repetitions(NUM_REPETITIONS)
->RangeMultiplier(2)->Range(STARTING_BUFFER_SIZE, STOPPING_BUFFER_SIZE)
->ComputeStatistics("min", calculateMin)
->ComputeStatistics("max", calculateMax)
->ComputeStatistics("percentile", [](const std::vector<double>& v) -> double {
    return calculatePercentile(v, PERCENTILE);
  })
->DisplayAggregatesOnly(false)
->UseManualTime();
