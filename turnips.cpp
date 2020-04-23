#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>



class RNG {
 public:
  RNG() : state{0, 0, 0, 0} {
  }

  RNG(uint32_t seed) {
    reseed(seed);
  }

  void reseed(uint32_t seed) {
    state[0] = 0x6C078965 * (seed ^ (seed >> 30)) + 1;
    state[1] = 0x6C078965 * (state[0] ^ (state[0] >> 30)) + 2;
    state[2] = 0x6C078965 * (state[1] ^ (state[1] >> 30)) + 3;
    state[3] = 0x6C078965 * (state[2] ^ (state[2] >> 30)) + 4;
  }

  inline uint32_t getRandomUint32() {
    uint32_t n = state[0] ^ (state[0] << 11);
    state[0] = state[1];
    state[1] = state[2];
    state[2] = state[3];
    state[3] = n ^ (n >> 8) ^ state[3] ^ (state[3] >> 19);
    return state[3];
  }

  inline bool getRandomBoolean() {
    return ((getRandomUint32() & 0x80000000) != 0);
  }

  inline int getRandomInt(int min, int max) {
    return ((static_cast<uint64_t>(getRandomUint32()) *
        static_cast<uint64_t>(max - min + 1)) >> 32) + min;
  }

  inline float getRandomFloat(float min, float max) {
    uint32_t val = 0x3F800000 | (getRandomUint32() >> 9);
    float fval = reinterpret_cast<float&>(val);
    return min + ((fval - 1.0f) * (max - min));
  }

 protected:
  uint32_t state[4];
};

inline int intCeil(float x) {
  return static_cast<int>(x + 0.99999f);
}



class Turnips {
 public:
  static const size_t numberOfPatterns = 4;
  static const size_t numberOfHalfDays = 14;
  static const size_t numberOfPrices = 700;

  inline bool computePrices(uint32_t prevPattern, RNG &rng,
      uint32_t userPattern, const int32_t userPrices[numberOfHalfDays],
      uint32_t &pattern, int32_t prices[numberOfHalfDays],
      const bool takeSundayPriceFromInput = false, const int32_t tol = 0)
      __attribute__((always_inline)) {
    if (takeSundayPriceFromInput) {
      prices[0] = userPrices[0];
      prices[1] = userPrices[1];
    } else {
      prices[0] = rng.getRandomInt(90, 110);
      if ((userPrices[0] > 0) && (std::abs(prices[0] - userPrices[0])) > tol) return false;
      prices[1] = prices[0];
    }

    x = rng.getRandomInt(0, 99);

    if (prevPattern == 0) {
      if      (x < 20) pattern = 0;
      else if (x < 50) pattern = 1;
      else if (x < 65) pattern = 2;
      else             pattern = 3;
    } else if (prevPattern == 1) {
      if      (x < 50) pattern = 0;
      else if (x < 55) pattern = 1;
      else if (x < 75) pattern = 2;
      else             pattern = 3;
    } else if (prevPattern == 2) {
      if      (x < 25) pattern = 0;
      else if (x < 70) pattern = 1;
      else if (x < 75) pattern = 2;
      else             pattern = 3;
    } else {
      if      (x < 45) pattern = 0;
      else if (x < 70) pattern = 1;
      else if (x < 85) pattern = 2;
      else             pattern = 3;
    }

    if ((userPattern < numberOfPatterns) && (pattern != userPattern)) return false;

    if (pattern == 0) {
      // PATTERN 0 - "Random": high, decreasing, high, decreasing, high
      decPhaseLen1 = (rng.getRandomBoolean() ? 3 : 2);
      decPhaseLen2 = 5 - decPhaseLen1;
      hiPhaseLen1 = rng.getRandomInt(0, 6);
      hiPhaseLen2And3 = 7 - hiPhaseLen1;
      hiPhaseLen3 = rng.getRandomInt(0, hiPhaseLen2And3 - 1);
      t = 2;

      // high phase 1
      for (size_t i = 0; i < hiPhaseLen1; i++) {
        prices[t++] = intCeil(rng.getRandomFloat(0.9, 1.4) * prices[0]);
        if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;
      }

      // decreasing phase 1
      rate = rng.getRandomFloat(0.8, 0.6);
      for (size_t i = 0; i < decPhaseLen1; i++) {
        prices[t++] = intCeil(rate * prices[0]);
        if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;
        rate -= 0.04;
        rate -= rng.getRandomFloat(0, 0.06);
      }

      // high phase 2
      for (size_t i = 0; i < (hiPhaseLen2And3 - hiPhaseLen3); i++) {
        prices[t++] = intCeil(rng.getRandomFloat(0.9, 1.4) * prices[0]);
        if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;
      }

      // decreasing phase 2
      rate = rng.getRandomFloat(0.8, 0.6);
      for (size_t i = 0; i < decPhaseLen2; i++) {
        prices[t++] = intCeil(rate * prices[0]);
        if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;
        rate -= 0.04;
        rate -= rng.getRandomFloat(0, 0.06);
      }

      // high phase 3
      for (size_t i = 0; i < hiPhaseLen3; i++) {
        prices[t++] = intCeil(rng.getRandomFloat(0.9, 1.4) * prices[0]);
        if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;
      }

    } else if (pattern == 1) {
      // PATTERN 1 - "Large Spike": decreasing middle, high spike, random low
      peakStart = rng.getRandomInt(3, 9);
      rate = rng.getRandomFloat(0.9, 0.85);

      for (t = 2; t < peakStart; t++) {
        prices[t] = intCeil(rate * prices[0]);
        if ((userPrices[t] > 0) && (std::abs(prices[t] - userPrices[t]) > tol)) return false;
        rate -= 0.03;
        rate -= rng.getRandomFloat(0, 0.02);
      }

      prices[t++] = intCeil(rng.getRandomFloat(0.9, 1.4) * prices[0]);
      if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;
      prices[t++] = intCeil(rng.getRandomFloat(1.4, 2.0) * prices[0]);
      if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;
      prices[t++] = intCeil(rng.getRandomFloat(2.0, 6.0) * prices[0]);
      if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;
      prices[t++] = intCeil(rng.getRandomFloat(1.4, 2.0) * prices[0]);
      if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;
      prices[t++] = intCeil(rng.getRandomFloat(0.9, 1.4) * prices[0]);
      if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;

      for (; t < numberOfHalfDays; t++) {
        prices[t] = intCeil(rng.getRandomFloat(0.4, 0.9) * prices[0]);
        if ((userPrices[t] > 0) && (std::abs(prices[t] - userPrices[t]) > tol)) return false;
      }

    } else if (pattern == 2) {
      // PATTERN 2 - "Decreasing": consistently decreasing
      rate = 0.9;
      rate -= rng.getRandomFloat(0, 0.05);

      for (size_t t = 2; t < numberOfHalfDays; t++) {
        prices[t] = intCeil(rate * prices[0]);
        if ((userPrices[t] > 0) && (std::abs(prices[t] - userPrices[t]) > tol)) return false;
        rate -= 0.03;
        rate -= rng.getRandomFloat(0, 0.02);
      }

    } else {
      // PATTERN 3 - "Small Spike": decreasing, spike, decreasing
      peakStart = rng.getRandomInt(2, 9);

      // decreasing phase before the peak
      rate = rng.getRandomFloat(0.9, 0.4);
      for (t = 2; t < peakStart; t++) {
        prices[t] = intCeil(rate * prices[0]);
        if ((userPrices[t] > 0) && (std::abs(prices[t] - userPrices[t]) > tol)) return false;
        rate -= 0.03;
        rate -= rng.getRandomFloat(0, 0.02);
      }

      prices[t++] = intCeil(rng.getRandomFloat(0.9, 1.4) * prices[0]);
      if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;
      prices[t++] = intCeil(rng.getRandomFloat(0.9, 1.4) * prices[0]);
      if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;
      rate = rng.getRandomFloat(1.4, 2.0);
      prices[t++] = intCeil(rng.getRandomFloat(1.4, rate) * prices[0]) - 1;
      if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;
      prices[t++] = intCeil(rate * prices[0]);
      if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;
      prices[t++] = intCeil(rng.getRandomFloat(1.4, rate) * prices[0]) - 1;
      if ((userPrices[t-1] > 0) && (std::abs(prices[t-1] - userPrices[t-1]) > tol)) return false;

      // decreasing phase after the peak
      if (t < numberOfHalfDays) {
        rate = rng.getRandomFloat(0.9, 0.4);

        for (; t < numberOfHalfDays; t++) {
          prices[t] = intCeil(rate * prices[0]);
          if ((userPrices[t] > 0) && (std::abs(prices[t] - userPrices[t]) > tol)) return false;
          rate -= 0.03;
          rate -= rng.getRandomFloat(0, 0.02);
        }
      }
    }

    return true;
  }

 protected:
  int x;
  size_t decPhaseLen1;
  size_t decPhaseLen2;
  size_t hiPhaseLen1;
  size_t hiPhaseLen2And3;
  size_t hiPhaseLen3;
  float rate;
  size_t t;
  size_t peakStart;
};



void compute(int argc, char *argv[]) {
  if (argc != 4) {
    throw std::runtime_error("wrong number of arguments");
  }

  const uint32_t seed = std::stoi(argv[2]);
  const uint32_t prevPattern = std::stoi(argv[3]);
  uint32_t pattern;
  int32_t prices[Turnips::numberOfHalfDays];
  RNG rng(seed);
  const uint32_t userPattern = 9999;
  const int32_t userPrices[Turnips::numberOfHalfDays] = {};

  Turnips turnips;
  turnips.computePrices(prevPattern, rng, userPattern, userPrices, pattern, prices);

  std::cout << pattern;
  for (size_t t = 0; t < Turnips::numberOfHalfDays; t++) std::cout << " " << prices[t];
  std::cout << std::endl;
}

void bruteForce(int argc, char *argv[]) {
  if (argc != 20) {
    throw std::runtime_error("wrong number of arguments");
  }

  const size_t numberOfWorkers = std::stoi(argv[2]);
  const size_t workerId = std::stoi(argv[3]);
  const uint32_t minSeed = static_cast<uint32_t>(static_cast<double>(workerId) *
      static_cast<double>(0xFFFFFFFF) / static_cast<double>(numberOfWorkers));
  const uint32_t maxSeed = (((numberOfWorkers > 1) && (workerId < numberOfWorkers - 1)) ?
      static_cast<uint32_t>((static_cast<double>(workerId) + 1.0) *
        static_cast<double>(0xFFFFFFFF) / static_cast<double>(numberOfWorkers)) :
      0);
  const uint64_t numberOfSeeds = static_cast<uint64_t>(maxSeed - 1) -
      static_cast<uint64_t>(minSeed) + 1;

  const uint32_t userPrevPattern = std::stoi(argv[4]);
  const uint32_t userPattern = std::stoi(argv[5]);
  int32_t userPrices[Turnips::numberOfHalfDays];

  for (size_t t = 0; t < Turnips::numberOfHalfDays; t++) {
    userPrices[t] = std::stoi(argv[t + 6]);
  }

  const uint32_t minPrevPattern = ((userPrevPattern < Turnips::numberOfPatterns) ?
      userPrevPattern : 0);
  const uint32_t maxPrevPattern = ((userPrevPattern < Turnips::numberOfPatterns) ?
      (userPrevPattern + 1) : Turnips::numberOfPatterns);
  uint64_t numberOfMatches = 0;
  std::cerr << std::fixed << std::setprecision(3);

  uint32_t pattern;
  int32_t prices[Turnips::numberOfHalfDays];
  RNG rng;
  Turnips turnips;

  for (uint32_t seed = minSeed; seed != maxSeed; seed++) {
    if ((seed % 2000000 == 0) && (seed != minSeed)) {
      const uint32_t seedsProcessed = seed - minSeed;
      const double seedsProcessedPercentage = 100.0 * static_cast<double>(seedsProcessed) /
          static_cast<double>(numberOfSeeds);
      const uint64_t numberOfPossibleMatches =
          static_cast<uint64_t>(maxPrevPattern - minPrevPattern) * seedsProcessed;
      const double matchesPercentage = 100.0 * static_cast<double>(numberOfMatches) /
          static_cast<double>(numberOfPossibleMatches);
      std::cerr << "Worker " << workerId << ": " << seedsProcessed << "/" << numberOfSeeds <<
          " processed (" << seedsProcessedPercentage << "%), " << numberOfMatches << "/" <<
          numberOfPossibleMatches << " matches (" << matchesPercentage << "%)" << std::endl;
    }

    for (uint32_t prevPattern = minPrevPattern; prevPattern < maxPrevPattern; prevPattern++) {
      rng.reseed(seed);

      if (turnips.computePrices(prevPattern, rng, userPattern, userPrices, pattern, prices)) {
        std::cout << seed << " " << prevPattern << " " << pattern;
        for (size_t t = 0; t < Turnips::numberOfHalfDays; t++) std::cout << " " << prices[t];
        std::cout << std::endl;
        numberOfMatches++;
      }
    }
  }
}



template <class T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T> vector) {
  for (size_t i = 0; i < vector.size(); i++) {
    if (i > 0) stream << " ";
    stream << vector[i];
  }

  return stream;
}



void sample(int argc, char *argv[]) {
  if (argc != 22) {
    throw std::runtime_error("wrong number of arguments");
  }

  const std::string sampleMode(argv[2]);
  const bool verbose = (sampleMode == "verbose");
  const size_t numberOfSamples = std::stoi(argv[3]);
  const size_t minimumNumberOfMatches = std::stoi(argv[4]);
  const uint32_t seed = std::stoi(argv[5]);
  const uint32_t userPrevPattern = std::stoi(argv[6]);
  const uint32_t userPattern = std::stoi(argv[7]);
  int32_t userPrices[Turnips::numberOfHalfDays];
  size_t now = 0;

  for (size_t t = 0; t < Turnips::numberOfHalfDays; t++) {
    userPrices[t] = std::stoi(argv[t + 8]);
    if (userPrices[t] > 0) now = t + 1;
  }

  uint64_t numberOfMatches = 0;
  std::cerr << std::fixed << std::setprecision(3);

  size_t i = 0;
  uint32_t prevPattern = userPrevPattern;
  const bool takeSundayPriceFromInput = (userPrices[0] > 0);
  const bool randomPrevPattern = (userPrevPattern >= Turnips::numberOfPatterns);
  uint32_t pattern;
  int32_t prices[Turnips::numberOfHalfDays];
  const int32_t tol = 1;
  RNG rng(seed);
  Turnips turnips;
  double x;

  std::vector<size_t> prevPatternCounts(Turnips::numberOfPatterns, 0);
  std::vector<size_t> patternCounts(Turnips::numberOfPatterns, 0);
  std::vector<std::vector<size_t>> pricesHistogram(Turnips::numberOfHalfDays,
      std::vector<size_t>(Turnips::numberOfPrices, 0));
  std::vector<size_t> maxPriceHistogram(Turnips::numberOfPrices, 0);

  while ((i < numberOfSamples) ||
        ((minimumNumberOfMatches == 0) || (numberOfMatches < minimumNumberOfMatches))) {
    if ((i % 2000000 == 0) && (i > 0)) {
      const double processedPercentage = 100.0 * static_cast<double>(i) /
          static_cast<double>(numberOfSamples);
      const double matchesPercentage = 100.0 * static_cast<double>(numberOfMatches) /
          static_cast<double>(i);
      std::cerr << "Seed " << seed << ": " << i << "/" << numberOfSamples << " processed (" <<
          processedPercentage << "%), " << numberOfMatches << "/" << i << " matches (" <<
          matchesPercentage << "%)" << std::endl;
    }

    if (randomPrevPattern) {
      x = rng.getRandomFloat(0.0, 1.0);

      if (x < 0.34627733) prevPattern = 0;
      else if (x < 0.59364012) prevPattern = 1;
      else if (x < 0.74124752) prevPattern = 2;
      else prevPattern = 3;
    }

    if (turnips.computePrices(prevPattern, rng, userPattern, userPrices, pattern, prices,
          takeSundayPriceFromInput, tol)) {
      numberOfMatches++;

      if (verbose) {
        std::cout << prevPattern << " " << pattern;
        for (size_t t = 0; t < Turnips::numberOfHalfDays; t++) std::cout << " " << prices[t];
        std::cout << std::endl;
      } else {
        prevPatternCounts[prevPattern]++;
        patternCounts[pattern]++;

        for (size_t t = 0; t < Turnips::numberOfHalfDays; t++) {
          pricesHistogram[t][prices[t]]++;
        }

        if (now < Turnips::numberOfHalfDays) {
          maxPriceHistogram[*std::max_element(std::begin(prices) + now, std::end(prices))]++;
        }
      }
    }

    i++;
  }

  if (!verbose) {
    std::cout << numberOfMatches << " " << prevPatternCounts << " " << patternCounts << " " <<
        pricesHistogram << " " << maxPriceHistogram << std::endl;
  }
}



int main(int argc, char *argv[]) {
  if (argc < 2) {
    throw std::runtime_error("missing mode");
  }

  const std::string mode(argv[1]);

  if (mode == "compute") {
    // turnips compute <seed> <prevPattern>
    compute(argc, argv);
  } else if (mode == "bruteForce") {
    // turnips bruteForce <numberOfWorkers> <workerId> <prevPattern> <pattern> <prices[0:14]>
    bruteForce(argc, argv);
  } else if (mode == "sample") {
    // turnips sample <sampleMode> <numberOfSamples> <minimumNumberOfMatches> <seed> <prevPattern>
    //     <pattern> <prices[0:14]>
    sample(argc, argv);
  } else {
    throw std::runtime_error("invalid mode");
  }

  return 0;
}
