#ifndef LOGGING_HPP
#define LOGGING_HPP
#include <cstdio>
#include <print>
#include <stdexcept>
#include <string_view>

namespace Logging {
enum class LOG_LEVEL {
  ERROR,
  WARNING,
  INFO,
  PDEBUG,
};

constexpr std::string_view levelToString(LOG_LEVEL level) {
  switch (level) {
  case LOG_LEVEL::ERROR:
    return "ERROR";
  case LOG_LEVEL::WARNING:
    return "WARNING";
  case LOG_LEVEL::INFO:
    return "INFO";
  case LOG_LEVEL::PDEBUG: // Named this to not conflict with DEBUG preprocessor
                          // macro
    return "DEBUG";
  default:
    throw std::invalid_argument("Invalid logging level");
  }
};
#ifdef DEBUG
constexpr static LOG_LEVEL max_log_level = LOG_LEVEL::PDEBUG;
#elif defined(VERBOSE)
constexpr static LOG_LEVEL max_log_level = LOG_LEVEL::INFO;
#else
constexpr static LOG_LEVEL max_log_level = LOG_LEVEL::WARNING;
#endif

template <LOG_LEVEL LEVEL, typename... Args>
inline void log(std::format_string<Args...> msg, Args &&...args) {
  if constexpr (LEVEL <= max_log_level) {
    if constexpr (LEVEL == LOG_LEVEL::ERROR) {
      std::print(stderr, "{}: ", levelToString(LEVEL));
      std::println(stderr, msg, std::forward<Args>(args)...);
    } else if constexpr (LEVEL == LOG_LEVEL::WARNING) {
      std::print(stderr, "{}: ", levelToString(LEVEL));
      std::println(stderr, msg, std::forward<Args>(args)...);
    } else if constexpr (LEVEL == LOG_LEVEL::INFO) {
      std::print("{}: ", levelToString(LEVEL));
      std::println(msg, std::forward<Args>(args)...);
    } else if constexpr (LEVEL == LOG_LEVEL::PDEBUG) {
#ifdef DEBUG
      std::print(stderr, "{}: ", levelToString(LEVEL));
      std::println(stderr, msg, std::forward<Args>(args)...);
#endif
    } else {
      throw std::invalid_argument("Invalid log level specified.");
    }
  }
}
} // namespace Logging
#endif // LOGGING_HPP
