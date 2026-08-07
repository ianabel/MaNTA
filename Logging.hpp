#ifndef LOGGING_HPP
#define LOGGING_HPP
#include <cstdio>
#include <print>
#include <stdexcept>
#include <string_view>
//
// #if defined(__GNUC__) && (__GNUC__ < 15)
//
// template <> struct std::formatter<std::vector<double>> {
//   // Parse format specifications
//   constexpr auto parse(std::format_parse_context &ctx) {
//     return ctx.begin(); // or parse custom specs
//   }
//
//   // Format the vector contents
//   auto format(const std::vector<double> &vec, std::format_context &ctx) const
//   {
//     auto out = ctx.out();
//     std::format_to(out, "[");
//     bool first = true;
//     for (double val : vec) {
//       if (!first)
//         std::format_to(out, ", ");
//       std::format_to(out, "{}", val);
//       first = false;
//     }
//     return std::format_to(out, "]");
//   }
// };
// #endif
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

/// @brief Logs a message using std::format. Level specified as a template
/// parameter
/// @tparam ...Args
/// @tparam LEVEL
/// @param msg
/// @param ...args
template <LOG_LEVEL LEVEL, typename... Args>
inline void logmsg(std::format_string<Args...> msg, Args &&...args) {
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
#endif // LOGGING_HPP
