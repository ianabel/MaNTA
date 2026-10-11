#include "Config.hpp"

#include "Logging.hpp"

#include <stdexcept>

// TOML distinguishes 1 from 1.0. A config author writing `tau = 1` means the
// number one, so accept integer nodes for float-valued options -- but read them
// with as_integer(). Calling as_floating() on an integer node throws
// toml::type_error, which is what these functions used to do.

double getFloatWithDefault(std::string const &name, toml::value const &config, double defaultValue)
{
	auto confCount = config.count(name);
	if (confCount == 0)
	{
		logmsg<LOG_LEVEL::INFO>("Using default value {} for configuration option {}", defaultValue, name);
		return defaultValue;
	}
	else if (confCount > 1)
	{
		throw std::invalid_argument(name + " was multiply specified.");
	}

	auto configElement = toml::find(config, name);

	if (configElement.is_integer())
		return static_cast<double>(configElement.as_integer());
	else if (configElement.is_floating())
		return static_cast<double>(configElement.as_floating());
	else
		throw std::invalid_argument(name + " specified incorrectly");
}

double getFloat(std::string const &name, toml::value const &config)
{
	auto confCount = config.count(name);
	if (confCount == 0)
		throw std::invalid_argument(name + " was not specified.");
	else if (confCount > 1)
		throw std::invalid_argument(name + " was multiply specified.");

	auto configElement = toml::find(config, name);
	if (configElement.is_integer())
		return static_cast<double>(configElement.as_integer());
	else if (configElement.is_floating())
		return static_cast<double>(configElement.as_floating());
	else
		throw std::invalid_argument(name + " specified incorrectly");
}

int getIntWithDefault(std::string const &name, toml::value const &config, int defaultValue)
{
	auto confCount = config.count(name);
	if (confCount == 0)
	{
		logmsg<LOG_LEVEL::INFO>("Using default value {} for configuration option {}", defaultValue, name);
		return defaultValue;
	}
	else if (confCount > 1)
	{
		throw std::invalid_argument(name + " was multiply specified.");
	}

	auto configElement = toml::find(config, name);
	if (configElement.is_integer())
		return static_cast<int>(configElement.as_integer());
	else
		throw std::invalid_argument(name + " specified incorrectly");
}
