#pragma once

#include <string>
#include "Constants.hpp"

typedef struct Species_t {
	enum Type {
		Electron,
		Ion,
		TraceImpurity,
		Neutral
	} type;
	double Charge; // Units of e
	double Mass; // kg
	std::string Name; // For reporting
} Species;

