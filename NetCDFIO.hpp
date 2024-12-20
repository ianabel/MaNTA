#ifndef NETCDFIO_HPP
#define NETCDFIO_HPP

#include <netcdf>
#include <string>
#include <vector>
#include <utility>

#include "gridStructures.hpp"

/*
 * Class for storing MaNTA run details in a NetCDF file.
 */

class NetCDFIO
{
public:
	NetCDFIO();
	~NetCDFIO();
	void Open(const std::string &file);
	void Close();
	void AddScalarVariable(std::string name, std::string description, std::string units, double value);
	void AddTextVariable(std::string name, std::string description, std::string units, std::string text);

	void SetOutputGrid(std::vector<double> const &gridpoints);

	void AddTimeSeries(std::string name, std::string description, std::string units, double initialValue);
	void AddTimeSeries(std::string groupName, std::string name, std::string description, std::string units, double initialValue);

	template <typename T>
	void AddVariable(std::string name, std::string description, std::string units, T const &initialValue)
	{
		netCDF::NcVar newvar = data_file.addVar(name, netCDF::NcDouble(), {TimeDim, SpaceDim});
		newvar.putAtt("description", description);
		if (units != "")
			newvar.putAtt("units", units);
		std::vector<double> gridValues;
		gridValues.resize(gridpoints.size());
		for (size_t i = 0; i < gridpoints.size(); ++i)
			gridValues[i] = initialValue(gridpoints[i]);

		newvar.putVar({0, 0}, {1, gridpoints.size()}, gridValues.data());
	}

	template <typename T>
	void AddVariable(std::string groupName, std::string name, std::string description, std::string units, T const &initialValue)
	{
		netCDF::NcGroup group = data_file.getGroup(groupName);
		netCDF::NcVar newvar = group.addVar(name, netCDF::NcDouble(), {TimeDim, SpaceDim});
		newvar.putAtt("description", description);
		if (units != "")
			newvar.putAtt("units", units);
		std::vector<double> gridValues;
		gridValues.resize(gridpoints.size());
		for (size_t i = 0; i < gridpoints.size(); ++i)
			gridValues[i] = initialValue(gridpoints[i]);

		newvar.putVar({0, 0}, {1, gridpoints.size()}, gridValues.data());
	}

	void AddGroup(std::string name, std::string description);

	size_t AddTimeSlice(double T);

	void AppendToTimeSeries(std::string const &name, double value, size_t tIndex);

	void AppendToTimeSeries(std::string const &groupName, std::string const &name, double value, size_t tIndex);

	template <typename T>
	void AppendToVariable(std::string const &name, T const &var, size_t tIndex)
	{
		netCDF::NcVar variable = data_file.getVar(name);
		std::vector<double> gridValues;
		gridValues.resize(gridpoints.size());
		for (size_t i = 0; i < gridpoints.size(); ++i)
			gridValues[i] = var(gridpoints[i]);

		variable.putVar({tIndex, 0}, {1, gridpoints.size()}, gridValues.data());
	}

	template <typename T>
	void AppendToGroup(std::string const &name, size_t tIndex, const std::initializer_list<std::pair<std::string, T const &>> &vars)
	{

		std::vector<double> gridValues;
		gridValues.resize(gridpoints.size());

		netCDF::NcGroup group = data_file.getGroup(name);
		for (auto &var : vars)
		{
			for (size_t i = 0; i < gridpoints.size(); ++i)
				gridValues[i] = var.second(gridpoints[i]);

			group.getVar(var.first).putVar({tIndex, 0}, {1, gridpoints.size()}, gridValues.data());
		}
	}

	template <typename T>
	void AppendToGroup(std::string const &name, size_t tIndex, std::string varname, T const &var)
	{

		std::vector<double> gridValues;
		gridValues.resize(gridpoints.size());

		netCDF::NcGroup group = data_file.getGroup(name);

		for (size_t i = 0; i < gridpoints.size(); ++i)
			gridValues[i] = var(gridpoints[i]);

		group.getVar(varname).putVar({tIndex, 0}, {1, gridpoints.size()}, gridValues.data());
	}

	void StoreGridInfo(const Grid &, unsigned int);

private:
	std::string filename;
	std::vector<double> gridpoints;
	netCDF::NcFile data_file;
	netCDF::NcDim TimeDim;
	netCDF::NcVar TimeVar;
	netCDF::NcDim SpaceDim;
	netCDF::NcVar SpaceVar;
};

#endif // NETCDFIO_HPP
