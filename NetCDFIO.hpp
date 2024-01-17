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

	template <typename T>
	void AddVariable(std::string name, std::string description, std::string units, T const &initialValue);

	template <typename T>
	void AddVariable(std::string groupName, std::string name, std::string description, std::string units, T const &initialValue);

	void AddGroup(std::string name, std::string description);

	size_t AddTimeSlice(double T);

	void AppendToTimeSeries(std::string const &name, double value, size_t tIndex);
	template <typename T>
	void AppendToVariable(std::string const &name, T const &var, size_t tIndex);

	template <typename T>
	void AppendToGroup(std::string const &name, size_t tIndex, const std::initializer_list<std::pair<std::string, T>> &vars);

	void StoreGridInfo( const Grid&, unsigned int );

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
