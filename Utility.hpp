#ifndef UTILITY_HPP
#define UTILITY_HPP

/*
	Utility Functions for writing Physics Cases
 */

#include <netcdf>
#include <string>
#include <vector>

#include <boost/math/interpolators/makima.hpp>
using Interpolant = boost::math::interpolators::makima::makima;

// Loads f(x) from a netcdf file
Interpolant&& LoadFunctionFromFile( std::string const& fname, std::string const& dim_name, std::string const& var_name )
{
	std::vector<double> x_vals,f_vals;
	netCDF::NcRoot data_file( fname, netCDF::NcFile::FileMode::read );

	NcVar x_var,f_var;

	x_var = data_file.getVar( dim_name );
	if( x_var.isNull() )
		throw std::runtime_error("No variable with name '" + dim_name + "' was found.");
	f_var = data_file.getVar( var_name );
	if( f_var.isNull() )
		throw std::runtime_error("No variable with name '" + var_name + "' was found.");

	NcDim x_dim = data_file.getDim( dim_name );
	if( x_dim.isNull() )
		throw std::runtime_error("No dimension with name '" + dim_name + "' was found.");

	size_t nPoints = x_dim.getSize();
	x_vals.resize( nPoints );
	f_vals.resize( nPoints );
	x_var.getVar( x_vals.data() );
	f_var.getVar( f_vals.data() );

	data_file.close();
	return Interpolant( std::move(x_vals), std::move(f_vals) );
}

#endif // UTILITY_HPP
