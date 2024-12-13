#ifndef TRANSPORTSYSTEM_HPP
#define TRANSPORTSYSTEM_HPP

#include "Types.hpp"
#include "DGSoln.hpp"
#include "NetCDFIO.hpp"

/*
	Pure interface class
	defines a problem in the form
		a_i d_t u_i + d_x ( sigma_i ) = S_i( u(x), q(x), x, t ) ; S_i can depend on the entire u & q vector, but only locally.
		sigma_i = sigma_hat_i( u( x ), q( x ), x, t ) ; so can sigma_hat_i

 */

class TransportSystem
{
    public:
        virtual ~TransportSystem() = default;

        Index getNumVars() const { return nVars; };
        Index getNumScalars() const { return nScalars; };
        Index getNumAux() const { return nAux; };

        // Function for passing boundary conditions to the solver
        virtual Value LowerBoundary(Index i, Time t) const { return uL[i]; };
        virtual Value UpperBoundary(Index i, Time t) const { return uR[i]; };

        virtual bool isLowerBoundaryDirichlet(Index i) const { return isLowerDirichlet; };
        virtual bool isUpperBoundaryDirichlet(Index i) const { return isUpperDirichlet; };

        // The same for the flux and source functions -- the vectors have length nVars
        virtual Value SigmaFn(Index i, const State &s, Position x, Time t) = 0;
        virtual Value Sources(Index i, const State &s, Position x, Time t) = 0;

        virtual Values SigmaFn( Index i, std::vector<State> const & states, std::vector<Position> const & abscissae, std::vector<Time> const & times ) {
            Values out( states.size() );
            for( size_t j = 0; j < states.size(); ++j) {
                out( j ) = SigmaFn( i, states[ j ], abscissae[ j ], times[ j ] );
            }
            return out;
        };

        virtual Values Sources( Index i, std::vector<State> const & states, std::vector<Position> const & abscissae, std::vector<Time> const & times ) {
            Values out( states.size() );
            for( size_t j = 0; j < states.size(); ++j) {
                out( j ) = Sources( i, states[ j ], abscissae[ j ], times[ j ] );
            }
            return out;
        };


        // This determines the a_i functions. Only one with a default option, but can be overriden
        virtual Value aFn(Index i, Position x) { return 1.0; };

        // We need derivatives of the flux functions
        virtual void dSigmaFn_du(Index i, Values &, const State &s, Position x, Time t) = 0;
        virtual void dSigmaFn_dq(Index i, Values &, const State &s, Position x, Time t) = 0;

        virtual void dSigma( Index i, std::vector<State> &out, std::vector<State> const & states, std::vector<Position> const & abscissae, std::vector<Time> const& times ) {
            for( size_t j = 0; j < states.size(); ++j) {
                dSigmaFn_du( i, out[ j ].Variable, states[ j ], abscissae[ j ], times[ j ] );
                dSigmaFn_dq( i, out[ j ].Derivative, states[ j ], abscissae[ j ], times[ j ] );
            }
        }

        // and for the sources
        virtual void dSources_du(Index i, Values &, const State &, Position x, Time t) = 0;
        virtual void dSources_dq(Index i, Values &, const State &, Position x, Time t) = 0;
        virtual void dSources_dsigma(Index i, Values &, const State &, Position x, Time t) = 0;

        virtual void dSources( Index i, std::vector<State> &out, std::vector<State> const & states, std::vector<Position> const & abscissae, std::vector<Time> const& times ) {
            for( size_t j = 0; j < states.size(); ++j) {
                dSources_du( i, out[ j ].Variable, states[ j ], abscissae[ j ], times[ j ] );
                dSources_dq( i, out[ j ].Derivative, states[ j ], abscissae[ j ], times[ j ] );
                dSources_dsigma( i, out[ j ].Flux, states[ j ], abscissae[ j ], times[ j ] );
            }
        }
        
        // and initial conditions for u & q
        virtual Value InitialValue(Index i, Position x) const = 0;
        virtual Value InitialDerivative(Index i, Position x) const = 0;

        virtual Value InitialScalarValue(Index s) const
        {
            if (nScalars != 0)
                throw std::logic_error("nScalars > 0 but no initial value provided");
            return 0.0;
        }

        // Only called if you set a scalar to be differential (rather than algebraic)
        virtual Value InitialScalarDerivative(Index s, const DGSoln &y, const DGSoln &dydt) const
        {
            return 0.0;
        }

        // Scalar functions
        virtual Value ScalarG(Index, const DGSoln &, Time)
        {
            if (nScalars != 0)
                throw std::logic_error("nScalars > 0 but no scalar G provided");
            return 0.0;
        }

        virtual Value ScalarGExtended( Index i, const DGSoln &y, const DGSoln &dydt, Time t ) {
            return ScalarG( i, y, t);
        }

        virtual void ScalarGPrime( Index i, State &out, const DGSoln &y, std::function<double( double )> phi, Interval I, Time t ) {
            throw std::logic_error( "nScalars > 0 but no scalar G derivative provided" );
        }


        virtual void ScalarGPrimeExtended( Index i, State &out, State &out_dt, const DGSoln &y, const DGSoln &dydt, std::function<double( double )> phi, Interval I, Time t ) {
            out_dt.zero();
            ScalarGPrime( i, out, y, phi, I, t );
        }

        virtual bool isScalarDifferential( Index i ) {
            return false;
        }

        virtual void dSources_dScalars( Index, Values &, const State &, Position, Time ) {
            if ( nScalars != 0 )
                throw std::logic_error( "nScalars > 0 but no coupling function provided" );
        }

        // Auxiliary variable functions

        virtual Value InitialAuxValue(Index i, Position x) const
        {
            if (nAux != 0)
                throw std::logic_error("nAux > 0 but no initial auxiliary value provided");
            return 0.0;
        }

        // G_i( a(x), {u_j(x), q_j(x), sigma_j(x)} , x ) = 0 is the equation
        // that defines the auxiliary variable a
        virtual Value AuxG(Index i, const State &, Position, Time)
        {
            if (nAux != 0)
                throw std::logic_error("nAux > 0 but no auxiliary G provided");
            return 0.0;
        }

        // AuxGPrime returns dG_i in out
        virtual void AuxGPrime(Index i, State &out, const State &, Position, Time)
        {
            throw std::logic_error("nAux > 0 but no G derivative provided");
        }

        virtual void dSources_dPhi(Index, Values &, const State &, Position, Time)
        {
            if (nAux != 0)
                throw std::logic_error("nAux > 0 but no coupling to the main sources provided");
        }

        virtual void dSigma_dPhi(Index, Values &v, const State &, Position, Time)
        {
          v.setZero();
          return;
        }


        virtual std::string getVariableName(Index i)
        {
            return std::string("Var") + std::to_string(i);
        }

        virtual std::string getScalarName(Index i)
        {
            return std::string("Scalar") + std::to_string(i);
        }

        virtual std::string getAuxVarName(Index i)
        {
            return std::string("AuxVariable") + std::to_string(i);
        }

        virtual std::string getVariableDescription(Index i)
        {
            return std::string("Variable ") + std::to_string(i);
        }

        virtual std::string getScalarDescription(Index i)
        {
            return std::string("Scalar ") + std::to_string(i);
        }

        virtual std::string getAuxDescription(Index i)
        {
            return std::string("Auxiliary Variable ") + std::to_string(i);
        }

        virtual std::string getVariableUnits(Index i)
        {
            return std::string("");
        }

        virtual std::string getScalarUnits(Index i)
        {
            return std::string("");
        }

        virtual std::string getAuxUnits(Index i)
        {
            return std::string("");
        }

        // Hooks for adding extra NetCDF outputs
        virtual void initialiseDiagnostics(NetCDFIO &)
        {
            return;
        }

        // Parameters are ( solution, time, netcdf output object, time index )
        virtual void writeDiagnostics(DGSoln const &, double, NetCDFIO &, size_t)
        {
            return;
        }

        virtual void finaliseDiagnostics(NetCDFIO &)
        {
            return;
        }

        std::map<int, std::string> subVars = {{0, "u"}, {1, "q"}, {2, "sigma"}, {3, "S"}};

    protected:
        Index nVars;
        Index nScalars = 0;

        Index nAux = 0;
        std::vector<Value> uL, uR;
        bool isUpperDirichlet, isLowerDirichlet;
};

#endif // TRANSPORTSYSTEM_HPP
