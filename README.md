# GEOCLIM-DynSoil Steady-State Weathering (Python)

`geoclim_steady_state_573.py` is a Python reimplementation of the forward steady-state silicate weathering calculation of the Fortran model [GEOCLIM-dynsoil-steady-state](https://github.com/piermafrost/GEOCLIM-dynsoil-steady-state).

## Usage

python geoclim_steady_state_573.py
The 573 parameter sets are embedded in the script. The erosion coefficient is set by the constant `KE` at the top of the script.

## Outputs
`A_weathering_flux_result.nc`: weathering rate for each grid cell  and global flux for each parameter set.
`A_results_summary.csv`: global flux and parameter values for each set.

## Notes
 Only the forward mode is implemented, using a single climate field and globally uniform lithology fractions. CO₂ interpolation and the equilibrium-pCO₂ solver of the Fortran model are not included.

- Lithology: no lithology file is read. Every land grid cell uses the same global lithology fractions, set by `FIXED_LITHOLOGY_FRACTIONS` in the script: metamorphic 14.4%, felsic 7.4%, intermediate 2.3%, mafic 5.3%, carbonate 9.0%, siliciclastic sediment 61.6%. 
