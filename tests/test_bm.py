from ..Businessmodulev1_clean import BM

# --------------------------------------------------------------------------
#                         INPUTS START
# Note: subscript i correspond to actors, h hours, t to each technology component.
# correspondence of i to t should be mapped i.e i = 0 ; t= 0,1,4
# ---------------------------------------------------------------------------

"""
           %From MM%
dispatch_ih :: MWh (per actor per hour for a whole year), int, 2D array
price_h :: EUR/MWh (per actor per hour for a whole year) float, 1D array
opcost_ih :: EUR (per hour per actor for a whole year), float, 2D array
 -> OR if no op_cost is given; then
co2taxtot_ih :: EUR (per hour per actor for a whole year), float, 2D array -> optional (not implemented yet)
fuelcost_ih :: EUR (per hour per actor for a whole year), float, 2D array -> optional (not implemented yet)
           %From TEO%
capex_t  :: EUR, int, 1D array
opex_t   :: EUR/year, int, 1D array
           %From KB%
projectduration :: int
discountrate_i :: float, 1D array with two elements (first socio-economic discount rate, second private one - simplified right now, doesn't take into account different discount rates per tech and/or actor)
emissionfactortech_t :: float CO2e Kg/MWh, 1D array -> optional (not implemented yet)
heatsupplyco2ent :: floeat CO2e Kg/MWh -> optional (not implemented yet)
            %From User / Platform%
rls :: important connects actors (first col) with different techs (second col) - ownership structure
"""


def test_bm():
    BM_input_dict = {
        # MWh (per actor per hour for a whole year), int, 2D array
        "dispatch_ih": [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [20, 30, 40, 50, 60]],
        # EUR/MWh (per actor per hour for a whole year) float, 1D array
        "price_h": [1, 2, 3, 4, 5],
        # EUR (per hour per actor for a whole year), float, 2D array
        "opcost_ih": [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5], [2, 30, 40, 5, 60]],
        "capex_t": [100, 200, 300, 400, 500, 600, 700],  # EUR, int, 1D array
        "opex_t": [10, 20, 30, 40, 50, 60, 70],  # EUR/year, int, 1D array
        "projectduration": 5,
        "discountrate_i": [3, 5.5],
        # important connects acors (first col) with different techs (second col)
        "rls": [[0, 0], [0, 2], [1, 1], [1, 3], [1, 6], [2, 4], [2, 5]],
        "sinks" : [0],
    }

    BM_output = BM(BM_input_dict)

    print(BM_output)

