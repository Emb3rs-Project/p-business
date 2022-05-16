from ..Businessmodulev1_clean import BM

# --------------------------------------------------------------------------
#                         INPUTS START
# Note: subscript i correspond to actors, h hours, t to each technology component.
# correspondence of i to t should be mapped i.e i = 0 ; t= 0,1,4
# ---------------------------------------------------------------------------


def test_bm():
    BM_input_dict = {
        # MWh (per actor per hour for a whole year), int, 2D array
        "market-module": {
            "dispatch_ih": [
                [-1, -2, -3, -4, -5],
                [10, 20, 30, 40, 50],
                [20, 30, 40, 50, 60],
            ],
            # EUR/MWh (per actor per hour for a whole year) float, 1D array
            "price_h": [1, 2, 3, 4, 5],
            # EUR (per hour per actor for a whole year), float, 2D array
            "opcost_i": [2, 30, 40],
        },
        "teo-module": {
            "capex_tt": [100, 200, 300, 400, 500, 600, 700],  # EUR, int, 1D array
            "opex_tt": [10, 20, 30, 40, 50, 60, 70],  # EUR total, int, 1D array
            "capex_st": [100, 200],
            "capex_s_names": ["s1", "s2"],
            "capex_t_names": ["100", "200", "300", "400", "500", "600", "700"],
            "sal_tt": [10, 20, 30, 40, 50, 60, 70],
            "sal_st": [10, 20],
        },
        # important connects actors (first col) with different techs (second col)
        "platform": {
            "rls": [[0, 0], [0, 2], [1, 1], [1, 3], [1, 6], [2, 4], [2, 5], [2, 7], [2, 8]],
            "sinks": [0],
            "projectduration": 5,
            "discountrate_i": [3, 5.5],
            "actorshare": [0, 0.50, 0.50],
            "co2_itensity": 0,
        },
        "gis-module": {
            "net_cost": [100],
        },
    }

    BM_output = BM(BM_input_dict)

    print(BM_output)
