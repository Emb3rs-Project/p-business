from ..Businessmodulev1_clean import BM
import json
# --------------------------------------------------------------------------
#                         INPUTS START
# Note: subscript i correspond to actors, h hours, t to each technology component.
# correspondence of i to t should be mapped i.e i = 0 ; t= 0,1,4
# ---------------------------------------------------------------------------


def test_bm():
    f = open('market.json')
    market = json.load(f)
    t = open('teo.json')
    teo = json.load(t)
    BM_input_dict = {
        # MWh (per actor per hour for a whole year), int, 2D array
        "market-module": market,
        "teo-module": teo,
        # important connects actors (first col) with different techs (second col)
        "platform": {
            "rls": [ ["source 1", "source 1 ext tech"],
                     ["source 2", "source 2 ext tech"],
                     ["source 3", "source 3 ext tech"],
                     ["source 3", "source 4 ext tech"],
                     ["sink 5", "sink 5 ext tech"],
                     ["sink 6", "sink 6 ext tech"],
                     ["sink 7", "sink 7 ext tech"],
                     ["sink 8", "sink 8 ext tech"],
                     ["sink 9", "sink 9 ext tech"],
                     ["sink 10", "sink 10 ext tech"],
                     ["sink 10", "sink 11 ext tech"],
                        ],
            "projectduration": 5,
            "discountrate_i": [3, 5.5],
            "actorshare": [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0 ],
            "co2_itensity": 0,
        },
        "gis-module": {
            "net_cost": [100],
        },
    }


    BM_output = BM(BM_input_dict)

    print(BM_output)

#test_bm()