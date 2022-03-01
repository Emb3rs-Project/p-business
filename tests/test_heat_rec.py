from ..Businessmodulev1_clean import int_heat_rec


def test_int_heat():

    heat_rec_input_dict = {
        "cf-module": {
            "capex": 100,  # total capex inv for internal heat recovery, eur/ number float/integer
            "O&M_fix": 20,  # fix o&m cost eur per year / number float/integer
            "energy_dispatch": 2000,  # total dispatch in a given year /number float/integer
            "discount_rate": 4,  # discount rate / float
            "money_sav": 300,  # energy saved in eur per year due to dispatch/ number, integer/float
            "carbon_sav_quant": 200,  # yearly carbon savings kg/MWh / number, integer/float
            #       "carbon_sav_money": 200, # yearly carbon savings eur / number, integer/float !!! not implemented
            "duration": 9,  # project duration / number, integer
        },
    }
    int_heat_output = int_heat_rec(heat_rec_input_dict)
    print(int_heat_output)
