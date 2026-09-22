Quickstart
==========

Mission dose report
-------------------

.. code-block:: bash

   # 500-day Mars mission starting 2025-06-01, 35-year-old female
   gcrisk-dose mission 2025-06-01 --surface-days 500 --age 35 --sex female

   # JSON output for downstream analysis
   gcrisk-dose mission 2025-06-01 --surface-days 500 --json > mission.json

Proton therapy RBE
------------------

.. code-block:: bash

   # Wedenberg RBE for 5 keV/µm LET, α/β = 3 Gy
   gcrisk-dose rbe --dose-gy 2.0 --letd-kev-um 5.0 --alpha-beta-gy 3.0 --model wedenberg

   # McNamara RBE
   gcrisk-dose rbe --dose-gy 2.0 --letd-kev-um 5.0 --alpha-beta-gy 3.0 --model mcnamara

Python API
----------

.. code-block:: python

   from gcrisk.mission import run_full_mission
   from gcrisk.sep import sep_event_dose, sep_mission_probability

   # Full Mars mission with organ-routed REID
   result = run_full_mission('2025-06-01', surface_days=500, age=35, sex='female')
   print(f"Total dose: {result['total']['D_total_mGy']:.0f} mGy")

   # SEP event risk
   sep = sep_event_dose('aug1972', shielding_x_gcm2=10.0)
   print(f"Aug 1972 BFO dose at 10 g/cm² Al: {sep['D_BFO_mGy']:.0f} mGy")
   print(f"Exceeds limit: {sep['exceeds_acute_limit']}")

   # 500-day mission SEP encounter probability
   prob = sep_mission_probability(500)
   print(f"P(>=1 large SEP event): {prob['P_one_or_more']:.1%}")
