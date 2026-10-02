Scripts accompanying GA_STT_HARDCLIP_STANDALONE.md

Setup: clone https://github.com/ForbiddenLord/ga_v5.git, then from a Python
session with that repo's ga_stt_py/lib/ on sys.path (edit the sys.path.insert
line at the top of ga_stt_sat.py to point at your clone), run any script
directly, e.g.:

    python3 verify_R1_final.py
    python3 test_closedloop_final.py

Files:
  ga_stt_sat.py                   - the one modification (saturated Stage R)
  sym_lemmas.py                   - Lemma 1 symbolic check
  num_energy_and_gain.py          - Lemma 1 numeric check on coupled sim
  num_M.py                        - Lemma 2 (thrust-direction geometry)
  fix_sign_check.py               - Lemma 3 (corrected-sign ground truth)
  verify_R1_final.py              - Lemma R1 (exhaustive global scan)
  num_mismatch.py                 - Lemma 4 (translation-rotation coupling)
  check_sat_Delta.py              - Lemma 5/6 bulk, theta-driven part
  verify_deeply_saturated_A1.py   - Lemma 5/6 bulk, z-driven part
  test_closedloop_final.py        - full closed-loop validation table
