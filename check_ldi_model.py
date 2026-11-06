import pickle, numpy as np
s = pickle.load(open("cvpr_2027_ssl_results_main_trail_ldis/results_ldi.pkl", "rb"))
for ds, hist in s["runs"]["vista"].items():
    c = hist[-1]["stop_confusion"]
    print(ds, "\n", (c / c.sum(1, keepdims=True).clip(1)).round(2))   # rows = GT class, normalized