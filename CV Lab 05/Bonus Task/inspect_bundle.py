import time
import numpy as np
import joblib
import hog_utils

print("Loading model bundle...")
t0 = time.time()
bundle = joblib.load('hog_defect_model.joblib')
print(f"Loaded bundle in {time.time()-t0:.2f}s")

img = np.full((200, 200, 3), 128, dtype=np.uint8)
t0 = time.time()
res = hog_utils.inspect_product(img, bundle)
dt = time.time() - t0

print(f"Inference on 200x200 image in {dt*1000:.1f}ms:")
print("Prediction:", res["prediction"])
print(f"Confidence: {res['confidence']*100:.1f}%")
print("Action:", res["action"])
print("Total sliding windows:", len(res["coords"]))
print("Defective windows:", int(res["positive"].sum()))
