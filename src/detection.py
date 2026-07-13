from model import model
from preprocessing import X_test

# Predict first network record
result = model.predict([X_test.iloc[0]])

#Display result
if result[0] == 0:
  print("Normal Traffic")
else:
  print("Attack Detected")
