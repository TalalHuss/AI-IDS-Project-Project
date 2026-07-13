import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# load the dataset

data = pd.read_csv("CICIDS2017.csv")

# remove missing values

data = data.dropna()

#remove duplicate rows

data = data.drop_duplicates()

#encode attack labels

encoder = LabelEncoder()
data["Label"] = encoder.fit_transform(data["Label"])

# Split features and labels

X = data.drop("Label", axis=1)
y = data["Label"]

# Split into training & testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
