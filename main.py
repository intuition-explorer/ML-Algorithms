from src.preprocessing import load_data
data = load_data()

df = data["EEG_Eye_State_Classification"]
print(df.head(10))