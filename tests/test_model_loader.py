import dataAnalyzer
model = dataAnalyzer.load_model("research_data/0001_2_308min/models/binary_DT.joblib")
print(model.X[:5])
print(model.trackData[0].history)