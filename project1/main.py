import datas
import config
import encoding
import pandas as pd

train_path = config.train_data_path
test_path = config.test_data_path

train_data = datas.read_data(train_path)
test_data = datas.read_data(test_path)

test_data["target"] = -1
final_data = pd.concat([train_data, test_data])
print(final_data.head())
categorical_column = [c for c in train_data.columns if c not in ["id", "target"]]
print(categorical_column)
encoded_data = encoding.Encoder(df=final_data, cat_features=categorical_column, encoding_type="ohe", handle_na=True)
transformed_data = encoded_data.transform()
print(transformed_data)
