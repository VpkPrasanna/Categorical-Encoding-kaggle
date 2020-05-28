from sklearn import preprocessing


class Encoder:
    def __init__(self, df, cat_features, encoding_type, handle_na=False):
        self.df = df
        self.encoding_type = encoding_type
        self.catFeatures = cat_features
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict

        if self.handle_na:
            for c in self.catFeatures:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True)

    def labelEncoder(self):
        for i in self.catFeatures:
            lb = preprocessing.LabelEncoder()
            lb.fit(self.df[i].values)
            self.output_df.loc[:, i] = lb.transform(self.df[i].values)
            self.label_encoders[i] = lb
        return self.output_df

    def labelBinarizer(self):
        for c in self.catFeatures:
            labelBin = preprocessing.LabelBinarizer()
            labelBin.fit(self.df[c].values)
            val = labelBin.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin__{j}"
                self.output_df[new_col_name] = val[:, j]
                self.binary_encoders = labelBin
        return self.output_df

    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.catFeatures].values)
        return ohe.transform(self.df[self.catFeatures].values)

    def transform(self):
        if self.encoding_type == "label":
            return self.labelEncoder()
        elif self.encoding_type == "binary":
            return self.labelBinarizer()
        elif self.encoding_type == "ohe":
            return self._one_hot()
        else:
            raise Exception("Choose the correct encoding type")
