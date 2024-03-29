from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class Cat_Norm():
    def __init__(self):
        self.encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        
    def Cat(self,df):
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = self.encoder.fit_transform(df[col])
        
    def Norm(self,df):
        numeric_cols = df.select_dtypes(include=['int64', 'float64'])
        for col in numeric_cols:
            df[col] = self.scaler.fit_transform(df[col].values.reshape(-1, 1))
        return df