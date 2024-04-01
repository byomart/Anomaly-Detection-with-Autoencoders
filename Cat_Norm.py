from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import logging

class Cat_Norm():
    def __init__(self):
        self.encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        df_encoded = None

    def Cat(self,df):
        self.df_encoded = df.copy() 
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df_encoded[col] = self.encoder.fit_transform(df[col])
        
    def Norm(self,df):
        numeric_cols = df.select_dtypes(include=['int64', 'float64'])
        for col in numeric_cols:
            self.df_encoded[col] = self.scaler.fit_transform(df[col].values.reshape(-1, 1))
            
        logging.info('encoded and normalized df')
        logging.info(self.df_encoded.head())

        return self.df_encoded