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
            
        return self.df_encoded
        
    def Norm(self,df):
        cols_to_normalize = df.columns.difference(['attack_cat'])
        self.df_encoded[cols_to_normalize] = df[cols_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        logging.info('encoded and normalized df')
        logging.info(self.df_encoded.head())

        return self.df_encoded