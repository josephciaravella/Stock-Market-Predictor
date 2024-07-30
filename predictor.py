# Make sure that you have all these libaries available to run the code successfully
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler 



def download_ticker_data(data_source, ticker):

    if data_source == 'alphavantage':
        # ====================== Loading Data from Alpha Vantage ==================================

        api_key = '13DF5UCH7N8YOYCB'

        # JSON file with all the stock market data for AAL from the last 20 years
        url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)

        # Save data to this file
        file_to_save = 'ticker_data/stock_market_data-%s.csv'%ticker

        # If you haven't already saved data,
        # Go ahead and grab the data from the url
        # And store date, low, high, volume, close, open values to a Pandas DataFrame
        if not os.path.exists(file_to_save):
            with urllib.request.urlopen(url_string) as url:
                data = json.loads(url.read().decode())
                # extract stock market data
                data = data['Time Series (Daily)']
                df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
                for k,v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d')
                    data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                                float(v['4. close']),float(v['1. open'])]
                    df.loc[-1,:] = data_row
                    df.index = df.index + 1
            print('Data saved to : %s'%file_to_save)        
            df.to_csv(file_to_save)


        # If the data is already there, just load it from the CSV
        else:
            print('File already exists. Loading data from CSV')
            df = pd.read_csv(file_to_save)


    else:

        # ====================== Loading Data from Kaggle ==================================
        # You will be using HP's data. Feel free to experiment with other data.
        # But while doing so, be careful to have a large enough dataset and also pay attention to the data normalization
        df = pd.read_csv(os.path.join('Stocks','hpq.us.txt'),delimiter=',',usecols=['Date','Open','High','Low','Close'])
        print('Loaded data from the Kaggle repository') 

    return df

def display_ticker_data(ticker):

    df = download_ticker_data('alphavantage', ticker)

    # Sort DataFrame by date
    df = df.sort_values('Date')

    # Double check the result
    df.head()

    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
    plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.show()

    return df

display_ticker_data("AMD")


# def split_data(ticker):

#     df = download_ticker_data("alphavantage", ticker)

#     # First calculate the mid prices from the highest and lowest
#     high_prices = df.loc[:,'High'].as_matrix()
#     low_prices = df.loc[:,'Low'].as_matrix()
#     mid_prices = (high_prices+low_prices)/2.0

#     # This is a common technique used in machine learning to split a dataset into a training set and a testing set.
#     # The training set is used to train a model, while the testing set is used to evaluate the performance of the model on unseen data.
#     train_data = mid_prices[:11000]
#     test_data = mid_prices[11000:]

#     # Scale the data to be between 0 and 1
#     # When scaling remember! You normalize both test and train data with respect to training data
#     # Because you are not supposed to have access to test data
#     scaler = MinMaxScaler()

#     # The code then reshapes the training and test data to be a single column using the reshape() function with the argument -1, which means that the number of rows is inferred from the length of the array and the number of columns is set to 1.
#     # This is necessary because the MinMaxScaler function expects a 2D array as input.
#     train_data = train_data.reshape(-1,1)
#     test_data = test_data.reshape(-1,1)

#     # Train the Scaler with training data and smooth data
#     smoothing_window_size = 2500
#     for di in range(0,10000,smoothing_window_size):
#         # Within each iteration, the fit method of the scaler object is called on a slice of the training data, from di to di+smoothing_window_size.
#         # This trains the scaler on that portion of the data.
#         scaler.fit(train_data[di:di+smoothing_window_size,:])

#         # Then, the transform method of the scaler is called on the same slice of the training data, which normalizes the data using the parameters learned during the fit step.
#         # The normalized data is then assigned back to the same slice of the training data.
#         train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

#     # You normalize the last bit of remaining data
#     scaler.fit(train_data[di+smoothing_window_size:,:])
#     train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

#     # Reshape both train and test data
#     train_data = train_data.reshape(-1)

#     # Normalize test data
#     test_data = scaler.transform(test_data).reshape(-1)
