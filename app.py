import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from keras.models import load_model
import streamlit as st
yf.pdr_override()

st.set_page_config(
    page_title="Stock Trend Prediction",
    # layout="wide",
    # initial_sidebar_state="expanded",
)


st.title('Stock Trend Prediction')
user_input=st.text_input('Enter the stock ticker','AAPL')
user_input_sd=st.text_input('Enter the start date','2010-01-01')
user_input_ed=st.text_input('Enter the end date','2024-01-01')
data=pdr.get_data_yahoo(user_input,user_input_sd,user_input_ed)

st.subheader('Data from {} - {}'.format(user_input_sd,user_input_ed))
st.write(data.describe())

st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100=data.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(data.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 200MA and 200MA')
ma100=data.Close.rolling(100).mean()
ma200=data.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(data.Close,'b')
st.pyplot(fig)

split_index = int(len(data) * 0.70)

# Create training and testing datasets
data_train = pd.DataFrame(data['Close'][:split_index])
data_test = pd.DataFrame(data['Close'][split_index:])
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)


#load my model
model = load_model('my_model3.keras')

#testing
past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)
input_data=scaler.fit_transform(final_df)
x_test =[]
y_test =[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)
y_pred=model.predict(x_test)
scaler=scaler.scale_

scale_factor=1/scaler[0]
y_pred=y_pred*scale_factor
y_test=y_test*scale_factor

#final graph
st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12,6))
# plt.plot(y_test,'b',label ='Original Price')
# plt.plot(y_pred,'r',label='Predicted Price')
test_dates = data.index[split_index:]

plt.plot(test_dates, y_test, 'b', label ='Original Price')
plt.plot(test_dates, y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader('Evaluation Metrics')
st.write(f'Mean Squared Error: {mse}')
st.write(f'Root Mean Squared Error: {rmse}')
st.write(f'Mean Absolute Error: {mae}')
st.write(f'R-squared: {r2}')

if y_pred[-1] > y_test[-1]:
    st.subheader('Investment Advice')
    st.write('The predicted price is higher than the actual price. It might be a good time to invest.')
else:
    st.subheader('Investment Advice')
    st.write('The predicted price is lower than the actual price. It might not be the best time to invest.')