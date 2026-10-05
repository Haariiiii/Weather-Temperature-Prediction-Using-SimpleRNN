# 🌤️ Weather Temperature Prediction Using SimpleRNN

A deep learning project that uses a **Simple Recurrent Neural Network (SimpleRNN)** to predict temperature from historical weather data.

The model uses the previous **7 time steps** of temperature, humidity, and wind speed to predict the temperature at the next time step. The project also performs **7-step future temperature forecasting** using recursive predictions.

---

## 📌 Project Overview

Weather data is naturally sequential because weather conditions at one point in time are related to conditions from previous time steps.

In this project, a **SimpleRNN** is trained to learn patterns from historical weather observations and predict future temperature values.

### Main objectives

* Explore historical weather data
* Visualize temperature trends
* Select relevant weather features
* Normalize numerical features
* Create sequential input data
* Train a SimpleRNN model
* Evaluate the model using RMSE, MAE, and R²
* Compare actual and predicted temperatures
* Forecast temperature for the next 7 time steps

---

## 📊 Dataset

The project uses a weather history dataset containing **24,508 rows and 12 columns**.

### Dataset features

| Feature                    | Description                  |
| -------------------------- | ---------------------------- |
| `Formatted Date`           | Date and time of observation |
| `Summary`                  | General weather condition    |
| `Precip Type`              | Type of precipitation        |
| `Temperature (C)`          | Recorded temperature         |
| `Apparent Temperature (C)` | Feels-like temperature       |
| `Humidity`                 | Humidity level               |
| `Wind Speed (km/h)`        | Wind speed                   |
| `Wind Bearing (degrees)`   | Wind direction               |
| `Visibility (km)`          | Visibility                   |
| `Loud Cover`               | Cloud-related measurement    |
| `Pressure (millibars)`     | Atmospheric pressure         |
| `Daily Summary`            | Daily weather description    |

For the RNN model, the following three numerical features were selected:

```text
Temperature (C)
Humidity
Wind Speed (km/h)
```

The target variable is:

```text
Temperature (C)
```

---

## 🧠 Machine Learning Approach

The project follows a time-series forecasting workflow:

```text
Weather Dataset
       ↓
Data Exploration
       ↓
Feature Selection
       ↓
Train / Validation / Test Split
       ↓
MinMax Scaling
       ↓
Create 7-Step Sequences
       ↓
SimpleRNN Model
       ↓
Model Training
       ↓
Test Prediction
       ↓
RMSE / MAE / R²
       ↓
7-Step Future Forecast
```

---

# 🔍 1. Data Exploration

The dataset is loaded using Pandas:

```python
import pandas as pd

df = pd.read_csv("/content/weatherHistory.csv")
```

The temperature trend is visualized over time:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.plot(
    df['Formatted Date'],
    df['Temperature (C)']
)

plt.xlabel('Date')
plt.ylabel('Temperature (C)')
plt.title('Daily Temperature Trend')

plt.show()
```

This helps identify the temporal pattern and variation in temperature.

---

# 🛠️ 2. Feature Selection

Three weather variables were selected as model inputs:

```python
features = [
    'Temperature (C)',
    'Humidity',
    'Wind Speed (km/h)'
]
```

The selected data is extracted as a NumPy array:

```python
data = df[features].values
```

The resulting dataset contains:

```text
24,507 observations × 3 features
```

---

# ✂️ 3. Train, Validation and Test Split

Since this is a **time-series problem**, the data is split chronologically rather than randomly.

The split is:

* **70% Training**
* **15% Validation**
* **15% Testing**

```python
n = len(data)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_data = data[:train_end]

val_data = data[
    train_end:val_end
]

test_data = data[val_end:]
```

### Dataset split

| Dataset    | Samples |
| ---------- | ------: |
| Training   |  17,154 |
| Validation |   3,676 |
| Testing    |   3,677 |

Chronological splitting is important because randomly mixing time-series observations can cause information from the future to enter the training data.

---

# 📏 4. Data Normalization

A `MinMaxScaler` is used to normalize the numerical features.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_data)

val_scaled = scaler.transform(val_data)

test_scaled = scaler.transform(test_data)
```

The scaler is fitted only on the training data and then applied to validation and test data.

This helps prevent data leakage from the validation and test sets.

---

# 🔄 5. Creating Time-Series Sequences

The model uses the previous **7 time steps** to predict the next temperature.

For example:

```text
Time 1 ─┐
Time 2  │
Time 3  │
Time 4  ├──→ SimpleRNN ──→ Next Temperature
Time 5  │
Time 6  │
Time 7 ─┘
```

The sequence creation function:

```python
import numpy as np

def create_sequences(data, sequence_length=7):

    X = []
    y = []

    for i in range(
        sequence_length,
        len(data)
    ):

        X.append(
            data[i-sequence_length:i]
        )

        # Temperature is the first column
        y.append(data[i, 0])

    return np.array(X), np.array(y)
```

Sequences are created for training, validation, and testing:

```python
X_train, y_train = create_sequences(
    train_scaled,
    7
)

X_val, y_val = create_sequences(
    val_scaled,
    7
)

X_test, y_test = create_sequences(
    test_scaled,
    7
)
```

### Input shape

The training data has the shape:

```text
(17147, 7, 3)
```

This means:

```text
17147 → number of sequences
7     → time steps
3     → features
```

Each sequence therefore contains:

```text
7 previous observations
×
3 weather features
```

---

# 🧠 6. SimpleRNN Architecture

The model uses a SimpleRNN with 64 units followed by Dropout and a Dense output layer.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

model = Sequential()

model.add(
    SimpleRNN(
        64,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(1)
)
```

### Architecture

```text
Input
  │
  │ 7 time steps × 3 features
  ↓
SimpleRNN
64 units
  ↓
Dropout
20%
  ↓
Dense
1 output
  ↓
Predicted Temperature
```

### Model parameters

The trained model contains approximately:

```text
4,417 trainable parameters
```

---

# ⚙️ 7. Model Compilation

The model is compiled using:

* **Adam** optimizer
* **Mean Squared Error (MSE)** loss
* **Mean Absolute Error (MAE)** metric

```python
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

### Why these metrics?

**MSE**

Measures the squared difference between actual and predicted temperatures and is used as the training loss.

**MAE**

Measures the average absolute prediction error.

---

# 🏋️ 8. Model Training

The model is trained for **50 epochs** with a batch size of **32**.

```python
model_trained = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)
```

The training history is stored in:

```python
model_trained
```

Training and validation loss can be visualized using:

```python
plt.figure(figsize=(10, 5))

plt.plot(
    model_trained.history['loss'],
    label='Training Loss'
)

plt.plot(
    model_trained.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')

plt.legend()
plt.show()
```

---

# 📈 9. Model Evaluation

The trained model is used to predict temperatures from the test set:

```python
y_pred = model.predict(X_test)
```

The predictions are converted back to the original temperature scale before calculating the final metrics.

The following evaluation metrics are used:

* RMSE
* MAE
* R² Score

### Final Test Results

| Metric       |         Score |
| ------------ | ------------: |
| **RMSE**     | **1.4831 °C** |
| **MAE**      | **1.0723 °C** |
| **R² Score** |    **0.9769** |

### Interpretation

The model achieved an **MAE of approximately 1.07°C**, meaning the average absolute difference between the predicted and actual temperatures on the test set was about 1.07°C.

The **R² score of 0.9769** indicates that the model was able to explain a large proportion of the variation in the test temperature data.

---

# 📊 10. Actual vs Predicted Temperature

The actual and predicted test temperatures are plotted together:

```python
plt.figure(figsize=(12, 5))

plt.plot(
    y_test_actual,
    label='Actual Temperature'
)

plt.plot(
    y_pred_actual,
    label='Predicted Temperature'
)

plt.xlabel('Test Period')
plt.ylabel('Temperature (°C)')
plt.title('Actual vs Predicted Temperature')

plt.legend()
plt.show()
```

This visualization helps assess how closely the SimpleRNN predictions follow the actual temperature pattern.

---

# 🔮 11. Seven-Step Temperature Forecast

The model is also used to forecast the next **7 time steps**.

The forecasting process starts with the most recent 7 observations:

```python
current_sequence = test_scaled[-7:].copy()

current_sequence = current_sequence.reshape(
    1, 7, 3
)
```

The model then predicts one temperature at a time.

After each prediction:

1. The predicted temperature is added to the sequence.
2. The oldest observation is removed.
3. The updated sequence is passed back into the model.
4. The next temperature is predicted.

This process is repeated seven times.

```python
future_predictions = []

for i in range(7):

    prediction = model.predict(
        current_sequence,
        verbose=0
    )[0, 0]

    future_predictions.append(prediction)

    next_day = current_sequence[0, -1].copy()

    next_day[0] = prediction

    current_sequence = np.vstack([
        current_sequence[0, 1:],
        next_day
    ])

    current_sequence = current_sequence.reshape(
        1, 7, 3
    )
```

The normalized predictions are converted back to Celsius:

```python
future_predictions = np.array(
    future_predictions
)

future_temperature = (
    future_predictions *
    (temp_max - temp_min)
    + temp_min
)
```

---

# 📉 12. Historical Temperature vs 7-Step Forecast

The last 30 historical temperature observations are plotted together with the seven forecasted values.

```python
recent_temperature = (
    df['Temperature (C)'].values[-30:]
)

historical_x = np.arange(
    len(recent_temperature)
)

future_x = np.arange(
    len(recent_temperature),
    len(recent_temperature) + 7
)

plt.figure(figsize=(12, 5))

plt.plot(
    historical_x,
    recent_temperature,
    label='Recent Historical Temperature'
)

plt.plot(
    future_x,
    future_temperature,
    marker='o',
    label='7-Day Forecast'
)

plt.axvline(
    x=len(recent_temperature) - 1,
    linestyle='--',
    label='Forecast Starts'
)

plt.xlabel('Days')
plt.ylabel('Temperature (°C)')
plt.title('7-Day Temperature Forecast')

plt.legend()
plt.show()
```

---

# ⚠️ Forecasting Assumption

For the future forecasting stage, the future humidity and wind speed values are not available.

Therefore, the implementation keeps the latest known humidity and wind speed values while recursively replacing the temperature with the model's prediction.

This is a **simplified forecasting approach** designed for demonstrating multi-step RNN forecasting.

A production weather forecasting system would require future/external weather variables or a multi-output forecasting model.

---

# 🧰 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow
* Keras
* Google Colab

---

# 📁 Project Structure

```text
SimpleRNN-Weather-Temperature-Prediction/
│
├── Weather_Temperature_SimpleRNN.ipynb
├── weatherHistory.csv
└── README.md
```

> The dataset can be excluded from the repository if it is too large or if its redistribution is restricted. The notebook can instead load the dataset from a local or Colab path.

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/SimpleRNN-Weather-Temperature-Prediction.git
```

### 2. Open the notebook

Open:

```text
Weather_Temperature_SimpleRNN.ipynb
```

using:

* Google Colab
* Jupyter Notebook
* JupyterLab

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
```

### 4. Place the dataset

Place:

```text
weatherHistory.csv
```

in the appropriate working directory.

The notebook currently loads it using:

```python
df = pd.read_csv("/content/weatherHistory.csv")
```

### 5. Run all cells

Run the notebook from top to bottom to:

1. Load the weather data
2. Explore the dataset
3. Prepare the features
4. Split the time series
5. Normalize the data
6. Create sequences
7. Train the SimpleRNN
8. Evaluate the model
9. Plot actual vs predicted temperatures
10. Generate the 7-step forecast

---

# 📌 Key Results

```text
Input Features:
Temperature + Humidity + Wind Speed

Sequence Length:
7 time steps

RNN Units:
64

Dropout:
0.20

Epochs:
50

Batch Size:
32

Optimizer:
Adam

Loss:
MSE

Test RMSE:
1.4831 °C

Test MAE:
1.0723 °C

Test R²:
0.9769
```

---

# 💡 Key Learning Outcomes

Through this project, the following concepts were implemented:

* Time-series data preprocessing
* Chronological train/validation/test splitting
* Min-Max normalization
* Sequence generation
* Recurrent Neural Networks
* SimpleRNN architecture
* Dropout regularization
* Regression using neural networks
* Model validation
* RMSE, MAE and R² evaluation
* Actual vs predicted visualization
* Recursive multi-step forecasting

---

# 🔮 Future Improvements

Possible improvements include:

* Experimenting with different sequence lengths such as 14 or 30 time steps
* Comparing SimpleRNN with LSTM and GRU
* Including atmospheric pressure and other weather variables
* Using proper future weather features for multi-step forecasting
* Adding EarlyStopping and learning-rate scheduling
* Hyperparameter tuning
* Saving and loading the trained model
* Creating a web interface for temperature forecasting
* Using a dedicated multi-step forecasting architecture

---

# 👨‍💻 Author

**Harigovind P.**

B.Tech Computer Science and Engineering
LBS College of Engineering

### Connect

* LinkedIn: [Harigovind P.](https://www.linkedin.com/in/harigovindp2004/)
* GitHub: [Haariiiii](https://github.com/Haariiiii)

---

## ⭐ Project Summary

This project demonstrates how a **SimpleRNN can learn temporal patterns from historical weather observations and perform temperature forecasting**. By using a 7-step sliding window containing temperature, humidity, and wind speed, the model achieved an **RMSE of 1.4831°C, MAE of 1.0723°C, and R² of 0.9769** on the test data.

The project provides a practical introduction to **time-series forecasting with recurrent neural networks**.
