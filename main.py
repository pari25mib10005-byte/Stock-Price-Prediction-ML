import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# 1. Download stock data
# -----------------------------
TICKER = "AAPL"

data = yf.download(
    TICKER,
    period="5y",
    auto_adjust=False,
    progress=False
)

if data.empty:
    raise ValueError("Could not download stock data.")

# Handle yfinance multi-level columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.dropna().copy()


# -----------------------------
# 2. Feature engineering
# -----------------------------
data["MA_5"] = data["Close"].rolling(5).mean()
data["MA_10"] = data["Close"].rolling(10).mean()
data["Previous_Close"] = data["Close"].shift(1)

# Target = next day's closing price
data["Target"] = data["Close"].shift(-1)

data = data.dropna()


features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA_5",
    "MA_10",
    "Previous_Close"
]

X = data[features]
y = data["Target"]


# -----------------------------
# 3. Chronological train/test split
# -----------------------------
split = int(len(data) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


# -----------------------------
# 4. Train models
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []
predictions = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    predictions[name] = prediction

    mae = mean_absolute_error(y_test, prediction)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    r2 = r2_score(y_test, prediction)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# -----------------------------
# 5. Display results
# -----------------------------
results_df = pd.DataFrame(results)

print("\nStock Price Prediction")
print("----------------------")
print(f"Stock: {TICKER}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("\nModel Performance:")
print(results_df.to_string(index=False))


# -----------------------------
# 6. Save results
# -----------------------------
results_df.to_csv("model_results.csv", index=False)


# -----------------------------
# 7. Plot stock price
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(data.index, data["Close"])
plt.title(f"{TICKER} Historical Closing Price")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.tight_layout()
plt.savefig("stock_price.png")
plt.close()


# -----------------------------
# 8. Actual vs predicted
# -----------------------------
best_model = results_df.sort_values("RMSE").iloc[0]["Model"]

plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label="Actual Price")
plt.plot(
    y_test.index,
    predictions[best_model],
    label=f"Predicted ({best_model})"
)

plt.title("Actual vs Predicted Stock Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()


# -----------------------------
# 9. Model comparison
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("Model Comparison - RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

print("\nBest model:", best_model)
print("Results saved to model_results.csv")
print("Graphs saved successfully.")
