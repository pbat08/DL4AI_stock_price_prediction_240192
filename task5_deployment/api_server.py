import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import concurrent.futures

# Bật XLA để dự đoán nhanh
tf.config.optimizer.set_jit(True)

app = Flask(__name__)

tickers = [
    "VIC","HPG","VCB","GAS","PLX",
    "BID","FPT","MBB","MSN","VNM"
]

models = {}   # lưu: (fast_predict, timesteps, num_features)

def load_one(ticker):
    # Dùng đường dẫn tương đối – Render sẽ chạy từ /app
    path = f"models/{ticker}_price_model.h5"
    model = tf.keras.models.load_model(path, compile=False)

    input_shape = model.input_shape          # (None, timesteps, features)
    timesteps = input_shape[1]
    num_feat = input_shape[2]

    # @tf.function để tăng tốc inference
    input_spec = tf.TensorSpec(shape=(1, timesteps, num_feat), dtype=tf.float32)
    @tf.function(input_signature=[input_spec])
    def fast_predict(x):
        return model(x, training=False)

    # Làm nóng model
    dummy = np.zeros((1, timesteps, num_feat), dtype=np.float32)
    fast_predict(dummy)

    return ticker, fast_predict, timesteps, num_feat

# Load song song để nhanh
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(load_one, t) for t in tickers]
    for future in concurrent.futures.as_completed(futures):
        ticker, fast_fn, tsteps, nf = future.result()
        models[ticker] = (fast_fn, tsteps, nf)
        print(f"✅ {ticker} loaded and pre‑warmed")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get("ticker")
    features = data.get("features")   # dict với Open, High, Low, Close, Volume

    if ticker not in models:
        return jsonify({"error": f"Unknown ticker. Choose from {list(models.keys())}"}), 400

    fast_predict, timesteps, num_feat = models[ticker]

    feature_list = [features["Open"], features["High"], features["Low"],
                    features["Close"], features["Volume"]]
    input_array = np.array([feature_list], dtype=np.float32)

    # Nếu model là LSTM (3D), mở rộng thêm chiều thời gian bằng cách lặp dữ liệu
    input_array = np.tile(input_array.reshape(1, 1, num_feat), (1, timesteps, 1))

    pred = fast_predict(input_array).numpy().tolist()
    return jsonify({"ticker": ticker, "prediction": pred})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=False)