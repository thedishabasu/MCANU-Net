from flask import Flask, render_template_string, request, send_file
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Load CSV at start for features
df = pd.read_csv('multi_task_results.csv')
dates = df['Date'].tolist()
actuals = df['Actual Price ($)'].tolist()
preds = df['Predicted Price ($)'].tolist()

date_format = "%Y-%m-%d"
latest_date = dates[-1]

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MCANU-Net | Demo UI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@500&display=swap" rel="stylesheet">
    <style>
        body { background: linear-gradient(112deg,#ebeaff 10%,#b9d6f2 90%);
               font-family: 'Inter', sans-serif; min-height: 100vh; margin:0; color: #23234c; }
        .card { background: #fff; max-width: 950px; margin: 48px auto 0 auto; border-radius: 22px;
                padding: 32px 25px 26px 25px; box-shadow:0 12px 56px #6c5ce74a,0 2px 14px #82cfff16;text-align:center;}
        h1 {font-size:2.10em;font-weight:800;margin-bottom:13px;background:linear-gradient(90deg,#2e6bf6 50%,#d21eea 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
        form { margin-bottom:17px;}
        label, input[type=date] {font-size:1.07em;}
        input[type=date] {padding:10px 15px;border-radius:11px;border:1.2px solid #acd8ff;margin-left:8px;}
        button {padding:10px 26px;background:linear-gradient(90deg,#0040bd,#fd40e5);color:#fff;border:none;border-radius:12px;font-size:1.08em;box-shadow:0 3px 15px #d4d4fe48;cursor:pointer; font-weight:700;}
        button:hover {background:linear-gradient(97deg,#fd40e5,#0040bd);}
        .output-section p {margin:5px 0 9px 0;line-height:1.5em;font-size:1.14em;max-width:440px;margin:auto;}
        .diff {color:#2ca86e;}
        .err { color: #c70030; font-size:1.08em; font-weight:600;}
        .graph-box {background: #f6f6fc; border-radius: 14px; margin:20px auto 0 auto; box-shadow: 0 2px 32px #becbff44; padding: 17px; max-width:900px;}
        img { max-width:97.5%; border-radius:10px; margin-top:7px;box-shadow: 0 8px 46px #b4c7fc63;}
        .footer { color:#8b8bb1;margin:25px 0 5px 0;font-size:1em;}
    </style>
</head>
<body>
    <div class="card">
        <h1>MCANU-Net | Bitcoin Price Prediction UI</h1>
        <form method="POST">
            <label for="date"><b>Select Date:</b></label>
            <input type="date" name="date" id="date" min="{{ dates[0] }}" max="" value="{{ selected_date or latest_date }}">
            <button type="submit">Show Prediction</button>
        </form>
        <div class="output-section">
            {% if msg %}<p class="err">{{ msg }}</p>{% endif %}
            {% if output %}
                <p><b>Date:</b> {{ output.date }}</p>
                <p>
                  <span style="color:#1576ec;font-weight:700;">Actual:</span> {{ output.actual }} &nbsp;|&nbsp;
                  <span style="color:#f6085a;font-weight:700;">Predicted:</span> {{ output.predicted }}
                </p>
                <p>Difference: <span class="diff">{{ output.diff }}</span></p>
            {% endif %}
            {% if forecast %}
                <p><span style="color:#ee20b6;font-weight:700;font-size:1.09em;">Forecast for {{ forecast.date }}: {{ forecast.predicted }}</span>
                <br><span style="font-size:0.97em;color:#937095;">Trend-based estimate</span>
                </p>
            {% endif %}
        </div>
        <div class="graph-box">
            <h3 style="font-weight:500;margin:7px 0 8px 0;">Actual vs Predicted Bitcoin Price Chart</h3>
            {% if graph_exists %}
                <img src="/inference-graph" alt="Actual vs Predicted Inference Graph"/>
            {% else %}
                <div class="err">inference_graph.png not found.</div>
            {% endif %}
        </div>
        <div class="footer">
            MCANU-Net UI ⓒ Powered by Flask
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    selected_date, output, forecast, msg = None, None, None, None
    selected_date = request.form.get('date', latest_date) if request.method == 'POST' else latest_date

    if selected_date:
        if selected_date in dates:
            idx = dates.index(selected_date)
            output = {
                'date': selected_date,
                'actual': f"${actuals[idx]:,.2f}",
                'predicted': f"${preds[idx]:,.2f}",
                'diff': f"${(preds[idx]-actuals[idx]):,.2f}",
            }
        elif selected_date > latest_date:
            # Linear forecast, same as in earlier Flask UIs
            N = min(8, len(preds))
            daily_delta = float(pd.Series(preds).diff().tail(N).mean())
            past = float(preds[-1])
            future_hops = (datetime.strptime(selected_date, date_format) - datetime.strptime(latest_date, date_format)).days
            f_pred = past + daily_delta*future_hops
            forecast = {
                "date": selected_date,
                "predicted": f"${f_pred:,.2f}"
            }
        else:
            msg = "Date not available in loaded predictions."
  
    graph_exists = os.path.exists("inference_graph.png")
    return render_template_string(TEMPLATE,
        dates=dates,
        latest_date=latest_date,
        selected_date=selected_date,
        output=output,
        forecast=forecast,
        msg=msg,
        graph_exists=graph_exists
    )

@app.route('/inference-graph')
def inference_graph():
    return send_file('inference_graph.png', mimetype='image/png')

if __name__ == "__main__":
    print("Open http://localhost:5000")
    app.run(debug=True, port=5000)
