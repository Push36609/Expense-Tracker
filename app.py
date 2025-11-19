# app.py
from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
from pyod.models.iforest import IForest
import os

app = Flask(__name__)     # FIXED: should be __name__

EXPECTED_COLUMNS = ['Date', 'Amount', 'Merchant', 'Description', 'Category']


def ensure_dataframe_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe always contains the expected schema."""
    for column in EXPECTED_COLUMNS:
        if column not in dataframe.columns:
            default_value = 0.0 if column == 'Amount' else ''
            dataframe[column] = default_value

    dataframe = dataframe[EXPECTED_COLUMNS]
    dataframe['Amount'] = pd.to_numeric(dataframe['Amount'], errors='coerce').fillna(0.0)
    return dataframe


# Load or create CSV file
if os.path.exists('expense_data.csv'):
    df = pd.read_csv('expense_data.csv')
    df = ensure_dataframe_schema(df)
else:
    df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    df.to_csv('expense_data.csv', index=False)

# Load trained models if available
if os.path.exists('model/category_model.pkl'):
    category_model = joblib.load('model/category_model.pkl')
else:
    category_model = None

if os.path.exists('model/anomaly_model.pkl'):
    anomaly_model = joblib.load('model/anomaly_model.pkl')
else:
    anomaly_model = None

# Predefined categories
categories = ['Food', 'Travel', 'Bills', 'Shopping', 'Entertainment', 'Others']


@app.route('/')
def index():
    global df
    expenses = df.copy()

    if expenses.empty:
        context = {
            'expenses': [],
            'total': 0,
            'stats': {
                'count': 0,
                'average': 0,
                'largest': None,
                'daily_projection': 0,
                'weekly_projection': 0,
            },
            'recent_expenses': [],
            'category_totals': [],
            'daily_totals': [],
            'categories': categories,
        }
        return render_template('index.html', **context)

    expenses['Amount'] = expenses['Amount'].astype(float)
    total_expense = float(expenses['Amount'].sum())
    expense_count = len(expenses)

    daily_totals = (
        expenses.groupby('Date')['Amount']
        .sum()
        .reset_index()
        .sort_values('Date')
    )
    max_daily_amount = float(daily_totals['Amount'].max()) if not daily_totals.empty else 0

    recent_expenses = (
        expenses.sort_values('Date', ascending=False)
        .head(5)
        .to_dict(orient='records')
    )

    highest_expense_row = expenses.loc[expenses['Amount'].idxmax()].to_dict()

    category_totals_df = (
        expenses.groupby('Category')['Amount']
        .sum()
        .reset_index()
        .sort_values('Amount', ascending=False)
    )

    category_totals = []
    for _, row in category_totals_df.iterrows():
        percent = (row['Amount'] / total_expense) * 100 if total_expense else 0
        category_totals.append({
            'label': row['Category'],
            'amount': round(float(row['Amount']), 2),
            'percent': round(percent, 1),
        })

    unique_days = expenses['Date'].nunique()
    daily_projection = total_expense / unique_days if unique_days else total_expense
    weekly_projection = daily_projection * 7

    stats = {
        'count': expense_count,
        'average': round(total_expense / expense_count, 2),
        'largest': {
            'merchant': highest_expense_row['Merchant'],
            'description': highest_expense_row['Description'],
            'amount': round(float(highest_expense_row['Amount']), 2),
            'date': highest_expense_row['Date'],
        },
        'daily_projection': round(daily_projection, 2),
        'weekly_projection': round(weekly_projection, 2),
    }

    context = {
        'expenses': expenses.to_dict(orient='records'),
        'total': total_expense,
        'stats': stats,
        'recent_expenses': recent_expenses,
        'category_totals': category_totals,
        'daily_totals': daily_totals.to_dict(orient='records'),
        'daily_max': max_daily_amount,
        'categories': categories,
    }

    return render_template('index.html', **context)


@app.route('/add', methods=['POST'])
def add_expense():
    global df, category_model, anomaly_model

    date = request.form['date']
    amount = float(request.form['amount'])
    merchant = request.form['merchant']
    description = request.form['description']

    # Simple ML prediction
    if category_model:
        cat_pred = category_model.predict([[len(description)]])[0]
    else:
        cat_pred = 'Others'

    # Add new row (FIXED: df.append is deprecated)
    new_expense = {
        'Date': date,
        'Amount': amount,
        'Merchant': merchant,
        'Description': description,
        'Category': cat_pred
    }

    df = pd.concat([df, pd.DataFrame([new_expense])], ignore_index=True)
    df = ensure_dataframe_schema(df)
    df.to_csv('expense_data.csv', index=False)

    # Optional anomaly detection
    if anomaly_model:
        anomaly_pred = anomaly_model.predict([[amount]])
        if anomaly_pred[0] == 1:
            print("Anomaly Detected:", new_expense)

    return redirect('/')


@app.route('/forecast')
def forecast():
    global df

    if df.empty:
        return "Not enough data for forecasting."

    df_prophet = df[['Date', 'Amount']].rename(columns={'Date': 'ds', 'Amount': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    forecast_df = forecast[['ds', 'yhat']].tail(7)

    forecast_df['ds'] = forecast_df['ds'].dt.strftime('%Y-%m-%d')
    forecast_df['yhat'] = forecast_df['yhat'].round(2)

    forecast_summary = {
        'average': round(forecast_df['yhat'].mean(), 2),
        'max': round(forecast_df['yhat'].max(), 2),
        'min': round(forecast_df['yhat'].min(), 2),
    }

    peak_day = forecast_df.loc[forecast_df['yhat'].idxmax()]
    low_day = forecast_df.loc[forecast_df['yhat'].idxmin()]
    forecast_summary['peak_day'] = peak_day['ds']
    forecast_summary['low_day'] = low_day['ds']

    return render_template(
        'forecast.html',
        forecast=forecast_df.to_dict(orient='records'),
        forecast_summary=forecast_summary,
    )


if __name__ == '__main__':   # FIXED
    app.run(debug=True)
