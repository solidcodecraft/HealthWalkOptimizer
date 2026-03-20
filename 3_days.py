# ======================================
# ИМПОРТЫ
# ======================================

import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import logging

from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ======================================
# НАСТРОЙКИ
# ======================================

CITIES = [
    "Moscow",
    "Krasnodar",
    "Omsk",
    "Phu Quoc",
    "Abu Dhabi",
    "Cologne",
    "Bergneustadt",
    "Limpde",
    "Sochi",
    "Goryachy Klyuch",
    "Adler"
]

print("Выберите город:")

for i, c in enumerate(CITIES, 1):
    print(f"{i}. {c}")

choice = int(input("Введите номер города: "))
CITY = CITIES[choice-1]

print(f"Выбран город: {CITY}")

TIMEZONE = "auto"

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# ======================================
# КООРДИНАТЫ
# ======================================

geo = requests.get(GEOCODE_URL, params={"name": CITY, "count": 1}).json()

lat = geo["results"][0]["latitude"]
lon = geo["results"][0]["longitude"]

# ======================================
# ДАТЫ
# ======================================

today = datetime.now().date()
yesterday = today - timedelta(days=1)
tomorrow = today + timedelta(days=1)

# ======================================
# ARCHIVE
# ======================================

archive = requests.get(
    ARCHIVE_URL,
    params={
        "latitude": lat,
        "longitude": lon,
        "start_date": yesterday,
        "end_date": today,
        "hourly": [
            "temperature_2m",
            "cloudcover",
            "windspeed_10m",
            "windgusts_10m",
        ],
        "timezone": TIMEZONE,
    },
).json()

df_archive = pd.DataFrame({
    "datetime": archive["hourly"]["time"],
    "temp": archive["hourly"]["temperature_2m"],
    "cloudcover": archive["hourly"]["cloudcover"],
    "wind_max": archive["hourly"]["windspeed_10m"],
    "wind_gusts": archive["hourly"]["windgusts_10m"],
})

# ======================================
# FORECAST
# ======================================

forecast = requests.get(
    FORECAST_URL,
    params={
        "latitude": lat,
        "longitude": lon,
        "start_date": tomorrow,
        "end_date": tomorrow,
        "hourly": [
            "temperature_2m",
            "cloudcover",
            "windspeed_10m",
            "windgusts_10m",
        ],
        "timezone": TIMEZONE,
    },
).json()

df_forecast = pd.DataFrame({
    "datetime": forecast["hourly"]["time"],
    "temp": forecast["hourly"]["temperature_2m"],
    "cloudcover": forecast["hourly"]["cloudcover"],
    "wind_max": forecast["hourly"]["windspeed_10m"],
    "wind_gusts": forecast["hourly"]["windgusts_10m"],
})

# ======================================
# ОБЪЕДИНЕНИЕ
# ======================================

df = pd.concat([df_archive, df_forecast], ignore_index=True)

df["datetime"] = pd.to_datetime(df["datetime"])
df["day"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour

df["wind"] = 0.3 * df["wind_max"] + 0.7 * df["wind_gusts"]

# ======================================
# СОЛНЕЧНАЯ ВЫСОТА
# ======================================

def solar_elevation(dt, lat, lon):

    day = dt.timetuple().tm_yday
    hour = dt.hour + dt.minute/60

    gamma = 2*np.pi/365 * (day - 1 + (hour - 12)/24)

    decl = (
        0.006918
        - 0.399912*np.cos(gamma)
        + 0.070257*np.sin(gamma)
        - 0.006758*np.cos(2*gamma)
        + 0.000907*np.sin(2*gamma)
        - 0.002697*np.cos(3*gamma)
        + 0.00148*np.sin(3*gamma)
    )

    lat_rad = np.radians(lat)
    hour_angle = np.radians((hour - 12)*15)

    elevation = np.arcsin(
        np.sin(lat_rad)*np.sin(decl)
        + np.cos(lat_rad)*np.cos(decl)*np.cos(hour_angle)
    )

    return np.degrees(elevation)

# ======================================
# LUX
# ======================================

def estimate_lux(row):

    elevation = solar_elevation(row["datetime"], lat, lon)

    if elevation <= 0:
        return 0

    sun = np.sin(np.radians(elevation))
    cloud = 1 - row["cloudcover"]/100

    return 100000 * sun * cloud

# ======================================
# HEALTH SCORE
# ======================================

def health_score(row):

    lux = estimate_lux(row)

    lux_factor = np.tanh(lux / 20000)
    wind_factor = np.exp(-(row["wind"]/18)**2)

    return 100 * lux_factor * wind_factor

# ======================================
# ИНТЕРПРЕТАЦИЯ
# ======================================

def explain_conditions(row):

    text = []

    if row["lux"] > 20000:
        text.append("🌞 яркое солнце")
    elif row["lux"] > 5000:
        text.append("🌤 умеренный свет")
    else:
        text.append("🌥 мало света")

    if row["cloudcover"] < 30:
        text.append("☀ ясно")
    elif row["cloudcover"] < 70:
        text.append("⛅ переменная облачность")
    else:
        text.append("☁ облачно")

    if row["wind"] < 8:
        text.append("🍃 слабый ветер")
    elif row["wind"] < 15:
        text.append("💨 ветер")
    else:
        text.append("🌬 сильный ветер")

    if row["temp"] < 0:
        text.append("🥶 холодно")
    elif row["temp"] < 10:
        text.append("🧥 прохладно")
    else:
        text.append("🙂 комфортно")

    return ", ".join(text)

# ======================================
# ЦВЕТ
# ======================================

def health_color(h):

    if h > 60:
        return "green"
    if h > 30:
        return "yellow"
    return "red"

# ======================================
# ЛУЧШЕЕ ОКНО
# ======================================

def best_walk_window(df_day):

    threshold = df_day["health"].max() * 0.6
    good = df_day[df_day["health"] >= threshold]

    if good.empty:
        return None

    hours = sorted(good["hour"].tolist())

    start = hours[0]
    prev = hours[0]
    best = (start, prev)

    for h in hours[1:]:

        if h == prev + 1:
            prev = h
        else:

            if prev - start > best[1] - best[0]:
                best = (start, prev)

            start = h
            prev = h

    if prev - start > best[1] - best[0]:
        best = (start, prev)

    return best

# ======================================
# ГРАФИК
# ======================================

def plot_day(day):

    df_day = df[df["day"] == day].copy()

    df_day["lux"] = df_day.apply(estimate_lux, axis=1)
    df_day["health"] = df_day.apply(health_score, axis=1)
    df_day["color"] = df_day["health"].apply(health_color)
    df_day["explanation"] = df_day.apply(explain_conditions, axis=1)

    best = df_day.sort_values("health", ascending=False).head(10)

    print()
    print(f"{CITY} — {day}")
    print("Лучшее время для прогулки")
    print("="*90)

    print(best[
        ["hour","health","lux","wind","cloudcover","explanation"]
    ].round(1).to_string(index=False))

    window = best_walk_window(df_day)

    if window:
        start, end = window
        print()
        print(f"⭐ Лучшее окно прогулки: {start}:00 – {end}:00")

    plt.figure(figsize=(12,6))

    plt.axhspan(0,30,color="red",alpha=0.08)
    plt.axhspan(30,60,color="yellow",alpha=0.08)
    plt.axhspan(60,100,color="green",alpha=0.08)

    palette = {
        "green":"green",
        "yellow":"yellow",
        "red":"red"
    }

    sns.scatterplot(
        data=df_day,
        x="hour",
        y="health",
        hue="color",
        palette=palette,
        size="health",
        sizes=(50,400),
        legend=False
    )

    if window:
        start, end = window
        for _, r in df_day.iterrows():
            if start <= r["hour"] <= end:
                plt.text(
                    r["hour"],
                    r["health"],
                    "🚶",
                    ha="center",
                    va="center",
                    fontsize=14
                )

    plt.title(f"{CITY} — {day}")
    plt.xlabel("час")
    plt.ylabel("Здоровье")

    plt.xlim(-0.5,23.5)
    plt.ylim(0,100)

    plt.xticks(range(24))
    plt.grid(alpha=0.3)

    plt.show()

# ======================================
# ЗАПУСК
# ======================================

for d in [today,yesterday, tomorrow]:
    plot_day(d)