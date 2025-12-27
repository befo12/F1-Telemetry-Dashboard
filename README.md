# 🏎️ F1 Telemetry & Analysis Dashboard

An interactive data analysis tool for Formula 1 fans and data enthusiasts. This dashboard leverages the **FastF1 API** to fetch real-time and historical race data, providing deep insights into driver performance, telemetry, and race strategies.

## 🚀 Features

* **Race & Qualifying Results:** Detailed leaderboards for every session from 2018 to present.
* **Telemetry Analysis:** Compare speed profiles of two drivers on any track to see where they gain/lose time.
* **Pit Stop Strategy:** Analyze tire strategies, pit stop durations, and delta to the leader.
* **Track Intelligence:** Visual circuit maps and location data.
* **Championship Standings:** Interactive charts showing the points progression throughout the season.
* **Event Timeline:** Automatic tracking of Safety Cars, Red Flags, and Penalties.

## 🛠️ Tech Stack

* **Python** (Core Logic)
* **Streamlit** (Web Dashboard Framework)
* **FastF1** (Official F1 Data API Wrapper)
* **Plotly & Matplotlib** (Data Visualization)
* **Pandas** (Data Manipulation)

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/befo12/F1-Telemetry-Dashboard.git](https://github.com/befo12/F1-Telemetry-Dashboard.git)
    ```
2.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the app:
    ```bash
    streamlit run app.py
    ```
