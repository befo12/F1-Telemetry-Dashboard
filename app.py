import streamlit as st
import fastf1
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fastf1 import plotting
import matplotlib.pyplot as plt
import numpy as np

# Matplotlib için F1 stilini yükle (Pist haritası için gerekli)
plotting.setup_mpl()

# Sayfa yapılandırmasını ayarla
st.set_page_config(layout="wide", page_title="F1 Analiz Pro")

# --- Takım Renkleri Paleti ---
TEAM_COLORS = {
    "Red Bull Racing": "#0600EF", "Mercedes": "#00D2BE", "Ferrari": "#DC0000",
    "McLaren": "#FF8700", "Aston Martin": "#006F62", "Alpine": "#0090FF",
    "Williams": "#005AFF", "Haas F1 Team": "#FFFFFF", "Sauber": "#006F62",
    "Kick Sauber": "#52E252", "Racing Bulls": "#0000FF", "AlphaTauri": "#2B4562",
    "Alfa Romeo": "#900000"
}

# --- YARDIMCI FONKSİYONLAR ---
def format_timedelta(td):
    if pd.isna(td) or not isinstance(td, pd.Timedelta): return "—"
    total_seconds = td.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f'{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}'
    else:
        return f'{int(minutes):02d}:{seconds:05.2f}'

# --- VERİ YÜKLEME FONKSİYONLARI ---
fastf1.Cache.enable_cache('f1_cache')

@st.cache_data(ttl=3600)
def get_events(year):
    try: return fastf1.get_event_schedule(year)
    except Exception as e:
        st.error(f"{year} yılı için yarış takvimi yüklenemedi. Hata: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_session(year, gp_name, session_identifier, load_messages=False):
    try:
        session = fastf1.get_session(year, gp_name, session_identifier)
        session.load(laps=True, telemetry=False, weather=False, messages=load_messages)
        st.write("Yarış verisi yüklendi", session.laps.head())  # Yüklenen veri kontrolü

        return session
    except Exception: return None

def display_session_results(session_name, session_obj):
    st.subheader(f"{session_name} Sonuçları")
    if session_obj and session_obj.results is not None and not session_obj.results.empty:
        results = session_obj.results.copy()
        display_cols = ['Position', 'Abbreviation', 'FullName', 'TeamName', 'Time', 'Laps']
        if 'Q' in session_name or 'Sıralama' in session_name:
            display_cols = ['Position', 'Abbreviation', 'TeamName', 'Q1', 'Q2', 'Q3']
            for q_col in ['Q1', 'Q2', 'Q3']:
                if q_col in results.columns: results[q_col] = results[q_col].apply(format_timedelta)
        elif 'Time' in results.columns: results['Time'] = results['Time'].apply(format_timedelta)
        final_cols = [col for col in display_cols if col in results.columns]
        st.dataframe(results[final_cols], use_container_width=True)
    else: st.warning(f"{session_name} için veri bulunamadı.")

@st.cache_data(ttl=86400)
def load_season_race_results(year):
    schedule = get_events(year)
    if schedule.empty: return pd.DataFrame()
    official_events = schedule[~schedule['EventName'].str.contains("Testing", na=False)].copy()
    official_events['EventDate'] = pd.to_datetime(official_events['EventDate'])
    official_events = official_events.sort_values(by='EventDate')
    all_results = []
    for _, event in official_events.iterrows():
        try:
            session = fastf1.get_session(year, event['EventName'], 'R')
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            if session.results is not None:
                results_df = session.results
                results_df['EventName'] = event['EventName']
                results_df['EventDate'] = event['EventDate']
                all_results.append(results_df)
        except Exception: continue
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

# --- STREAMLIT ARAYÜZÜ ---
st.title("🏎️ F1 Analiz Pro")
st.markdown("Detaylı yarış ve sürücü analiz platformu.")

with st.sidebar:
    st.image("https://media.formula1.com/image/upload/f_auto,c_limit,w_1920,q_auto/fom-website/2018-redesign-assets/Formula%201%20logo", width=150)
    st.header("Seçimler")
    year = st.selectbox("Yıl seçin:", list(range(pd.Timestamp.now().year, 2018, -1)))
    events = get_events(year)
    gp_choice = None
    if not events.empty:
        gp_names = events[~events['EventName'].str.contains("Testing", na=False)]['EventName'].tolist()
        if gp_names:
            gp_choice = st.selectbox("GP Seçin:", gp_names)

if gp_choice:
    st.header(f"🗓️ {year} {gp_choice} Analizi")
    st.info("**Not:** Veri mevcudiyeti (özellikle eski sezonlardaki pist haritaları ve olay akışları) yarıştan yarışa değişiklik gösterebilir. Bazı özellikler her Grand Prix için mevcut olmayabilir.")

    race_session = load_session(year, gp_choice, 'R', load_messages=True)
    q_session = load_session(year, gp_choice, 'Q')
    fp1_session = load_session(year, gp_choice, 'FP1')
    fp2_session = load_session(year, gp_choice, 'FP2')
    fp3_session = load_session(year, gp_choice, 'FP3')

    # --- TAMAMEN YENİLENEN PİST BİLGİLERİ BÖLÜMÜ ---
    st.subheader("Pist Bilgileri")
    if race_session:
        col1, col2 = st.columns([1, 2])
        with col1:
            try:
                circuit_info = race_session.session_info['Meeting']['Circuit']
                st.metric("Pist Adı", circuit_info['ShortName'])
                st.metric("Lokasyon", circuit_info['Location'])
                st.metric("Tur Sayısı", race_session.total_laps)
            except Exception:
                st.warning("Pist metrikleri alınamadı.")
        with col2:
            st.markdown("###### Pist Haritası")
            if st.button("Pist Haritasını Oluşturmayı Dene"):
                with st.spinner("Harita verisi aranıyor..."):
                    try:
                        laps_for_map = race_session.laps.pick_quicklaps(1.05) # Sadece temsili turları al
                        if laps_for_map.empty:
                            st.error("Harita çizmek için yeterli hızda bir tur verisi bulunamadı.")
                        else:
                            # Herhangi bir turdan konum verisi al
                            pos_data = laps_for_map.iloc[0].get_pos_data()
                            if pos_data.empty:
                                st.error("Seçilen tur için konum verisi (X,Y) bulunamadı.")
                            else:
                                fig, ax = plt.subplots(figsize=(6, 4))
                                ax.plot(pos_data['X'], pos_data['Y'], color='grey')
                                ax.axis('off')
                                st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Harita oluşturulurken beklenmedik bir hata oluştu. Hata Detayı: {e}")
    else:
        st.warning("Yarış seansı yüklenemediği için pist bilgileri gösterilemiyor.")
    
    st.subheader("Seans Sonuçları")
    ses_tab1, ses_tab2, ses_tab3 = st.tabs(["🏆 Yarış Sonuçları", "⏱️ Sıralama Turları", "🛠️ Antrenman Turları"])
    with ses_tab1: display_session_results("Yarış", race_session)
    with ses_tab2: display_session_results("Sıralama", q_session)
    with ses_tab3:
        display_session_results("Antrenman 3 (FP3)", fp3_session)
        display_session_results("Antrenman 2 (FP2)", fp2_session)
        display_session_results("Antrenman 1 (FP1)", fp1_session)
    st.divider()

    if race_session and not race_session.results.empty:
        results_df = race_session.results.copy()
        laps_df = race_session.laps.copy()
        team_color_map = {row['TeamName']: (f"#{row['TeamColor']}" if row['TeamColor'] and len(row['TeamColor']) == 6 else TEAM_COLORS.get(row['TeamName'], "#FFFFFF")) for i, row in results_df.iterrows()}
        driver_color_map = {row['Abbreviation']: team_color_map.get(row['TeamName'], "#FFFFFF") for i, row in results_df.iterrows()}

        st.header("📋 Sürücü Yarış Karnesi")
        laps_led = laps_df[laps_df['Position'] == 1].groupby('Driver')['LapNumber'].count().to_dict()
        fastest_laps = laps_df.groupby('Driver')['LapTime'].min().to_dict()
        results_df['PositionGain'] = results_df['GridPosition'] - results_df['Position']
        results_df['LapsLed'] = results_df['Abbreviation'].map(laps_led).fillna(0).astype(int)
        results_df['FastestLapTime'] = results_df['Abbreviation'].map(fastest_laps)
        kpi_cols = {'Position': 'Pozisyon', 'Abbreviation': 'Sürücü', 'TeamName': 'Takım', 'GridPosition': 'Start Poz.', 'PositionGain': '+/- Poz.', 'LapsLed': 'Lider Tur', 'FastestLapTime': 'En Hızlı Tur', 'Status': 'Durum'}
        kpi_df = results_df.rename(columns=kpi_cols)
        kpi_df['En Hızlı Tur'] = kpi_df['En Hızlı Tur'].apply(format_timedelta)
        st.dataframe(kpi_df[list(kpi_cols.values())], use_container_width=True)
        st.divider()

        st.header("🚩 Yarışın Kilit Anları ve Olaylar")
        with st.expander("Yarış zaman akışını görmek için tıklayın"):
            if hasattr(race_session, 'messages') and not race_session.messages.empty:
                keywords = ['SAFETY CAR', 'VIRTUAL SAFETY CAR', 'RED FLAG', 'YELLOW FLAG', 'PENALTY', 'INVESTIGATION', 'RETIRED', 'OUT OF THE RACE', 'BLACK AND WHITE FLAG']
                filtered_messages = race_session.messages[race_session.messages['Message'].str.upper().str.contains('|'.join(keywords), na=False)]
                if filtered_messages.empty:
                    st.info("Bu seans için raporlanmış bir Güvenlik Aracı, Ceza veya Bayrak periyodu gibi önemli bir olay bulunamadı.")
                else:
                    for _, msg in filtered_messages.iterrows():
                        time_str = format_timedelta(msg['Time'])
                        lap_str = f"(Tur {int(msg['LapNumber'])})" if pd.notna(msg['LapNumber']) and msg['LapNumber'] > 0 else ""
                        icon = "ℹ️";
                        if "SAFETY CAR" in msg['Message'].upper(): icon = "🚓"
                        elif "FLAG" in msg['Message'].upper(): icon = "🚩"
                        elif "PENALTY" in msg['Message'].upper(): icon = "⚖️"
                        elif "INVESTIGATION" in msg['Message'].upper(): icon = "🔍"
                        elif "RETIRED" in msg['Message'].upper() or "OUT" in msg['Message'].upper(): icon = "💥"
                        st.markdown(f"**{icon} {time_str} {lap_str}:** {msg['Message']}")
            else:
                st.warning("Bu seans için yarış kontrol mesajları (olay akışı) verisi bulunamadı.")
        st.divider()

        st.header("🔧 Pit Stop Analizi")
        pits_df = laps_df[laps_df['PitInTime'].notna()].copy()
        if not pits_df.empty:
            pits_df['PitDuration'] = (pits_df['PitOutTime'] - pits_df['PitInTime']).dt.total_seconds()
            pit_tab1, pit_tab2, pit_tab3 = st.tabs(["🏆 En Hızlı Pit Stoplar", "⏱️ Toplam Süreler", "📊 Lidere Göre Farklar"])
            with pit_tab1:
                st.subheader("Yarışın En Hızlı Pit Stopları Sıralaması")
                st.info("Bu sıralama, yarış boyunca atılan en hızlı tekil pit stopları gösterir.")
                fastest_pits = pits_df.sort_values(by='PitDuration').reset_index(drop=True)
                fastest_pits_display = fastest_pits[['Driver', 'Team', 'LapNumber', 'Stint', 'PitDuration']]
                fastest_pits_display.rename(columns={'Driver': 'Sürücü', 'Team': 'Takım', 'LapNumber': 'Tur', 'PitDuration': 'Süre (sn)'}, inplace=True)
                fastest_pits_display.index += 1
                st.dataframe(fastest_pits_display.head(10), use_container_width=True)
            with pit_tab2:
                st.subheader("Sürücülerin Pitlerde Geçirdiği Toplam Süre")
                pit_summary = pits_df.groupby('Driver')['PitDuration'].agg(['sum', 'count']).sort_values(by='sum').reset_index()
                pit_summary.rename(columns={'sum': 'Toplam Süre (sn)', 'count': 'Pit Sayısı', 'Driver': 'Sürücü'}, inplace=True)
                st.info("Bu grafik, her sürücünün yarış boyunca pit yolunda geçirdiği toplam süreyi gösterir.")
                fig_pits = px.bar(pit_summary, x='Toplam Süre (sn)', y='Sürücü', orientation='h', title="Sürücülerin Pitlerde Geçirdiği Toplam Süre", color='Sürücü', color_discrete_map=driver_color_map, hover_data=['Pit Sayısı'])
                fig_pits.update_layout(yaxis={'categoryorder':'total ascending'})
                fig_pits.update_xaxes(range=[0, pit_summary['Toplam Süre (sn)'].max() * 1.1])
                st.plotly_chart(fig_pits, use_container_width=True)
            with pit_tab3:
                st.subheader("Pitteki Toplam Sürelerin Lidere Göre Farkı")
                st.info("Bu grafik, her sürücünün pitte en az zaman geçiren sürücüye kıyasla ne kadar 'ekstra' saniye harcadığını gösterir. Kısa çubuk daha iyi performansı belirtir.")
                if 'pit_summary' not in locals():
                    pit_summary = pits_df.groupby('Driver')['PitDuration'].agg(['sum', 'count']).sort_values(by='sum').reset_index()
                    pit_summary.rename(columns={'sum': 'Toplam Süre (sn)', 'count': 'Pit Sayısı', 'Driver': 'Sürücü'}, inplace=True)
                leader_time = pit_summary['Toplam Süre (sn)'].min()
                pit_summary['Fark (sn)'] = pit_summary['Toplam Süre (sn)'] - leader_time
                fig_delta_pits = px.bar(pit_summary.sort_values(by='Fark (sn)'), x='Fark (sn)', y='Sürücü', orientation='h', title="Pit Sürelerinin Lidere Göre Saniye Farkı", color='Sürücü', color_discrete_map=driver_color_map)
                fig_delta_pits.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_delta_pits, use_container_width=True)
        else:
            st.info("Bu yarışta pit stop verisi bulunamadı.")
        st.divider()

        st.header("🔎 Tek Sürücü Detay Analizi")
        all_drivers_map = dict(zip(results_df['Abbreviation'], results_df['FullName']))
        selected_driver_abbr = st.selectbox("Analiz için bir sürücü seçin:", options=results_df['Abbreviation'], format_func=lambda abbr: f"{all_drivers_map.get(abbr, abbr)} ({abbr})")
        if selected_driver_abbr:
            driver_results = results_df[results_df['Abbreviation'] == selected_driver_abbr].iloc[0]
            driver_laps = laps_df.pick_drivers([selected_driver_abbr])
            col1, col2 = st.columns([1.5, 2])
            with col1:
                st.subheader(f"{driver_results['FullName']}")
                st.markdown(f"**Takım:** {driver_results['TeamName']}")
                pos_gain = int(driver_results['PositionGain'])
                st.metric("Bitiş Pozisyonu", f"P{int(driver_results['Position'])}", f"{pos_gain} Pozisyon", delta_color=("inverse" if pos_gain > 0 else "normal"))
                st.metric("Alınan Puan", int(driver_results['Points']))
                st.metric("En Hızlı Tur", format_timedelta(driver_results['FastestLapTime']))
            with col2:
                st.subheader("Lastik Stratejisi")
                stints = driver_laps.groupby("Stint")
                stint_data = [{"Stint": num, "Lastik": laps['Compound'].iloc[0], "Başlangıç Turu": laps['LapNumber'].min(), "Tur Sayısı": len(laps)} for num, laps in stints]
                st.dataframe(pd.DataFrame(stint_data), use_container_width=True)
        st.divider()

        st.header("🔬 Gelişmiş Analiz: Telemetri Karşılaştırması")
        driver_list = results_df['Abbreviation'].tolist()
        default_drivers = driver_list[:2] if len(driver_list) >= 2 else driver_list
        selected_drivers_telemetry = st.multiselect("Hız profili karşılaştırması için sürücüleri seçin:", options=driver_list, default=default_drivers)
        
        if st.button("Hız Profillerini Karşılaştır"):
            if len(selected_drivers_telemetry) < 1:
                st.warning("Lütfen en az bir sürücü seçin.")
            else:
                with st.spinner("Telemetri verileri yükleniyor..."):
                    try:
                        telemetry_session = fastf1.get_session(year, gp_choice, 'R')
                        telemetry_session.load(telemetry=True, laps=True)
                        fig = go.Figure()
                        for driver in selected_drivers_telemetry:
                            lap = telemetry_session.laps.pick_drivers([driver]).pick_fastest()
                            if not pd.isna(lap['LapTime']):
                                tel_data = lap.get_car_data().add_distance()
                                fig.add_trace(go.Scatter(x=tel_data['Distance'], y=tel_data['Speed'], name=driver, mode='lines', line=dict(color=driver_color_map.get(driver))))
                        fig.update_layout(title="En Hızlı Tur Hız Karşılaştırması", xaxis_title="Pist Mesafesi (m)", yaxis_title="Hız (km/s)")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e: st.error(f"Telemetri verileri yüklenirken bir hata oluştu: {e}")
        
        if len(selected_drivers_telemetry) == 2:
            st.subheader(f"Hız Farkı (Delta) Analizi: {selected_drivers_telemetry[0]} vs {selected_drivers_telemetry[1]}")
            st.info(f"Bu grafik, **{selected_drivers_telemetry[1]}**'in **{selected_drivers_telemetry[0]}**'e göre hız avantajını gösterir. Çizgi sıfırın üzerindeyse {selected_drivers_telemetry[1]} daha hızlı, altındaysa {selected_drivers_telemetry[0]} daha hızlıdır.")
            if st.button("Hız Farkı Grafiğini Oluştur"):
                with st.spinner("Delta analizi için veriler işleniyor..."):
                    try:
                        telemetry_session = fastf1.get_session(year, gp_choice, 'R')
                        telemetry_session.load(telemetry=True, laps=True)
                        d1, d2 = selected_drivers_telemetry[0], selected_drivers_telemetry[1]
                        d1_lap = telemetry_session.laps.pick_drivers([d1]).pick_fastest()
                        d2_lap = telemetry_session.laps.pick_drivers([d2]).pick_fastest()
                        if pd.isna(d1_lap['LapTime']) or pd.isna(d2_lap['LapTime']):
                            st.error("Delta analizi için her iki sürücünün de geçerli bir en hızlı turu bulunmalıdır.")
                        else:
                            d1_tel = d1_lap.get_car_data().add_distance()
                            d2_tel = d2_lap.get_car_data().add_distance()
                            d2_speed_on_d1_dist = np.interp(d1_tel['Distance'], d2_tel['Distance'], d2_tel['Speed'])
                            speed_delta = d2_speed_on_d1_dist - d1_tel['Speed']
                            fig_delta = go.Figure()
                            fig_delta.add_trace(go.Scatter(x=d1_tel['Distance'], y=speed_delta, mode='lines', name=f"{d2} vs {d1} Hız Farkı", line=dict(color=driver_color_map.get(d2, 'white'))))
                            fig_delta.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
                            fig_delta.update_layout(title=f"{d1} ve {d2} Arasındaki Hız Farkı", xaxis_title="Pist Mesafesi (m)", yaxis_title=f"Hız Avantajı (km/s) - {d2}")
                            st.plotly_chart(fig_delta, use_container_width=True)
                    except Exception as e:
                        st.error(f"Delta analizi sırasında bir hata oluştu: {e}")
    else:
        st.warning("Bu GP için analiz edilecek yarış verisi bulunamadı.")
else:
    st.info("Lütfen analiz yapmak için sol menüden bir yıl ve Grand Prix seçin.")

st.divider()
st.header(f"📊 {year} Sezonu Şampiyona Analizi")
tab_champ1, tab_champ2, tab_champ3 = st.tabs(["🏆 Puan Durumu", "📈 Şampiyona Gidişatı", "📊 Grafiksel Puan Dağılımı"])
with st.spinner(f'{year} sezonu için şampiyona verileri yükleniyor...'):
    season_results_df = load_season_race_results(year)

if not season_results_df.empty:
    season_team_color_map = {row['TeamName']: (f"#{row['TeamColor']}" if row['TeamColor'] and len(row['TeamColor']) == 6 else TEAM_COLORS.get(row['TeamName'], "#FFFFFF")) for i, row in season_results_df.drop_duplicates(subset='TeamName').iterrows()}
    season_driver_color_map = {row['FullName']: season_team_color_map.get(row['TeamName'], "#FFFFFF") for i, row in season_results_df.drop_duplicates(subset='FullName').iterrows()}
    with tab_champ1:
        champ_col1, champ_col2 = st.columns(2)
        with champ_col1:
            st.subheader("Sürücüler Şampiyonası")
            driver_standings = season_results_df.groupby(['Abbreviation', 'FullName', 'TeamName'])['Points'].sum().sort_values(ascending=False).reset_index()
            driver_standings.index += 1
            st.dataframe(driver_standings[['FullName', 'TeamName', 'Points']], use_container_width=True)
        with champ_col2:
            st.subheader("Takımlar Şampiyonası")
            constructor_standings = season_results_df.groupby('TeamName')['Points'].sum().sort_values(ascending=False).reset_index()
            constructor_standings.index += 1
            st.dataframe(constructor_standings, use_container_width=True)
    with tab_champ2:
        data = season_results_df.sort_values(by='EventDate')
        driver_points = data.groupby(['EventName', 'EventDate', 'FullName'])['Points'].sum().reset_index()
        driver_points['CumulativePoints'] = driver_points.groupby('FullName')['Points'].cumsum()
        top_drivers = data.groupby('FullName')['Points'].sum().nlargest(5).index
        plot_data = driver_points[driver_points['FullName'].isin(top_drivers)]
        fig = px.line(plot_data, x='EventName', y='CumulativePoints', color='FullName', title="Şampiyona Liderliği Gidişatı (Top 5 Sürücü)", markers=True, color_discrete_map=season_driver_color_map)
        st.plotly_chart(fig, use_container_width=True)
    with tab_champ3:
        st.subheader("Şampiyona Puanlarının Grafiksel Dağılımı")
        driver_standings_chart = season_results_df.groupby('FullName')['Points'].sum().sort_values(ascending=False).reset_index().head(10)
        fig_drivers = px.bar(driver_standings_chart, x='Points', y='FullName', orientation='h', title="Sürücüler Şampiyonası Puan Durumu (İlk 10)", color='FullName', color_discrete_map=season_driver_color_map)
        fig_drivers.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_drivers, use_container_width=True)
        constructor_standings_chart = season_results_df.groupby('TeamName')['Points'].sum().sort_values(ascending=False).reset_index()
        fig_constructors = px.bar(constructor_standings_chart, x='Points', y='TeamName', orientation='h', title="Takımlar Şampiyonası Puan Durumu", color='TeamName', color_discrete_map=season_team_color_map)
        fig_constructors.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_constructors, use_container_width=True)
else:
    st.error(f"{year} yılı için şampiyona puan durumu verisi yüklenemedi.")