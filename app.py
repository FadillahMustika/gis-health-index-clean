#!/usr/bin/env python
# coding: utf-8

# In[8]:


for value in [1, 10, 30, 100]:
    print(f"Input all = {value}")
    print(prediksi_health_index(value, value, value, value, value))


# In[23]:


import gradio as gr
import numpy as np
import joblib
import pandas as pd
import os


# Load RF Health Index Model
model = joblib.load("xgboost_tuned_model.pkl")

# Tambahkan di awal (di luar fungsi mana pun)
primary_params = []
dielectric_params = []
mechanism_params = []
secondary_params = []
construction_params = []

# Dropdown manufacturer global
manufacturer_dropdown = gr.Dropdown(
    label="Manufacturer Name",
    choices=[
        "XIAN", "SIEMENS", "SPRECHER ENERGIE", "FUJI ELECTRIC", "EHH", "NISSIN ELECTRIC",
        "ALSTHOM", "GANZ ELECTRIC", "HITACHI", "ABB", "HOLEC", "AREVA",
        "MITSUBISHI", "HYOSUNG", "NHVS", "OTHER"
    ],
    value="OTHER"
)

def hitung_score_cumulative(actual):
    try:
        if actual is None or actual == "" or not isinstance(actual, (int, float)):
            return 0
        baseline = 20000
        ratio = float(actual) / baseline * 100
        if ratio < 20:
            return 1
        elif ratio < 40:
            return 3
        elif ratio < 70:
            return 10
        elif ratio < 100:
            return 30
        else:
            return 100
    except:
        return 0 

def hitung_score_res(latest, ref):
    if latest is None or ref is None or ref == 0:
        return 0
    ratio = abs((latest - ref) / ref) * 100  # Dalam persen
    if ratio < 5:
        return 1
    elif ratio < 10:
        return 10
    elif ratio < 20:
        return 30
    else:
        return 100

def hitung_score_hotspot(value):
    if value == "No Hot Spot":
        return 1
    elif value == "Hot Spot":
        return 100
    return 0

def hitung_score_primary(score_icum, score_sci, static_score, score_hotspot):
    skor_list = [s for s in [score_icum, score_sci, static_score, score_hotspot] if s is not None]
    return max(skor_list) if skor_list else 0

def hitung_score_gas_pressure(latest, ref):
    if latest is None or ref is None or ref == 0:
        return 0
    ratio = abs((latest - ref) / ref) * 100  # Ubah ke persen
    if ratio < 0.5:
        return 1
    elif ratio < 1:
        return 3
    elif ratio < 4:
        return 10
    elif ratio < 7:
        return 30
    else:
        return 100

def hitung_score_gas_density(latest, ref):
    return hitung_score_gas_pressure(latest, ref)  # Sama dengan gas pressure

def hitung_score_purity(value):
    if value is None:
        return 0
    if value >= 98.7:
        return 1
    elif value >= 97.8:
        return 10
    elif value >= 97.0:
        return 30
    else:
        return 100

def hitung_score_so2(value):
    if value is None:
        return 0
    if value < 1:
        return 1
    elif value < 4.6:
        return 10
    elif value < 10:
        return 30
    else:
        return 100

def hitung_score_sf6_byproduct(value):
    if value is None:
        return 0
    if value < 1:
        return 1
    elif value < 4.6:
        return 10
    elif value < 10:
        return 30
    else:
        return 100

def hitung_score_pd(nilai):
    mapping = {
        "PD Pattern: No; PD Growth: No": 1,
        "PD Pattern: Yes; PD Growth: No": 30,
        "PD Pattern: Yes; PD Growth: Yes": 100
    }
    skor = mapping.get(nilai, 0)
    return skor, skor  # satu untuk model, satu untuk ditampilkan

def hitung_score_humidity(humidity, cb_type):
    if humidity is None or cb_type is None:
        return 0
    # Cegah skor muncul jika belum input manual (masih default)
    if humidity == 0:
        return 0
    if cb_type == "CB":
        if humidity <= 135:
            return 1
        elif humidity <= 277:
            return 10
        elif humidity <= 336:
            return 30
        else:
            return 100
    elif cb_type == "NON CB":
        if humidity <= 209:
            return 1
        elif humidity <= 660:
            return 10
        elif humidity <= 804:
            return 30
        else:
            return 100
    return 0

def hitung_score_dew_point(value):
    if value is None:
        return 0
    return 1 if value <= -5 else 100

def hitung_score_dielectric(score_gas_pressure, score_gas_density, score_purity, score_so2, score_non_so2, pd_growth_score, humidity_score, dew_point_score):
    skor_list = [score_gas_pressure, score_gas_density, score_purity, score_so2, score_non_so2, pd_growth_score, humidity_score, dew_point_score]
    return max(s for s in skor_list if s is not None)

def hitung_score_mech_work(mech_work, manufacturer):
    if mech_work is None or mech_work <= 0 or not manufacturer:
        return 0, 0  # Jangan hitung jika belum diisi/mech_work nol

    batas_maks = {
        "XIAN": 10000, "SIEMENS": 6000, "SPRECHER ENERGIE": 2500,
        "FUJI ELECTRIC": 2000, "EHH": 2000, "NISSIN ELECTRIC": 10000,
        "ALSTHOM": 2500, "GANZ ELECTRIC": 10000, "HITACHI": 10000,
        "ABB": 10000, "HOLEC": 10000, "AREVA": 10000, "MITSUBISHI": 10000,
        "HYOSUNG": 10000, "NHVS": 10000, "OTHER": 10000
    }

    max_val = batas_maks.get(manufacturer.upper(), 10000)
    ratio = (mech_work / max_val) * 100
    if ratio < 5:
        skor = 1
    elif ratio < 10:
        skor = 3
    elif ratio < 50:
        skor = 10
    elif ratio < 100:
        skor = 30
    else:
        skor = 100
    return skor, skor

def hitung_score_gas_topup(freq):
    if freq == "No Leak":
        return 1
    elif freq == "1–2x/year":
        return 3
    elif freq == "3–12x/year":
        return 10
    elif freq == ">12x/year":
        return 100
    return 0

def hitung_score_open_close(open_latest, open_ref, close_latest, close_ref):
    if None in [open_latest, open_ref, close_latest, close_ref] or 0 in [open_ref, close_ref]:
        return 0
    open_pct = abs((open_latest - open_ref) / open_ref) * 100
    close_pct = abs((close_latest - close_ref) / close_ref) * 100
    max_pct = max(open_pct, close_pct)
    if max_pct < 2:
        return 1
    elif max_pct < 5:
        return 3
    elif max_pct < 10:
        return 10
    else:
        return 100

def hitung_score_travel(record):
    if record == "Good":
        return 1
    elif record == "Problem Found":
        return 100
    return 0  

def hitung_score_motor(latest, comparison):
    if latest is None or comparison in (None, 0):
        return 0
    pct = abs((latest - comparison) / comparison) * 100
    if pct < 2:
        return 1
    elif pct < 5:
        return 3
    elif pct < 15:
        return 10
    else:
        return 100

def hitung_score_mechanism(score_mech_work, score_gas_topup, score_open_close, score_travel, score_motor):
    skor_list = [score_mech_work, score_gas_topup, score_open_close, score_travel, score_motor]
    return max(s for s in skor_list if s is not None)

def hitung_score_corrosion_lcc(nilai):
    mapping = {
        "No Corrosion": 1,
        "Slight Corrosion": 3,
        "Severe Corrosion": 30,
        "Massive Corrosion": 100
    }
    skor = mapping.get(nilai, 0)
    return skor, skor

def hitung_score_dust_lcc(nilai):
    mapping = {
        "No Dust": 1,
        "Slight Dust": 3,
        "Severe Dust": 30,
        "Massive Dust": 100
    }
    skor = mapping.get(nilai, 0)
    return skor, skor  # Untuk model + untuk tampil

def hitung_score_hot_spot_lcc(hot_spot_lcc):
    if hot_spot_lcc == "With Hot Spot":
        skor = 100
    elif hot_spot_lcc == "No Hot Spot":
        skor = 1
    else:
        skor = 0
    return skor, skor

def hitung_score_rele_function(nilai):
    mapping = {
        "All OK": 1,
        "Any indicator fails": 30,
        "Any relay fails": 100
    }
    skor = mapping.get(nilai, 0)
    return skor, skor

def hitung_score_secondary(corrosion_lcc, dust_lcc, hot_spot_lcc, rele_function):
    skor_corrosion_lcc, _ = hitung_score_corrosion_lcc(corrosion_lcc)
    skor_dust_lcc, _ = hitung_score_dust_lcc(dust_lcc)
    skor_hot_spot_lcc, _ = hitung_score_hot_spot_lcc(hot_spot_lcc)
    skor_rele_function, _ = hitung_score_rele_function(rele_function)

    skor_list = [skor_corrosion_lcc, skor_dust_lcc, skor_hot_spot_lcc, skor_rele_function]
    return max(s for s in skor_list if s is not None)

def hitung_score_corrosion(condition):
    if condition == "As good as new":
        return 1
    elif condition == "Slight corrosion, No leaks":
        return 3
    elif condition == "Moderate Corrosion, No leaks":
        return 10
    elif condition == "Severe Corrosion, Small leaks":
        return 30
    elif condition == "Catastrophic Corrosion, Big leaks":
        return 100
    return 0

def hitung_score_pollutant(condition):
    if condition == "As good as new":
        return 1
    elif condition == "Slightly polluted":
        return 3
    elif condition == "Moderately Polluted":
        return 10
    elif condition == "Severely Polluted":
        return 30
    elif condition == "Catastrophic":
        return 100
    return 0

def hitung_score_foundation(condition):
    if condition == "No Crack":
        return 1
    elif condition == "With Crack":
        return 100
    return 0

def hitung_score_construction(score_corrosion, score_pollutant, score_foundation):
    skor_list = [score_corrosion, score_pollutant, score_foundation]
    return max(s for s in skor_list if s is not None)

# Fungsi rekomendasi berbasis kategori HI
def get_rekomendasi(kategori):
    rekomendasi_map = {
        "VERY GOOD": "As good as new, no evidence of ageing or deterioration. No action needed.",
        "GOOD": """Slight deterioration/ ageing process is observed, but it is considered at normal stage.
Minor defect may be observed, but it does not influence the GIS performance both in short and longer terms.""",
        "MODERATE": """Deterioration/ aging process has been observed beyond the normal stage.
Intervention is required as deterioration/ aging may interfere the GIS performance in long-term.""",
        "BAD": "Severe deterioration/ aging has been observed. Intervention is required in short-term.",
        "VERY BAD": "Very severe deterioration/ aging (i.e. at a final stage) has been observed. Emergency action is required."
    }
    return rekomendasi_map.get(kategori, "Unknown condition. Further assessment needed.")


# Fungsi prediksi HI
def prediksi_health_index(primary, dielectric, mechanism, secondary, construction):
    input_dict = {
        "PRIMARY": primary,
        "DILECTRIC": dielectric,
        "MECHANICAL": mechanism,
        "SECONDARY": secondary,
        "CONSTRUCTION": construction
    }
    df_input = pd.DataFrame([input_dict])
    missing_cols = set(model.feature_names_in_) - set(df_input.columns)
    for col in missing_cols:
        df_input[col] = 0
    df_input = df_input[model.feature_names_in_]

    prediksi_label = model.predict(df_input)[0]

    label_map = {
        0: "VERY BAD",
        1: "BAD",
        2: "MODERATE",
        3: "GOOD",
        4: "VERY GOOD"
    }

    return label_map.get(prediksi_label, "Unknown")

# Fungsi utama untuk update skor dan prediksi HI
def update_scores(*args):
    if len(args) < 5:
        return 0, "-", "", "0% data terisi"

    primary, dielectric, mechanism, secondary, construction = args[:5]
    all_fields = args[5:]

    input_values = [primary, dielectric, mechanism, secondary, construction]
    filled_fields = sum(isinstance(v, (int, float)) and v > 0 for v in input_values)

    if filled_fields == 0:
        return 0, "-", "", "0% data terisi"

    ahi = sum(v if isinstance(v, (int, float)) else 0 for v in input_values)

    kategori_asli = prediksi_health_index(
        primary or 0,
        dielectric or 0,
        mechanism or 0,
        secondary or 0,
        construction or 0
    )

    kategori_tampil = kategori_asli
    if filled_fields < 5:
        kategori_tampil += " (data tidak lengkap)"

    rekomendasi = get_rekomendasi(kategori_asli)
    total_fields = len(all_fields)
    filled_total = sum(bool(v) for v in all_fields)
    persen = int((filled_total / total_fields) * 100) if total_fields > 0 else 0

    return ahi, kategori_tampil, rekomendasi, f"{persen}% data terisi"

# Fungsi untuk menyimpan history
def simpan_history(gis, tegangan, bay, manufacturer, primary, dielectric, mechanism, secondary, construction, ahi, prediksi_hi, rekomendasi, persentase):
    riwayat_baru = pd.DataFrame([{
        "GIS": gis,
        "Tegangan": tegangan,
        "Bay": bay,
        "Pabrikan": manufacturer,
        "Primary": primary,
        "Dielectric": dielectric,
        "Mechanism": mechanism,
        "Secondary": secondary,
        "Construction": construction,
        "AHI": ahi,
        "Prediksi HI": prediksi_hi,
        "Rekomendasi": rekomendasi,
        "Persentase Kelengkapan Data": persentase
    }])

    if os.path.exists("hi_prediction_history.csv"):
        riwayat_lama = pd.read_csv("hi_prediction_history.csv")
        riwayat = pd.concat([riwayat_lama, riwayat_baru], ignore_index=True)
    else:
        riwayat = riwayat_baru

    riwayat.to_csv("hi_prediction_history.csv", index=False)
    return "✅ History successfully saved!"

score_primary_total = gr.Number(visible=False)
score_dielectric_total = gr.Number(visible=False)
score_driving_total = gr.Number(visible=False)
score_secondary_total = gr.Number(visible=False)
score_construction_total = gr.Number(visible=False)

clear_button = gr.Button("Clear All")

def clear_all():
    return (
        None, None, None, None, None,  # 5 skor subsystem
        None, None, None, None, None,  # 5 primary
        None, None, None, None,        # 4 gas
        None, None, None, None,        # 4 sf6/pd
        None, None, None,              # cb_type, humidity_sf6, dew_point
        None, None, None, None, None, None, None,  # 7 mekanisme
        None, None,                    # motor_latest, motor_comp
        None, None, None, None,        # 4 secondary
        None, None, None,              # 3 construction
        None,                          # ahi_score
        "", "", "",                    # 4 teks output
        ""                             # <== tambahan untuk notif!
    )


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("### 🔌 GIS Health Index Prediction System")

    # Tambahkan CSS kecilkan font, kotak luar, dan palet abu-abu dengan font profesional
    gr.HTML("""
    <style>
      * {
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
        font-size: 10px !important;
        color: #222222 !important;
      }
      body, .gr-container {
        background-color: #F2F2F2 !important;
        max-width: 600px;
        margin: auto;
      }
      label {
        color: #333333 !important;
        margin-bottom: 2px !important;
      }
      input, select, textarea {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-color: #CCCCCC !important;
        margin-bottom: 4px !important;
        padding: 4px !important;
      }
      .gr-form, .gr-box, .gr-block.gr-input {
        margin-bottom: 4px !important;
      }
    </style>
    """)
    with gr.Tabs():
        with gr.Tab("Identitas GIS"):
            gis_name = gr.Textbox(label="GIS Name")
            voltage = gr.Number(label="Voltage (kV)")
            bay = gr.Textbox(label="Bay Name")
            manufacturer = gr.Dropdown(
                label="Manufacturer",
                choices=[
                    "XIAN", "SIEMENS", "SPRECHER ENERGIE", "FUJI ELECTRIC", "EHH",
                    "NISSIN ELECTRIC", "ALSTHOM", "GANZ ELECTRIC", "HITACHI", "ABB",
                    "HOLEC", "AREVA", "MITSUBISHI", "HYOSUNG", "NHVS", "OTHER"
                ],
                value=None
            )

        with gr.Tab("Primary Subsystem"):
            with gr.Accordion("Primary Subsystem", open=True):
                gr.Markdown("**Cumulative Short Circuit Current**")
                with gr.Row():
                    primary_icum = gr.Number(label="Data Record")
                    score_icum = gr.Number(label="Cumulative Short Circuit Current", visible=False)
                    score_icum_display = gr.Number(label="Score", interactive=True)

                primary_icum.change(
                    lambda val: (hitung_score_cumulative(val), hitung_score_cumulative(val)),
                    inputs=primary_icum,
                    outputs=[score_icum, score_icum_display]
                )

                gr.Markdown("**Short Circuit Interruption**")
                with gr.Row():
                    primary_sci = gr.Number(label="Data Record")
                    score_sci = gr.Number(label="Short Circuit Interruption", visible=False)
                    score_sci_display = gr.Number(label="Score", interactive=True)

                primary_sci.change(
                    lambda val: (hitung_score_cumulative(val), hitung_score_cumulative(val)),
                    inputs=primary_sci,
                    outputs=[score_sci, score_sci_display]
                )

                gr.Markdown("**Static Contact Resistance**")
                with gr.Row():
                    static_latest = gr.Number(label="Latest Test Data (µΩ)")
                    static_compare = gr.Number(label="Comparison Data (µΩ)")
                    static_score = gr.Number(label="Static Contact Resistance", visible=False)
                    static_score_display = gr.Number(label="Score", visible=True)

                static_latest.change(
                    lambda latest, ref: (hitung_score_res(latest, ref), hitung_score_res(latest, ref)),
                    inputs=[static_latest, static_compare],
                    outputs=[static_score, static_score_display]
                )
                static_compare.change(
                    lambda latest, ref: (hitung_score_res(latest, ref), hitung_score_res(latest, ref)),
                    inputs=[static_latest, static_compare],
                    outputs=[static_score, static_score_display]
                )

                with gr.Row():
                    primary_hotspot = gr.Radio(label="Hot Spot on the enclosure", choices=["No Hot Spot", "Hot Spot"])
                    score_hotspot = gr.Number(label="Score", visible=False)
                    score_hotspot_display = gr.Number(label="Hot Spot on the enclosure", visible=True)

                primary_hotspot.change(
                    lambda val: (hitung_score_hotspot(val), hitung_score_hotspot(val)),
                    inputs=primary_hotspot,
                    outputs=[score_hotspot, score_hotspot_display]
                )

        with gr.Tab("Dielectric Subsystem"):
            with gr.Accordion("Dielectric Subsystem", open=True):

                gr.Markdown("**Gas Pressure**")
                with gr.Row():
                    gas_pressure_latest = gr.Number(label="Latest Test Data")
                    gas_pressure_compare = gr.Number(label="Comparison Data ")
                    gas_pressure_score = gr.Number(label="Gas Pressure", visible=False)
                    gas_pressure_score_display = gr.Number(label="Score", visible=True)

                gas_pressure_latest.change(
                    lambda latest, ref: [hitung_score_gas_pressure(latest, ref)] * 2,
                    inputs=[gas_pressure_latest, gas_pressure_compare],
                    outputs=[gas_pressure_score, gas_pressure_score_display]
                )

                gas_pressure_compare.change(
                    lambda latest, ref: [hitung_score_gas_pressure(latest, ref)] * 2,
                    inputs=[gas_pressure_latest, gas_pressure_compare],
                    outputs=[gas_pressure_score, gas_pressure_score_display]
                )

                gr.Markdown("**Gas Density**")
                with gr.Row():
                    gas_density_latest = gr.Number(label="Latest Test Data")
                    gas_density_compare = gr.Number(label="Comparison Data")
                    gas_density_score = gr.Number(label="Gas Density", visible=False)
                    gas_density_score_display = gr.Number(label="score", visible=True)

                gas_density_latest.change(
                    lambda latest, ref: [hitung_score_gas_density(latest, ref)] * 2,
                    inputs=[gas_density_latest, gas_density_compare],
                    outputs=[gas_density_score, gas_density_score_display]
                )

                gas_density_compare.change(
                    lambda latest, ref: [hitung_score_gas_density(latest, ref)] * 2,
                    inputs=[gas_density_latest, gas_density_compare],
                    outputs=[gas_density_score, gas_density_score_display]
                )

                gr.Markdown("**SF6 Purity**")
                with gr.Row():
                    sf6_purity = gr.Number(label="Latest Test Data (%)")
                    score_purity = gr.Number(label="Purity", visible=False)
                    score_purity_display = gr.Number(label="Score", visible=True)

                sf6_purity.change(
                    lambda val: [hitung_score_purity(val)] * 2,
                    inputs=sf6_purity,
                    outputs=[score_purity, score_purity_display]
                )

                gr.Markdown("**SO2 Content**")
                with gr.Row():
                    so2_content = gr.Number(label="Latest Test Data (ppmV)")
                    score_so2 = gr.Number(label="SO2", visible=False)
                    score_so2_display = gr.Number(label="Score", visible=True)

                so2_content.change(
                    lambda val: [hitung_score_so2(val)] * 2,
                    inputs=so2_content,
                    outputs=[score_so2, score_so2_display]
                )

                gr.Markdown("**SF6 By-product other than SO2**")
                with gr.Row():
                    non_so2 = gr.Number(label="Latest Test Data (%)")
                    score_non_so2 = gr.Number(label="SF6 By-product other than SO2 (%)", visible=False)
                    score_non_so2_display = gr.Number(label="Score", visible=True)

                non_so2.change(
                    lambda val: [hitung_score_sf6_byproduct(val)] * 2,
                    inputs=non_so2,
                    outputs=[score_non_so2, score_non_so2_display]
                )

                with gr.Row():
                    pd_growth = gr.Dropdown(
                        label="PD Pattern & PD Growth",
                        choices=[
                            "PD Pattern: No; PD Growth: No",
                            "PD Pattern: Yes; PD Growth: No",
                            "PD Pattern: Yes; PD Growth: Yes"
                        ]
                    )
                    pd_growth_score = gr.Number(label="PD Pattern PD Growth", visible=False)
                    pd_growth_score_display = gr.Number(label="Score", visible=True)

                    pd_growth.change(hitung_score_pd, inputs=pd_growth, outputs=[pd_growth_score, pd_growth_score_display])

                gr.Markdown("**Humidity in SF6**")
                with gr.Row():
                    cb_type = gr.Dropdown(choices=["CB", "NON CB"], label="CB Type")
                    humidity_sf6 = gr.Number(label="Test Data (ppmV)")
                    humidity_score = gr.Number(label="Humidity", visible=False)
                    humidity_score_display = gr.Number(label="Score", visible=True)

                cb_non_cb_selector = gr.Number(visible=False)

                cb_type.change(
                    lambda humidity, cb: [hitung_score_humidity(humidity, cb)] * 2,
                    inputs=[humidity_sf6, cb_type],
                    outputs=[humidity_score, humidity_score_display]
                )

                humidity_sf6.change(
                    lambda humidity, cb: [hitung_score_humidity(humidity, cb)] * 2,
                    inputs=[humidity_sf6, cb_type],
                    outputs=[humidity_score, humidity_score_display]
                )

                cb_type.change(
                    lambda val: 1 if val == "NON CB" else 0,
                    inputs=cb_type,
                    outputs=cb_non_cb_selector
                )

                gr.Markdown("**Dew Point**")
                with gr.Row():
                    dew_point_measured = gr.Number(label="Test Data (°C)")
                    dew_point_score = gr.Number(label="Dew Point", visible=False)
                    dew_point_display = gr.Number(label="Score", visible=True)

                dew_point_measured.change(
                    lambda val: [hitung_score_dew_point(val)] * 2,
                    inputs=dew_point_measured,
                    outputs=[dew_point_score, dew_point_display]
                )

        with gr.Tab("Driving Mechanism Subsystem"):
            with gr.Accordion("Driving Mechanism Subsystem", open=True):

                gr.Markdown("**Number of Mechanical Work**")
                with gr.Row():
                    mech_work = gr.Number(label="Latest Test Data")
                    score_mech_work = gr.Number(label="Mechanical Work", visible=False)
                    score_mech_work_display = gr.Number(label="Score", interactive=True)

                manufacturer.change(
                    hitung_score_mech_work, 
                    inputs=[mech_work, manufacturer], 
                    outputs=[score_mech_work, score_mech_work_display]
                )
                mech_work.change(
                    hitung_score_mech_work, 
                    inputs=[mech_work, manufacturer], 
                    outputs=[score_mech_work, score_mech_work_display]
                )

                with gr.Row():
                    gas_topup = gr.Radio(
                        label="Compressor Tightness", 
                        choices=["No Leak", "1–2x/year", "3–12x/year", ">12x/year"]
                    )
                    score_gas_topup = gr.Number(label="Gas Replenishment Frequency", visible=False)
                    score_gas_topup_display = gr.Number (label="Score", visible=True)

                gas_topup.change(
                    lambda val: [hitung_score_gas_topup(val)] * 2,
                    inputs=[gas_topup],
                    outputs=[score_gas_topup, score_gas_topup_display]
                )

                gr.Markdown("**Opening & Closing**")
                with gr.Row():
                    open_latest = gr.Number(label="Latest Opening Time (ms)")
                    open_ref = gr.Number(label="Reference Opening Time (ms)")
                with gr.Row():    
                    close_latest = gr.Number(label="Latest Closing Time (ms)")
                    close_ref = gr.Number(label="Reference Closing Time (ms)")
                with gr.Row():   
                    score_open_close = gr.Number(label="Opening Closing", visible=False)
                    score_open_close_display = gr.Number(label="Score", visible=True)

                for comp in [open_latest, open_ref, close_latest, close_ref]:
                    comp.change(
                        lambda a, b, c, d: [hitung_score_open_close(a, b, c, d)] * 2,
                        inputs=[open_latest, open_ref, close_latest, close_ref],
                        outputs=[score_open_close, score_open_close_display]
                    )

                with gr.Row():
                    contact_travel = gr.Radio(label="Contact Travel Record", choices=["Good", "Problem Found"])
                    score_travel = gr.Number(label="CTR", visible=False)
                    score_travel_display = gr.Number(label="Score", visible=True)
                contact_travel.change(
                    lambda val: [hitung_score_travel(val)] * 2,
                    inputs=[contact_travel],
                    outputs=[score_travel, score_travel_display]
                )

                gr.Markdown("**Motor Current**")
                with gr.Row():
                    motor_latest = gr.Number(label="Latest Data Test (%)")
                    motor_comp = gr.Number(label="Comparison Data (%)")
                    score_motor = gr.Number(label="Motor Current", visible=False)
                    score_motor_display = gr.Number(label="Score", visible=True)
                motor_latest.change(
                    lambda latest, ref: [hitung_score_motor(latest, ref)] * 2,
                    inputs=[motor_latest, motor_comp],
                    outputs=[score_motor, score_motor_display]
                )
                motor_comp.change(
                    lambda latest, ref: [hitung_score_motor(latest, ref)] * 2,
                    inputs=[motor_latest, motor_comp],
                    outputs=[score_motor, score_motor_display]
                )

        with gr.Tab("Secondary Subsystem"):
            with gr.Accordion("Secondary Subsystem", open=True):
                with gr.Row():
                    corrosion_lcc = gr.Dropdown(
                        choices=["No Corrosion", "Slight Corrosion", "Severe Corrosion", "Massive Corrosion"], 
                        label="Corrosion LCC"
                    )
                    score_corrosion_lcc = gr.Number(label="Corrosion LCC", visible=False)
                    score_corrosion_lcc_display = gr.Number(label="Score", visible=True)

                    corrosion_lcc.change(
                        hitung_score_corrosion_lcc,
                        inputs=[corrosion_lcc],
                        outputs=[score_corrosion_lcc, score_corrosion_lcc_display]
                    )

                with gr.Row():
                    dust_lcc = gr.Dropdown(
                        choices=["No Dust", "Slight Dust", "Severe Dust", "Massive Dust"], 
                        label="Dust LCC"
                    )
                    score_dust_lcc = gr.Number(label="Dust LCC", visible=False)
                    score_dust_lcc_display = gr.Number(label="Score", visible=True)
                dust_lcc.change(
                    hitung_score_dust_lcc,
                    inputs=[dust_lcc],
                    outputs=[score_dust_lcc, score_dust_lcc_display]
                )
                with gr.Row():
                    hot_spot_lcc = gr.Dropdown(
                        choices=["No Hot Spot", "With Hot Spot"], 
                        label="Hot Spot LCC"
                    )
                    score_hotspot_lcc = gr.Number(label="Hot Spot LCC", visible=False)
                    score_hotspot_lcc_display = gr.Number(label="Score", visible=True)

                hot_spot_lcc.change(
                    hitung_score_hot_spot_lcc,
                    inputs=[hot_spot_lcc],
                    outputs=[score_hotspot_lcc, score_hotspot_lcc_display]
                )

                with gr.Row():
                    rele_function = gr.Dropdown(
                        choices=["All OK", "Any indicator fails", "Any relay fails"], 
                        label="Rele Function"
                    )
                    score_rele_function = gr.Number(label="Rele Function", visible=False)
                    score_rele_function_display = gr.Number(label="Score", visible=True)

                rele_function.change(
                    hitung_score_rele_function,
                    inputs=[rele_function],
                    outputs=[score_rele_function, score_rele_function_display]
                )

        with gr.Tab("Construction & Support"):
            with gr.Accordion("Construction & Support", open=True):
                with gr.Row():
                    corrosion = gr.Radio(label="Corrosion Level", choices=[
                        "As good as new",
                        "Slight corrosion, No leaks",
                        "Moderate Corrosion, No leaks",
                        "Severe Corrosion, Small leaks",
                        "Catastrophic Corrosion, Big leaks"
                    ])
                    score_corrosion = gr.Number(label="Corrosion Level", visible=False)
                    score_corrosion_display = gr.Number(label="Score", visible=True)

                corrosion.change(
                    lambda val: [hitung_score_corrosion(val)] * 2,
                    inputs=corrosion,
                    outputs=[score_corrosion, score_corrosion_display]
                )

                with gr.Row():
                    pollutant = gr.Radio(label="Deposited Pollutants", choices=[
                        "As good as new",
                        "Slightly polluted",
                        "Moderately Polluted",
                        "Severely Polluted",
                        "Catastrophic"
                    ])
                    score_pollutant = gr.Number(label="Deposited Pollutant", visible=False)
                    score_pollutant_display = gr.Number(label="Score", visible=True)

                pollutant.change(
                    lambda val: [hitung_score_pollutant(val)] * 2,
                    inputs=pollutant,
                    outputs=[score_pollutant, score_pollutant_display]
                )

                with gr.Row():
                    foundation = gr.Radio(label="Foundation Integrity", choices=[
                        "No Crack", 
                        "With Crack"
                    ])
                    score_foundation = gr.Number(label="Foundation strength", visible=False)
                    score_foundation_display = gr.Number(label="Score", visible=True)

                foundation.change(
                    lambda val: [hitung_score_foundation(val)] * 2, 
                    inputs=foundation, 
                    outputs=[score_foundation, score_foundation_display]
                )

        with gr.Tab("Health Index Analysis"):
            with gr.Column():
                # Parameter per subsistem
                primary_params.clear()
                primary_params.extend([
                    score_icum, 
                    score_sci, 
                    static_score, 
                    score_hotspot, 
                    score_primary_total
                ])

                dielectric_params.clear()
                dielectric_params.extend([
                    gas_pressure_score, 
                    gas_density_score, 
                    score_purity, 
                    score_so2,
                    score_non_so2, 
                    pd_growth_score, 
                    humidity_score, 
                    dew_point_score,
                    score_dielectric_total
                ])

                mechanism_params.clear()
                mechanism_params.extend([
                    score_mech_work,
                    score_gas_topup,
                    score_open_close,
                    score_travel,
                    score_motor,
                    score_driving_total
                ])

                secondary_params.clear()
                secondary_params.extend([
                    score_corrosion_lcc, 
                    score_dust_lcc, 
                    score_hotspot_lcc, 
                    score_rele_function,
                    score_secondary_total
                ])

                construction_params.clear()
                construction_params.extend([
                    score_corrosion, 
                    score_pollutant, 
                    score_foundation,
                    score_construction_total
                ])

                # Total score Primary
                with gr.Row():
                    score_primary_total = gr.Number(label="Primary Subsystem Score", visible=True)

                for comp in [score_icum, score_sci, static_score, score_hotspot]:
                    comp.change(
                        hitung_score_primary,
                        inputs=[score_icum, score_sci, static_score, score_hotspot],
                        outputs=score_primary_total
                    )
                # Total score Dielectric
                with gr.Row():
                    score_dielectric_total = gr.Number(label="Dielectric Subsystem Score", visible=True)

                for comp in [gas_pressure_score, gas_density_score, score_purity, score_so2,
                    score_non_so2, pd_growth_score, humidity_score, dew_point_score]:
                    comp.change(
                        hitung_score_dielectric,
                        inputs=[gas_pressure_score, gas_density_score, score_purity, score_so2,
                                score_non_so2, pd_growth_score, humidity_score, dew_point_score],
                        outputs=score_dielectric_total
                    )
                # Total score Driving
                with gr.Row():
                    score_driving_total = gr.Number(label="Driving Mechanism Subsystem Score", visible=True)

                for comp in [score_mech_work, score_gas_topup, score_open_close, score_travel, score_motor]:
                    comp.change(
                        lambda a, b, c, d, e: max([v for v in [a, b, c, d, e] if isinstance(v, (int, float))]),
                        inputs=[score_mech_work, score_gas_topup, score_open_close, score_travel, score_motor],
                        outputs=score_driving_total
                    )
                # Total score Secondary
                with gr.Row():
                    score_secondary_total = gr.Number(label="Secondary Subsystem Score", visible=True)

                for comp in [corrosion_lcc, dust_lcc, hot_spot_lcc, rele_function]:
                    comp.change(
                        hitung_score_secondary,
                        inputs=[corrosion_lcc, dust_lcc, hot_spot_lcc, rele_function],
                        outputs=score_secondary_total
                    )        
                # Total score Construction
                with gr.Row():
                    score_construction_total = gr.Number(label="Construction & Support Subsystem Score", visible=True)

                for comp in [score_corrosion, score_pollutant, score_foundation]:
                    comp.change(
                        lambda a, b, c: max(a, b, c),
                        inputs=[score_corrosion, score_pollutant, score_foundation],
                        outputs=score_construction_total
                    )

                ahi_score = gr.Number(label="Asset Health Index (AHI)", interactive=False)
                prediksi_hi_output = gr.Textbox(label="Health Index Prediction (ML)", interactive=False)
                rekomendasi_output = gr.Textbox(label="Description", lines=3, interactive=False)
                completeness_total_output = gr.Textbox(label="Total Data Completeness (%)", interactive=False)

                submit_button = gr.Button("Count All")
                simpan_button = gr.Button("Save History")
                clear_button = gr.Button("Clear All")
                notif = gr.Textbox(label="Status", interactive=False)


                submit_button.click(
                    fn=update_scores,
                    inputs=[
                        score_primary_total,
                        score_dielectric_total,
                        score_driving_total,
                        score_secondary_total,
                        score_construction_total,
                        primary_icum, primary_sci, static_latest, static_compare, primary_hotspot,       
                        gas_pressure_latest, gas_pressure_compare, gas_density_latest, gas_density_compare, 
                        sf6_purity, so2_content, non_so2, pd_growth,
                        cb_type, humidity_sf6, dew_point_measured, 
                        mech_work, gas_topup, open_latest, open_ref, close_latest, 
                        close_ref, contact_travel, motor_latest, motor_comp,            
                        corrosion_lcc, dust_lcc, hot_spot_lcc, rele_function, 
                        corrosion, pollutant, foundation
                    ],
                    outputs=[ahi_score, prediksi_hi_output, rekomendasi_output, completeness_total_output]
                )
                simpan_button.click(
                    simpan_history,
                    inputs=[
                        gis_name, voltage, bay, manufacturer,
                        score_primary_total, score_dielectric_total, score_driving_total, 
                        score_secondary_total, score_construction_total,
                        ahi_score, prediksi_hi_output, rekomendasi_output, completeness_total_output
                    ],
                    outputs=notif
                )

                clear_button.click(
                    fn=clear_all,
                    inputs=[],
                    outputs=[
                        score_primary_total, score_dielectric_total, score_driving_total,
                        score_secondary_total, score_construction_total,
                        primary_icum, primary_sci, static_latest, static_compare, primary_hotspot,
                        gas_pressure_latest, gas_pressure_compare, gas_density_latest, gas_density_compare,
                        sf6_purity, so2_content, non_so2, pd_growth,
                        cb_type, humidity_sf6, dew_point_measured,
                        mech_work, gas_topup, open_latest, open_ref, close_latest,
                        close_ref, contact_travel, motor_latest, motor_comp,
                        corrosion_lcc, dust_lcc, hot_spot_lcc, rele_function,
                        corrosion, pollutant, foundation,
                        ahi_score, prediksi_hi_output, rekomendasi_output, completeness_total_output,
                        notif
                    ]
                )    
demo.launch()


# In[ ]:




