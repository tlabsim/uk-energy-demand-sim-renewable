from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def fill_missing(values):
    values = np.asarray(values, dtype=float)
    idx = np.arange(len(values))
    good = ~np.isnan(values)
    return np.interp(idx, idx[good], values[good])

def moving_average(values, window=41):
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")

def extract_profiles_from_figure(image_path):
    img = np.array(Image.open(image_path).convert("RGB"))

    x_start, x_end = 203, 1183
    y_top, y_zero, y_five = 80, 527, 130

    crop = img[y_top:y_zero + 1, x_start:x_end + 1]

    red_mask = (
        (crop[:, :, 0] > 180) &
        (crop[:, :, 1] < 120) &
        (crop[:, :, 2] < 120)
    )

    wind_mask = (
        (crop[:, :, 2] > 220) &
        (crop[:, :, 1] > 190) &
        (crop[:, :, 0] > 100) &
        (crop[:, :, 0] < 200)
    )

    demand_y = np.full(crop.shape[1], np.nan)
    wind_top_y = np.full(crop.shape[1], np.nan)

    for x in range(crop.shape[1]):
        red_pixels = np.where(red_mask[:, x])[0]
        if red_pixels.size:
            demand_y[x] = np.median(red_pixels)

        wind_pixels = np.where(wind_mask[:, x])[0]
        if wind_pixels.size:
            wind_top_y[x] = wind_pixels.min()

    demand_y = fill_missing(demand_y)
    wind_top_y = fill_missing(wind_top_y)

    pixels_per_gw = (y_zero - y_five) / 5.0
    demand_abs_y = demand_y + y_top
    wind_abs_y = wind_top_y + y_top

    demand_gw = (y_zero - demand_abs_y) / pixels_per_gw
    wind_gw = (y_zero - wind_abs_y) / pixels_per_gw

    hours = np.linspace(0, 24, crop.shape[1])

    return pd.DataFrame({
        "hour": hours,
        "demand_gw_raw": demand_gw,
        "wind_gw_raw": wind_gw,
    })

def make_high_res_demand(hours, extracted_hours, extracted_demand):
    base = np.interp(hours, extracted_hours, moving_average(extracted_demand, 31))

    fluct = (
        0.09 * np.sin(2 * np.pi * hours / 0.75 + 0.50) +
        0.06 * np.sin(2 * np.pi * hours / 1.40 + 1.30) +
        0.03 * np.sin(2 * np.pi * hours / 2.80 + 2.20)
    )

    envelope = (
        0.65
        + 0.18 * np.exp(-0.5 * ((hours - 9.5) / 2.3) ** 2)
        + 0.15 * np.exp(-0.5 * ((hours - 18.5) / 2.6) ** 2)
    )

    demand = base + envelope * fluct
    return np.clip(demand, 0, None)

def make_solar_with_clouds(hours):
    base = 4.0 * np.sin(np.pi * (hours - 6.0) / 12.0)
    base = np.clip(base, 0, None)

    cloud_mod = (
        1.0
        - 0.07 * np.exp(-0.5 * ((hours - 9.2) / 0.55) ** 2)
        - 0.05 * np.exp(-0.5 * ((hours - 11.1) / 0.40) ** 2)
        - 0.08 * np.exp(-0.5 * ((hours - 13.4) / 0.60) ** 2)
        - 0.04 * np.exp(-0.5 * ((hours - 15.0) / 0.45) ** 2)
    )

    solar = base * cloud_mod
    solar = moving_average(solar, 5)
    return np.clip(solar, 0, None)

def simulate_energy_balance(extracted_df):
    dt = 0.25
    hours = np.arange(0, 24, dt)

    demand = make_high_res_demand(
        hours,
        extracted_df["hour"].to_numpy(),
        extracted_df["demand_gw_raw"].to_numpy()
    )

    wind_template = np.interp(
        hours,
        extracted_df["hour"],
        moving_average(extracted_df["wind_gw_raw"].to_numpy(), 41)
    )
    wind = 0.52 * wind_template

    solar = make_solar_with_clouds(hours)
    total_generation = wind + solar

    p_max = 1.6
    e_max = 5.8
    eta_c = 0.95
    eta_d = 0.95
    soc = 2.8

    battery_power = np.zeros_like(hours)
    soc_trace = np.zeros_like(hours)

    for i, gap in enumerate(demand - total_generation):
        if gap < 0:
            available_charge = min(-gap, p_max)
            charge_limit = (e_max - soc) / (eta_c * dt)
            p_charge = max(0.0, min(available_charge, charge_limit))
            battery_power[i] = -p_charge
            soc += p_charge * eta_c * dt
        else:
            required_discharge = min(gap, p_max)
            discharge_limit = soc * eta_d / dt
            p_discharge = max(0.0, min(required_discharge, discharge_limit))
            battery_power[i] = p_discharge
            soc -= (p_discharge / eta_d) * dt
        soc_trace[i] = soc

    return pd.DataFrame({
        "hour": hours,
        "demand_gw": demand,
        "wind_gw": wind,
        "solar_gw": solar,
        "total_generation_gw": total_generation,
        "battery_discharge_gw": np.where(battery_power > 0, battery_power, 0),
        "battery_charge_gw": np.where(battery_power < 0, battery_power, 0),
        "battery_soc_gwh": soc_trace,
    })

def main():
    source_image = Path("original_figure.png")

    extracted_df = extract_profiles_from_figure(source_image)
    sim_df = simulate_energy_balance(extracted_df)

    sim_df.to_csv("simulated_intra_day_energy_balance_v3.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.8, 6.6), dpi=180)

    ax.fill_between(sim_df["hour"], sim_df["solar_gw"],
                    alpha=0.65, color="#F4C542", label="Solar PV")
    ax.fill_between(sim_df["hour"], sim_df["wind_gw"],
                    alpha=0.60, color="#87CEFA", label="Wind")
    ax.plot(sim_df["hour"], sim_df["total_generation_gw"],
            linewidth=1.3, color="#555555", label="Total Generation")
    ax.plot(sim_df["hour"], sim_df["demand_gw"],
            linewidth=2.2, color="#8B0000", label="Demand")
    ax.bar(sim_df["hour"], sim_df["battery_discharge_gw"],
           width=0.18, alpha=0.75, color="#F4A6A6", edgecolor="#F4A6A6",
           label="Battery Discharge")
    ax.bar(sim_df["hour"], sim_df["battery_charge_gw"],
           width=0.18, alpha=0.80, color="#98D89E", edgecolor="#98D89E",
           label="Battery Charge")

    ax.axhline(0, linewidth=1.0)
    ax.set_xlim(0, 24)
    ax.set_ylim(-2.2, 6.2)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Power (GW)")
    ax.set_title("Simulated intra-day energy balance using Li-ion as short-term storage")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig("simulated_intra_day_energy_balance_v3.png", dpi=300, bbox_inches="tight")
    fig.savefig("simulated_intra_day_energy_balance_v3.pdf", bbox_inches="tight")
    fig.savefig("simulated_intra_day_energy_balance_v3.svg", bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    main()
