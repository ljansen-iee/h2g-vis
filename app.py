import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from scripts.plot_helpers import (
    load_plot_config,
    read_stats_dict,
    read_csv_nafix,
    prepare_dataframe,
    rename_electricity,
    rename_h2,
    rename_gas,
    rename_oil,
    rename_co2,
    rename_costs,
    colors,
    nice_title,
    get_scen_col_function,
    apply_standard_styling,
    plot_energy_balance,
    plot_capacity,
    plot_gwkm,
    register_template,
)

register_template()

st.set_page_config(page_title="H2Global meets Africa", layout="wide")

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def _load_yaml_config():
    return load_plot_config()


@st.cache_data
def _load_energy_data(sdir_str, balance_keys, capacity_keys, cost_keys):
    sdir = Path(sdir_str)
    balance_dict = read_stats_dict("balance_dict", sdir, keys=balance_keys)
    capacity_dict = read_stats_dict("optimal_capacity_dict", sdir, keys=capacity_keys)
    costs_dict = read_stats_dict("costs_dict", sdir, keys=cost_keys)
    return balance_dict, capacity_dict, costs_dict


@st.cache_data
def _load_marginal_prices(sdir_str):
    return read_csv_nafix(Path(sdir_str) / "marginal_prices_prepared.csv")


@st.cache_data
def _load_gwkm(sdir_str, filename):
    return read_csv_nafix(Path(sdir_str) / filename)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

yaml_config = _load_yaml_config()

RUN_NAME_PREFIX = yaml_config["run"]["name_prefix"]
SUMMARY_VERSION = yaml_config["run"]["summary_version"]
SDIR = Path(__file__).parent / "results" / f"{RUN_NAME_PREFIX}_summary_{SUMMARY_VERSION}"

ALL_COUNTRIES = yaml_config["data"]["countries"]
DEFAULT_COUNTRIES = ALL_COUNTRIES[5:7]
ALL_YEARS = yaml_config["data"]["years"]
INDEX_LEVELS_TO_DROP = yaml_config["data"]["index_levels_to_drop"]
SCEN_FILTER = yaml_config["data"]["scen_filter"]
SCEN_ORDER = yaml_config["data"]["scen_order"]

set_scen_col = get_scen_col_function(yaml_config["data"]["scen_col_function"])

INDEX_COLS = [
    "run_name_prefix", "run_name", "country", "year",
    "simpl", "clusters", "ll", "opts", "sopts",
    "discountrate", "demand", "h2export",
]

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "Marginal prices": {"type": "marginal"},
    "Electricity supply": {
        "type": "balance", "carrier": "AC", "chart_key": "electricity_supply",
        "rename": rename_electricity, "show_supply": True,
    },
    "Electricity demand": {
        "type": "balance", "carrier": "AC", "chart_key": "electricity_demand",
        "rename": rename_electricity, "show_supply": False,
    },
    "Electricity capacity": {
        "type": "capacity", "carrier": "AC", "chart_key": "electricity_capacity",
        "rename": rename_electricity,
    },
    "Hydrogen balance": {
        "type": "balance", "carrier": "H2", "chart_key": "hydrogen_balance",
        "rename": rename_h2,
    },
    "Hydrogen capacity": {
        "type": "capacity", "carrier": "H2", "chart_key": "hydrogen_capacity",
        "rename": rename_h2,
    },
    "Liquid fuel balance": {
        "type": "balance", "carrier": "oil", "chart_key": "liquid_fuel_balance",
        "rename": rename_oil,
    },
    "CH4 balance": {
        "type": "balance", "carrier": "gas", "chart_key": "ch4_balance",
        "rename": rename_gas,
    },
    "CO2 capture and usage": {
        "type": "balance", "carrier": "co2 stored", "chart_key": "co2_stored_balance",
        "rename": rename_co2,
    },
    "CO2 emissions": {
        "type": "balance", "carrier": "co2", "chart_key": "co2_emissions",
        "rename": None,
    },
    "System costs": {"type": "costs", "chart_key": "system_costs"},
    "Grid km (AC)": {"type": "gwkm", "filename": "gwkm_dict_AC.csv"},
    "Grid km (H2 pipeline)": {"type": "gwkm", "filename": "gwkm_dict_H2 pipeline.csv"},
}

# ---------------------------------------------------------------------------
# Marginal-price dumbbell chart (one per year)
# ---------------------------------------------------------------------------

def _marginal_price_dumbbell(df, year, data_start, data_mid, data_end):

    df = df[(df.year == year)].copy()
    countries = df["country"].unique()

    line_x, line_y = [], []
    vals_start, vals_mid, vals_end, valid = [], [], [], []
    skipped = []

    for country in countries:
        s = df.loc[(df.scen == f"{country}-{data_start}") & (df.country == country), "value"]
        m = df.loc[(df.scen == f"{country}-{data_mid}") & (df.country == country), "value"]
        e = df.loc[(df.scen == f"{country}-{data_end}") & (df.country == country), "value"]
        if len(s) > 0 and len(e) > 0:
            vs, ve = s.values[0], e.values[0]
            vals_start.append(vs)
            vals_end.append(ve)
            vals_mid.append(m.values[0] if len(m) > 0 else None)
            line_x.extend([vs, ve, None])
            line_y.extend([country, country, None])
            valid.append(country)
        else:
            missing = []
            if len(s) == 0:
                missing.append(data_start)
            if len(e) == 0:
                missing.append(data_end)
            skipped.append(f"{country} (missing: {', '.join(missing)})")

    fig = go.Figure(data=[
        go.Scatter(x=line_x, y=line_y, mode="lines", showlegend=False,
                   marker=dict(color="grey")),
        go.Scatter(
            x=vals_start, y=valid, mode="markers+text", name=data_start,
            text=[f"{v:.1f}" for v in vals_start], textposition="middle left",
            textfont=dict(size=11), marker=dict(color="#99bdcc", size=13),
        ),
        go.Scatter(
            x=[v for v in vals_mid if v is not None],
            y=[c for c, v in zip(valid, vals_mid) if v is not None],
            mode="markers", name=data_mid,
            marker=dict(color="#669db2 ", size=13),
        ),
        go.Scatter(
            x=vals_end, y=valid, mode="markers+text", name=data_end,
            text=[f"{v:.1f}" for v in vals_end], textposition="middle right",
            textfont=dict(size=11), marker=dict(color="#005b7f", size=13), #669db2 
        ),
    ])

    fig.update_layout(
        title=dict(text=nice_title(
            f"Marginal price for H2 at export port in {year}",
            "Per country and H2 export volume in EUR/MWh_H2_LHV",
        )),
        height=max(500, len(valid) * 45),
        legend_itemclick=False,
    )
    return fig, skipped


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("H2Global meets Africa Project")
    st.subheader("Comparative assessment of hydrogen production costs using PyPSA-Earth")
    st.selectbox("Select experiment", ["H2 Export from 13 African countries"])
    st.markdown("Display options")
    use_rename = st.checkbox("Rename variables", value=True)

# ---------------------------------------------------------------------------
# Main area — controls
# ---------------------------------------------------------------------------

col_dataset, col_year = st.columns([2, 1])
with col_dataset:
    dataset = st.selectbox("Select dataset", list(DATASETS.keys()))
with col_year:
    if dataset == "Marginal prices":
        selected_year = st.selectbox("Select year", ALL_YEARS)

default_all = dataset == "Marginal prices"
col_check, col_select = st.columns([1, 3])
with col_check:
    select_all = st.checkbox("Select all countries", value=default_all)
with col_select:
    selected_countries = st.multiselect(
        "Select countries", options=ALL_COUNTRIES,
        default=ALL_COUNTRIES if select_all else DEFAULT_COUNTRIES,
        disabled=select_all,
    )
    if select_all:
        selected_countries = ALL_COUNTRIES

if not selected_countries:
    st.warning("Select at least one country.")
    st.stop()

tab_chart, tab_table = st.tabs(["Chart", "Table"])

# ---------------------------------------------------------------------------
# Build plot config for energy-system charts
# ---------------------------------------------------------------------------

idx_slice = pd.IndexSlice
IDX_GROUP = idx_slice[[RUN_NAME_PREFIX], :, selected_countries, ALL_YEARS]

width = yaml_config["plot"]["width"]
heights = yaml_config["plot"]["heights"]
fig_kwargs = yaml_config["plot"]["default_kwargs"].copy()
fig_kwargs["width"] = width
fig_kwargs["height"] = heights["medium"]

plot_config = {
    "idx_group": IDX_GROUP,
    "idx_group_name": "".join(selected_countries) + "_MAIN",
    "index_levels_to_drop": INDEX_LEVELS_TO_DROP,
    "set_scen_col_func": set_scen_col,
    "scen_filter": SCEN_FILTER,
    "scen_order": SCEN_ORDER,
    "fig_kwargs": fig_kwargs,
    "heights": heights,
}

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

spec = DATASETS[dataset]

if spec["type"] == "marginal":
    df = _load_marginal_prices(str(SDIR))
    df = df[df["country"].isin(selected_countries)].copy()
    df = df.query("variable == 'H2 export bus'").copy()
    if SCEN_FILTER:
        df = df.query("scen in @SCEN_FILTER")

    ds, dm, de = "low", "mid", "high"

    with tab_chart:
        fig, skipped = _marginal_price_dumbbell(df, selected_year, ds, dm, de)
        if skipped:
            st.caption(f"Skipped: {', '.join(skipped)}")
        st.plotly_chart(fig, use_container_width=True)

    with tab_table:
        st.dataframe(
            df[["country", "year", "scen", "value"]]
            .sort_values(["year", "country", "scen"]),
            use_container_width=True,
        )

elif spec["type"] in ("balance", "capacity"):
    balance_dict, capacity_dict, costs_dict = _load_energy_data(
        str(SDIR),
        yaml_config["carriers"]["balance"],
        yaml_config["carriers"]["capacity"],
        yaml_config["carriers"]["costs"],
    )
    plot_config["chart_config"] = yaml_config["charts"][spec["chart_key"]]

    try:
        rename_func = spec.get("rename") if use_rename else None
        if spec["type"] == "balance":
            fig, df = plot_energy_balance(
                carrier=spec["carrier"],
                balance_dict=balance_dict,
                config=plot_config,
                rename_function=rename_func,
                show_supply=spec.get("show_supply"),
            )
        else:
            fig, df = plot_capacity(
                carrier=spec["carrier"],
                capacity_dict=capacity_dict,
                config=plot_config,
                rename_function=rename_func,
            )

        with tab_chart:
            st.plotly_chart(fig, use_container_width=True)
        with tab_table:
            st.dataframe(df, use_container_width=True)

    except (ValueError, KeyError) as e:
        st.error(f"No data available for this selection: {e}")

elif spec["type"] == "costs":
    balance_dict, capacity_dict, costs_dict = _load_energy_data(
        str(SDIR),
        yaml_config["carriers"]["balance"],
        yaml_config["carriers"]["capacity"],
        yaml_config["carriers"]["costs"],
    )
    chart_config = yaml_config["charts"].get("system_costs", {
        "title": "System Costs",
        "save_name": "system_costs_billion_eur",
        "chart_type": "bar_with_negatives",
    })
    plot_config["chart_config"] = chart_config

    try:
        stats_df = pd.concat([costs_dict["capex"], costs_dict["opex"]], axis=0)
        df = prepare_dataframe(stats_df, IDX_GROUP, INDEX_LEVELS_TO_DROP, set_scen_col)
        if use_rename:
            df.variable = df.variable.map(rename_costs)
        df = df[df["value"] != 0]
        if SCEN_FILTER:
            df = df.query("scen in @SCEN_FILTER")

        fig_kw = fig_kwargs.copy()
        fig_kw["height"] = heights["large"]

        fig = px.bar(
            df, **fig_kw,
            color_discrete_map=colors["costs"],
            category_orders={
                "variable": list(colors["costs"].keys()),
                "scen": SCEN_ORDER,
            },
            labels={"value": "System costs [billion EUR/year]", "year": "", "scen": ""},
        )
        fig = apply_standard_styling(fig, chart_config.get("chart_type", "bar"),
                                     chart_config)

        with tab_chart:
            st.plotly_chart(fig, use_container_width=True)
        with tab_table:
            st.dataframe(df, use_container_width=True)

    except (ValueError, KeyError) as e:
        st.error(f"No data available for this selection: {e}")

elif spec["type"] == "gwkm":
    gwkm_df = _load_gwkm(str(SDIR), spec["filename"])
    gwkm_df = gwkm_df[gwkm_df["country"].isin(selected_countries)].copy()

    try:
        fig, gwkm_plot_df = plot_gwkm(
            gwkm_df,
            config=plot_config,
        )
        with tab_chart:
            st.plotly_chart(fig, use_container_width=True)
        with tab_table:
            st.dataframe(
                gwkm_plot_df[["country", "year", "scen", "variable", "value"]]
                .sort_values(["year", "country"]),
                use_container_width=True,
            )
    except (ValueError, KeyError) as e:
        st.error(f"No data available for this selection: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("Licensed under CC BY 4.0")
    st.caption(
        "Achhammer, A., Jansen, L., Schumm, L., Meisinger, A., "
        "Duque Pérez, E., Schamel, M., Pilsl, A., Nguyen, H. H., & Sterner, M., "
        "2026, "
        "Comparative assessment of hydrogen production costs in thirteen African "
        "countries using PyPSA-Earth."
    )

with col2:
    st.caption(
        "OTH Regensburg (FENES) & Fraunhofer IEE, 2026"
    )
    st.caption(
        "Repository: [H2G-A-Vis-Repo](https://github.com/doneachh/h2g-a)"
    )

with col3:
    st.caption("[Imprint ??](https://www.iee.fraunhofer.de/en/publishing-notes.html)")
    st.caption("[Data Protection ??](https://www.iee.fraunhofer.de/en/data_protection.html)")
