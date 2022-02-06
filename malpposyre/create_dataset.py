import pandas as pd

# Import all needed dataframes
df_flow = pd.read_csv("data/flow_280921.csv")
df_costs = pd.read_csv("data/costs_280921.csv")
df_opf = pd.read_csv("data/opf_280921.csv", header=[0, 1])


corr_cost_list = list(df_costs.columns)[4:]

failure_rate = 0.1
sps_success = 1  # success rate for SPS

# df_costs["Total cost"] = df_costs.apply(lambda row: row["Dispatch costs"] + (failure_rate*row[corr_cost_list].sum()), axis=1)

df_costs.rename(
    columns={"Unnamed: 0": "prev_id", "Unnamed: 1": "fault", "Unnamed: 2": "corr"},
    inplace=True,
)


def add_sps_succes(row):
    corr_cost_list = ["CENS", "FCR cost", "Redispatch cost", "SPS cost"]
    df = df_costs.copy()
    if row["corr"] != "fault":  # if scenario with SPS
        sps_fault = df[
            (df["prev_id"] == row["prev_id"])
            & (df["fault"] == row["fault"])
            & (df["corr"] == "fault")
        ].iloc[0]
        total_cost = row["Dispatch costs"] + failure_rate * (
            sps_success * (row[corr_cost_list].sum())
            + ((1 - sps_success) * (sps_fault[corr_cost_list].sum()))
        )
    else:
        total_cost = row["Dispatch costs"] + (failure_rate * row[corr_cost_list].sum())

    return total_cost


df_costs["Total cost"] = df_costs.apply(add_sps_succes, axis=1)

# Join multi-index column names in dataframe
df_opf.columns = ["_".join(col) for col in df_opf.columns.values]


# Dictionary assigns each generator to area
gen_dict = {
    "Area1": ["G0" + str(i) + "_Pdisp" for i in range(10)]
    + ["G" + str(i) + "_Pdisp" for i in range(10, 14)],
    "Area2": ["G" + str(i) + "_Pdisp" for i in range(14, 26)],
    "Area3": ["G" + str(i) + "_Pdisp" for i in range(26, 34)],
}

# Dictionary assigns each load to area
load_dict = {
    "Area1": ["L02_P"],
    "Area2": ["L03_P", "L04_P", "L05_P"],
    "Area3": ["L06_P", "L07_P", "L08_P", "L09_P"],
}


# Sum of all generation each area
df_opf["Area1_prod"] = df_opf[gen_dict["Area1"]].sum(axis=1)
df_opf["Area2_prod"] = df_opf[gen_dict["Area2"]].sum(axis=1)
df_opf["Area3_prod"] = df_opf[gen_dict["Area3"]].sum(axis=1)

# Sum of all load each area
df_opf["Area1_load"] = df_opf[load_dict["Area1"]].sum(axis=1)
df_opf["Area2_load"] = df_opf[load_dict["Area2"]].sum(axis=1)
df_opf["Area3_load"] = df_opf[load_dict["Area3"]].sum(axis=1)


df_opf2 = df_opf[
    [
        "unit_variable",
        "Line10_P",
        "Line11_P",
        "Line18_P",
        "Line19_P",
        "Area1_prod",
        "Area2_prod",
        "Area3_prod",
        "Area1_load",
        "Area2_load",
        "Area3_load",
    ]
]

# Set prev_id as index in all dataframes before merging
df_opf2.set_index("unit_variable", inplace=True)
df_flow.set_index("Unnamed: 0", inplace=True)
df_costs.set_index("prev_id", inplace=True)


df_merge = df_opf2.join(df_flow)
df_tot = df_merge.join(df_costs)

df_tot.reset_index(inplace=True)
df_tot.rename(
    columns={
        "index": "prev_id",
        "Unnamed: 1": "fault",
        "Unnamed: 2": "corr",
        "Line10_P": "Line10",
        "Line11_P": "Line11",
        "Line18_P": "Line18",
        "Line19_P": "Line19",
        "Inter-area power flow (MW)": "inter_area_flow",
    },
    inplace=True,
)

merge2 = df_tot.copy()


# One hot encoding to represent fault and corrective measures in nodes
merge2["Area1-corr"] = merge2["corr"].apply(lambda x: 1 if x == "corr_G12" else 0)
merge2["Area2-corr"] = merge2["corr"].apply(lambda x: 1 if x == "corr_G18" else 0)
merge2["Area3-corr"] = merge2["corr"].apply(lambda x: 0.5 if x == "corr_L09" else 0)

merge2["Area1-fault"] = merge2["fault"].apply(
    lambda x: 1 if x == "fault_L10" or x == "fault_L11" else 0
)
merge2["Area2-fault"] = merge2["fault"].apply(
    lambda x: 1 if x == "fault_L10" or x == "fault_L18" else 0
)
merge2["Area3-fault"] = merge2["fault"].apply(
    lambda x: 1 if x == "fault_L11" or x == "fault_L18" else 0
)

merge2["Area1-fault2"] = merge2["fault"].apply(lambda x: 0)
merge2["Area2-fault2"] = merge2["fault"].apply(lambda x: 1 if x == "fault_L19" else 0)
merge2["Area3-fault2"] = merge2["fault"].apply(lambda x: 1 if x == "fault_L19" else 0)

# Save as .csv-file
# merge2.to_csv('data/tot_cost_failure_scenarios_180122_01.csv', index = False)
