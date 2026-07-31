from importlib.resources import files
from smart_data_hub.add_default import add_default_df
from smart_data_hub.add_default import load_rock_property


def test_add_default():

    property_df = load_rock_property(
        [files("smart_data_hub") / "dataset" / "rock_property" / "DE_South_Claystone" / "Muschelkalk_Middle.yaml"],
    )
    lithologies = ["Rocksalt", "Mudstone"]
    add_default_property_df = add_default_df(property_df, lithologies)

    # check if there are still missing properties
    missing = add_default_property_df.loc[add_default_property_df["ID"].isna()]
    assert missing.empty, f"add_default_df did not fill all missing IDs. Rows with missing ID:\n{missing}"
