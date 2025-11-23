from src.add_default import *


def test_add_default():

    property_df = load_rock_property(
        [f"database/rock_property/DE_South_Claystone/Muschelkalk_Middle.yaml"]
    )
    lithologies = ["Rocksalt", "Mudstone"]
    add_default_property_df = add_default_df(property_df, lithologies)

    # check if there are still missing properties
    assert (add_default_property_df.loc[add_default_property_df["ID"].isna()]).empty
    print("All missing properties have been filled!")
