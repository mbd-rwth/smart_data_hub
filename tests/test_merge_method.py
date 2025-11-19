from src.merge_method import *
import numpy as np
import scipy.constants as sc


def test_merge_method():
    # test parameter settings
    # parameters for velocity
    value = 2500
    value_std = 100
    value_min = 1200
    value_max = 4000
    sample_size = 100000000
    tol = 10
    sampled_data = f"scipy.stats.norm(loc={value}, scale={value_std}).rvs(size=1000000, random_state=21)"
    # parameters for porosity
    value_rho = 0.35
    value_std_rho = 0.1
    value_min_rho = 0.05
    value_max_rho = 1.0
    tol_rho = 0.01
    # parameters for hydraulic conductivity
    value_hc = 1e-7
    value_std_hc = 1e-11
    value_min_hc = 1e-8
    value_max_hc = 1e-6
    tol_hc = 1e-10
    sampled_data_hc = f"scipy.stats.norm(loc={value_hc}, scale={value_std_hc}).rvs(size=1000000, random_state=21)"
    type_scalar = "scalar"
    type_expression = "expression"
    type_dict = "dictionary"
    expression_string = "6.1/(1+0.0045*(x-273.15))"
    dictionary_string = {280: 1e-11, 290: 1e-9}
    # parameters for intrinsic permeability
    value_ip = 1e-14

    # create test pandas DataFrame
    test_masks = [
        [
            value,
            value_min,
            value_max,
            value_std,
            sampled_data,
            "s_wave_velocity",
            "is_pdf_df",
            sample_size,
            type_scalar,
        ],
        [
            None,
            None,
            None,
            None,
            sampled_data,
            "s_wave_velocity",
            "is_pdf_df",
            sample_size,
            type_scalar,
        ],
        [
            value,
            value_min,
            value_max,
            value_std,
            None,
            "s_wave_velocity",
            "is_truncnorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value,
            value_min,
            value_max,
            None,
            None,
            "s_wave_velocity",
            "is_PERT_df",
            sample_size,
            type_scalar,
        ],
        [
            value,
            None,
            value_max,
            value_std,
            None,
            "s_wave_velocity",
            "is_lognorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value,
            value_min,
            None,
            value_std,
            None,
            "s_wave_velocity",
            "is_lognorm_df",
            sample_size,
            type_scalar,
        ],
        [
            None,
            value_min,
            value_max,
            value_std,
            None,
            "s_wave_velocity",
            "is_invalid_df",
            sample_size,
            type_scalar,
        ],
        [
            value,
            value_min,
            None,
            None,
            None,
            "s_wave_velocity",
            "is_lognorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value,
            None,
            value_max,
            None,
            None,
            "s_wave_velocity",
            "is_lognorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value,
            None,
            None,
            value_std,
            None,
            "s_wave_velocity",
            "is_lognorm_df",
            sample_size,
            type_scalar,
        ],
        [
            None,
            value_min,
            value_max,
            None,
            None,
            "s_wave_velocity",
            "is_uniform_df",
            sample_size,
            type_scalar,
        ],
        [
            None,
            value_min,
            value_max,
            None,
            None,
            "s_wave_velocity",
            "is_uniform_df",
            sample_size,
            type_scalar,
        ],
        [
            None,
            value_min,
            None,
            value_std,
            None,
            "s_wave_velocity",
            "is_invalid_df",
            sample_size,
            type_scalar,
        ],
        [
            None,
            None,
            value_max,
            value_std,
            None,
            "s_wave_velocity",
            "is_invalid_df",
            sample_size,
            type_scalar,
        ],
        [
            value,
            None,
            None,
            None,
            None,
            "s_wave_velocity",
            "is_lognorm_df",
            sample_size,
            type_scalar,
        ],
        [
            None,
            value_min,
            None,
            None,
            None,
            "s_wave_velocity",
            "is_invalid_df",
            sample_size,
            type_scalar,
        ],
        [
            None,
            None,
            value_max,
            None,
            None,
            "s_wave_velocity",
            "is_invalid_df",
            sample_size,
            type_scalar,
        ],
        [
            None,
            None,
            None,
            value_std,
            None,
            "s_wave_velocity",
            "is_invalid_df",
            sample_size,
            type_scalar,
        ],
        [
            None,
            None,
            None,
            None,
            None,
            "s_wave_velocity",
            "empty",
            sample_size,
            type_scalar,
        ],
        [
            value_rho,
            value_min_rho,
            None,
            value_std_rho,
            None,
            "porosity",
            "is_truncnorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value_rho,
            value_min_rho,
            None,
            None,
            None,
            "porosity",
            "is_PERT_df",
            sample_size,
            type_scalar,
        ],
        [
            value_rho,
            None,
            None,
            None,
            None,
            "porosity",
            "is_lognorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value_rho,
            value_min_rho,
            None,
            value_std_rho,
            None,
            "porosity",
            "is_truncnorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value_rho,
            value_min_rho,
            None,
            None,
            None,
            "porosity",
            "is_PERT_df",
            sample_size,
            type_scalar,
        ],
        [
            value_rho,
            None,
            None,
            None,
            None,
            "porosity",
            "is_lognorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value_rho,
            value_min_rho,
            None,
            value_std_rho,
            None,
            "porosity",
            "is_truncnorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value_rho,
            value_min_rho,
            None,
            None,
            None,
            "porosity",
            "is_PERT_df",
            sample_size,
            type_scalar,
        ],
        [
            value_rho,
            None,
            None,
            None,
            None,
            "porosity",
            "is_lognorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value_rho,
            value_min_rho,
            None,
            value_std_rho,
            None,
            "porosity",
            "is_truncnorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value_rho,
            value_min_rho,
            None,
            None,
            None,
            "porosity",
            "is_PERT_df",
            sample_size,
            type_scalar,
        ],
        [
            value_hc,
            value_min_hc,
            value_max_hc,
            value_std_hc,
            sampled_data_hc,
            "hydraulic_conductivity",
            "is_pdf_df",
            sample_size,
            type_scalar,
        ],
        [
            value_hc,
            value_min_hc,
            value_max_hc,
            value_std_hc,
            None,
            "hydraulic_conductivity",
            "is_truncnorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value_hc,
            None,
            None,
            value_std_hc,
            None,
            "hydraulic_conductivity",
            "is_lognorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value_hc,
            value_min_hc,
            value_max_hc,
            None,
            None,
            "hydraulic_conductivity",
            "is_PERT_df",
            sample_size,
            type_scalar,
        ],
        [
            None,
            value_min_hc,
            value_max_hc,
            None,
            None,
            "hydraulic_conductivity",
            "is_uniform_df",
            sample_size,
            type_scalar,
        ],
        [
            value_ip,
            None,
            None,
            None,
            None,
            "intrinsic_permeability",
            "is_lognorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value_hc,
            value_min_hc,
            value_max_hc,
            value_std_hc,
            sampled_data_hc,
            "hydraulic_conductivity",
            "is_pdf_df",
            sample_size,
            type_scalar,
        ],
        [
            value_hc,
            value_min_hc,
            value_max_hc,
            value_std_hc,
            None,
            "hydraulic_conductivity",
            "is_truncnorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value_hc,
            None,
            None,
            value_std_hc,
            None,
            "hydraulic_conductivity",
            "is_lognorm_df",
            sample_size,
            type_scalar,
        ],
        [
            value_hc,
            value_min_hc,
            value_max_hc,
            None,
            None,
            "hydraulic_conductivity",
            "is_PERT_df",
            sample_size,
            type_scalar,
        ],
        [
            None,
            value_min_hc,
            value_max_hc,
            None,
            None,
            "hydraulic_conductivity",
            "is_uniform_df",
            sample_size,
            type_scalar,
        ],
        [
            value_ip,
            None,
            None,
            None,
            None,
            "intrinsic_permeability",
            "is_lognorm_df",
            sample_size,
            type_scalar,
        ],
        [
            expression_string,
            None,
            None,
            None,
            None,
            "hydraulic_conductivity",
            "no_pdf",
            None,
            type_expression,
        ],
        [
            dictionary_string,
            None,
            None,
            None,
            None,
            "hydraulic_conductivity",
            "no_pdf",
            None,
            type_dict,
        ],
    ]

    # Create the DataFrame
    test_masks_df = pd.DataFrame(
        test_masks,
        columns=[
            "value",
            "value_min",
            "value_max",
            "value_std",
            "sampled_data",
            "property",
            "assumed_pdf",
            "sample_size",
            "type",
        ],
    )
    test_masks_df["ID"] = np.array(np.arange(0, len(test_masks_df)), dtype=str)
    test_masks_df = test_masks_df.replace({np.nan: None})

    test_masks_df[["value", "value_min", "value_max", "value_std"]] = test_masks_df[
        ["value", "value_min", "value_max", "value_std"]
    ].map(preserve_value_type)
    # Keep the sample_size as integer
    test_masks_df["sample_size"] = pd.to_numeric(
        test_masks_df["sample_size"], errors="coerce"
    ).astype("Int64")
    # Convert list to numpy array
    test_masks_df["sampled_data"] = test_masks_df["sampled_data"].map(
        lambda sampled_data: (
            np.array(sampled_data, dtype=np.float64)
            if isinstance(sampled_data, list)
            else sampled_data
        )
    )
    # Replace np.nan with None
    test_masks_df = test_masks_df.replace({np.nan: None})

    # --- Test masks --- #
    is_empty_mask = value_empty_mask(test_masks_df)
    assert set(test_masks_df[is_empty_mask]["assumed_pdf"]) == {
        "empty"
    }, f"Incorrect empty mask detected!"

    is_invalid_mask = value_invalid_mask(test_masks_df)
    assert set(
        test_masks_df[is_invalid_mask]["assumed_pdf"] == {"is_invalid_df"}
    ), f"Incorrect invalid mask detected!"

    is_sampled_data_mask = value_pdf_mask(test_masks_df)
    assert set(
        test_masks_df[is_sampled_data_mask]["assumed_pdf"] == {"is_pdf_df"}
    ), f"Incorrect is_pdf_df mask detected!"

    is_uniform_mask = value_uniform_mask(test_masks_df)
    assert set(
        test_masks_df[is_uniform_mask]["assumed_pdf"] == {"is_uniform_df"}
    ), f"Incorrect uniform mask detected!"

    is_truncnorm_mask = value_truncnorm_mask(test_masks_df)
    assert set(
        test_masks_df[is_truncnorm_mask]["assumed_pdf"] == {"is_truncnorm_df"}
    ), f"Incorrect truncnorm mask detected!"

    is_lognorm_mask = value_lognorm_mask(test_masks_df)
    assert set(
        test_masks_df[is_lognorm_mask]["assumed_pdf"] == {"is_lognorm_df"}
    ), f"Incorrect lognorm mask detected!"

    is_PERT_mask = value_PERT_mask(test_masks_df)
    assert set(
        test_masks_df[is_PERT_mask]["assumed_pdf"] == {"is_PERT_df"}
    ), f"Incorrect PERT mask detected!"

    # --- Test different distributions --- #
    # test truncated normal distribution
    trunc_samples = generate_truncnorm(value, value_std, value_min, value_max).rvs(
        size=sample_size
    )
    assert np.min(trunc_samples) >= value_min
    assert np.max(trunc_samples) <= value_max
    assert abs(np.std(trunc_samples) - value_std) < tol
    assert abs(np.mean(trunc_samples) - value) < tol
    # test PERT distribution
    PERT_samples = generate_PERT(value, value_min, value_max).rvs(size=sample_size)
    assert np.min(PERT_samples) >= value_min
    assert np.max(PERT_samples) <= value_max
    assert abs(np.mean(PERT_samples) - (value_min + 4 * value + value_max) / 6) < tol
    assert (
        abs(
            np.std(PERT_samples)
            - np.sqrt((value - value_min) * (value_max - value) / 7)
        )
        < tol
    )
    # test lognormal distribution
    lognorm_samples = generate_lognorm(value, value_std, value_min).rvs(
        size=sample_size
    )
    assert np.min(lognorm_samples) >= value_min
    assert abs(np.std(lognorm_samples) - value_std) < tol
    assert abs(np.mean(lognorm_samples) - value) < tol
    # test uniform distribution
    uniform_samples = generate_uniform(value_min, value_max).rvs(size=sample_size)
    assert np.min(uniform_samples) >= value_min
    assert np.max(uniform_samples) <= value_max
    assert abs(np.mean(uniform_samples) - (value_min + value_max) / 2) < tol
    assert (
        abs(np.std(uniform_samples) - np.sqrt((value_max - value_min) ** 2 / 12)) < tol
    )

    # --- Test function : generate_samples --- #
    # create property masks
    porosity_mask = test_masks_df["property"] == "porosity"
    velocity_mask = test_masks_df["property"] == "s_wave_velocity"
    hc_mask = test_masks_df["property"] == "hydraulic_conductivity"
    ip_mask = test_masks_df["property"] == "intrinsic_permeability"
    # test custom sampled data
    velocity_sampled_data_df = test_masks_df[is_sampled_data_mask & velocity_mask]
    velocity_sampled_data_df_samples = generate_samples(
        velocity_sampled_data_df,
        list(set(velocity_sampled_data_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert (
        np.std(velocity_sampled_data_df_samples) < value_std
    )  # samples were filtered within three standard deviation of the mean
    assert abs(np.mean(velocity_sampled_data_df_samples) - value) < tol

    hc_sampled_data_df = test_masks_df[is_sampled_data_mask & hc_mask]
    hc_sampled_data_df_samples = generate_samples(
        hc_sampled_data_df,
        list(set(hc_sampled_data_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert (
        np.std(hc_sampled_data_df_samples) < value_std_hc
    )  # samples were filtered within three standard deviation of the mean
    assert abs(np.mean(hc_sampled_data_df_samples) - value_hc) < tol_hc
    # test samples from uniform distribution
    velocity_uniform_df = test_masks_df[is_uniform_mask & velocity_mask]
    velocity_uniform_df_samples = generate_samples(
        velocity_uniform_df,
        list(set(velocity_uniform_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert np.min(velocity_uniform_df_samples) >= value_min
    assert np.max(velocity_uniform_df_samples) <= value_max
    assert abs(np.mean(velocity_uniform_df_samples) - (value_min + value_max) / 2) < tol
    assert (
        abs(
            np.std(velocity_uniform_df_samples)
            - np.sqrt((value_max - value_min) ** 2 / 12)
        )
        < tol
    )

    hc_uniform_df = test_masks_df[is_uniform_mask & hc_mask]
    hc_uniform_df_samples = generate_samples(
        hc_uniform_df,
        list(set(hc_uniform_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert np.min(hc_uniform_df_samples) >= value_min_hc
    assert np.max(hc_uniform_df_samples) <= value_max_hc
    assert (
        abs(np.mean(hc_uniform_df_samples) - (value_min_hc + value_max_hc) / 2) < tol_hc
    )
    assert (
        abs(
            np.std(hc_uniform_df_samples)
            - np.sqrt((value_max_hc - value_min_hc) ** 2 / 12)
        )
        < tol_hc
    )
    # test samples from truncated normal distribution
    velocity_truncnorm_df = test_masks_df[is_truncnorm_mask & velocity_mask]
    velocity_truncnorm_df_samples = generate_samples(
        velocity_truncnorm_df,
        list(set(velocity_truncnorm_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert np.min(velocity_truncnorm_df_samples) >= value_min
    assert np.max(velocity_truncnorm_df_samples) <= value_max
    assert (
        np.std(velocity_truncnorm_df_samples) < value_std
    )  # samples were filtered within three standard deviation of the mean
    assert abs(np.mean(velocity_truncnorm_df_samples) - value) < tol

    porosity_truncnorm_df = test_masks_df[is_truncnorm_mask & porosity_mask]
    porosity_truncnorm_df_samples = generate_samples(
        porosity_truncnorm_df,
        list(set(porosity_truncnorm_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert np.min(porosity_truncnorm_df_samples) >= value_min_rho
    assert np.max(porosity_truncnorm_df_samples) <= value_max_rho
    assert (
        np.std(porosity_truncnorm_df_samples) < value_std_rho
    )  # samples were filtered within three standard deviation of the mean
    assert abs(np.mean(porosity_truncnorm_df_samples) - value_rho) < tol_rho

    hc_truncnorm_df = test_masks_df[is_truncnorm_mask & hc_mask]
    hc_truncnorm_df_samples = generate_samples(
        hc_truncnorm_df,
        list(set(hc_truncnorm_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert np.min(hc_truncnorm_df_samples) >= value_min_hc
    assert np.max(hc_truncnorm_df_samples) <= value_max_hc
    assert (
        np.std(hc_truncnorm_df_samples) < value_std_hc
    )  # samples were filtered within three standard deviation of the mean
    assert abs(np.mean(hc_truncnorm_df_samples) - value_hc) < tol_hc

    # test samples from lognormal distribution
    velocity_lognorm_df = test_masks_df[is_lognorm_mask & velocity_mask]
    velocity_lognorm_df_samples = generate_samples(
        velocity_lognorm_df,
        list(set(velocity_lognorm_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert np.min(velocity_lognorm_df_samples) >= 0.0
    assert (
        np.std(velocity_lognorm_df_samples)
        < value
        * 0.13030  # multiple with the coefficient of variation (CV) for S-wave velocity
    )  # samples were filtered within three standard deviation of the mean
    assert abs(np.mean(velocity_lognorm_df_samples) - value) < tol

    porosity_lognorm_df = test_masks_df[is_lognorm_mask & porosity_mask]
    porosity_lognorm_df_samples = generate_samples(
        porosity_lognorm_df,
        list(set(porosity_lognorm_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert np.min(porosity_lognorm_df_samples) >= 0.0
    assert (
        np.std(porosity_lognorm_df_samples)
        < value_rho
        * 0.27030  # multiple with the coefficient of variation (CV) for S-wave velocity
    )  # samples were filtered within three standard deviation of the mean
    assert abs(np.mean(porosity_lognorm_df_samples) - value_rho) < tol_rho

    ip_lognorm_df = test_masks_df[is_lognorm_mask & ip_mask]
    ip_lognorm_df_samples = generate_samples(
        ip_lognorm_df,
        list(set(ip_lognorm_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert np.min(ip_lognorm_df_samples) >= 0.0
    assert (
        np.std(ip_lognorm_df_samples)
        < value_ip
        * 4.18875  # multiple with the coefficient of variation (CV) for S-wave velocity
    )  # samples were filtered within three standard deviation of the mean
    assert abs(np.mean(ip_lognorm_df_samples) - value_ip) < value_ip * 4.18875

    hc_lognorm_df = test_masks_df[is_lognorm_mask & hc_mask]
    hc_lognorm_df_samples = generate_samples(
        hc_lognorm_df,
        list(set(hc_lognorm_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert np.min(hc_lognorm_df_samples) >= 0.0
    assert (
        np.std(hc_lognorm_df_samples)
        < value_std_hc  # multiple with the coefficient of variation (CV) for S-wave velocity
    )  # samples were filtered within three standard deviation of the mean
    assert abs(np.mean(hc_lognorm_df_samples) - value_hc) < value_std_hc

    # test samples from PERT distribution
    velocity_PERT_df = test_masks_df[is_PERT_mask & velocity_mask]
    velocity_PERT_df_samples = generate_samples(
        velocity_PERT_df,
        list(set(velocity_PERT_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert np.min(velocity_PERT_df_samples) >= value_min
    assert np.max(velocity_PERT_df_samples) <= value_max
    assert (
        abs(np.mean(velocity_PERT_df_samples) - (value_min + 4 * value + value_max) / 6)
        < tol
    )
    assert (
        abs(
            np.std(velocity_PERT_df_samples)
            - np.sqrt((value - value_min) * (value_max - value) / 7)
        )
        < tol
    )

    porosity_PERT_df = test_masks_df[is_PERT_mask & porosity_mask]
    porosity_PERT_df_samples = generate_samples(
        porosity_PERT_df,
        list(set(porosity_PERT_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert np.min(porosity_PERT_df_samples) >= value_min_rho
    assert np.max(porosity_PERT_df_samples) <= value_max_rho
    assert (
        abs(
            np.mean(porosity_PERT_df_samples)
            - (value_min_rho + 4 * value_rho + value_max_rho) / 6
        )
        < tol_rho
    )
    assert (
        abs(
            np.std(porosity_PERT_df_samples)
            - np.sqrt((value_rho - value_min_rho) * (value_max_rho - value_rho) / 7)
        )
        < tol_rho
    )

    hc_PERT_df = test_masks_df[is_PERT_mask & hc_mask]
    hc_PERT_df_samples = generate_samples(
        hc_PERT_df,
        list(set(hc_PERT_df["assumed_pdf"]))[0],
        random_state=21,
    )

    assert np.min(hc_PERT_df_samples) >= value_min_hc
    assert np.max(hc_PERT_df_samples) <= value_max_hc
    assert abs(
        np.mean(hc_PERT_df_samples) - (value_min_hc + 4 * value_hc + value_max_hc) / 6
    ) < np.std(hc_PERT_df_samples)
    assert abs(
        np.std(hc_PERT_df_samples)
        - np.sqrt((value_hc - value_min_hc) * (value_max_hc - value_hc) / 7)
    ) < np.std(hc_PERT_df_samples)

    # --- Test function : get_sample_statistics --- #
    def samples_statistics(samples):
        return (
            format_number_adaptive(np.mean(samples)),
            format_number_adaptive(np.std(samples)),
            format_number_adaptive(np.min(samples)),
            format_number_adaptive(np.max(samples)),
        )

    # compare the statistics from get_sample_statistics and previously generated samples
    assert np.allclose(
        np.array(get_sample_statistics(velocity_sampled_data_df)),
        np.array(samples_statistics(velocity_sampled_data_df_samples)),
    )
    assert np.allclose(
        np.array(get_sample_statistics(velocity_uniform_df)),
        np.array(samples_statistics(velocity_uniform_df_samples)),
    )
    assert np.allclose(
        np.array(get_sample_statistics(velocity_truncnorm_df)),
        np.array(samples_statistics(velocity_truncnorm_df_samples)),
    )
    assert np.allclose(
        np.array(get_sample_statistics(velocity_lognorm_df)),
        np.array(samples_statistics(velocity_lognorm_df_samples)),
    )
    assert np.allclose(
        np.array(get_sample_statistics(velocity_PERT_df)),
        np.array(samples_statistics(velocity_PERT_df_samples)),
    )
    assert np.allclose(
        np.array(get_sample_statistics(porosity_truncnorm_df)),
        np.array(samples_statistics(porosity_truncnorm_df_samples)),
    )
    assert np.allclose(
        np.array(get_sample_statistics(porosity_PERT_df)),
        np.array(samples_statistics(porosity_PERT_df_samples)),
    )
    assert np.allclose(
        np.array(get_sample_statistics(porosity_lognorm_df)),
        np.array(samples_statistics(porosity_lognorm_df_samples)),
    )

    assert np.allclose(
        np.array(get_sample_statistics(hc_sampled_data_df)),
        np.array(samples_statistics(hc_sampled_data_df_samples)),
    )
    assert np.allclose(
        np.array(get_sample_statistics(hc_uniform_df)),
        np.array(samples_statistics(hc_uniform_df_samples)),
    )
    assert np.allclose(
        np.array(get_sample_statistics(hc_truncnorm_df)),
        np.array(samples_statistics(hc_truncnorm_df_samples)),
    )
    assert np.allclose(
        np.array(get_sample_statistics(hc_lognorm_df)),
        np.array(samples_statistics(hc_lognorm_df_samples)),
    )
    assert np.allclose(
        np.array(get_sample_statistics(hc_PERT_df)),
        np.array(samples_statistics(hc_PERT_df_samples)),
    )

    assert np.allclose(
        np.array(get_sample_statistics(ip_lognorm_df)),
        np.array(samples_statistics(ip_lognorm_df_samples)),
    )

    # --- Test function : merge_property_value --- #
    # compare the statistics from merge_property_value and previously generated samples
    # test the custom sampled data
    merged_velocity_sampled_data_df = merge_property_value(velocity_sampled_data_df)
    assert np.allclose(
        merged_velocity_sampled_data_df[
            ["value", "value_std", "value_min", "value_max"]
        ].values[0],
        np.array(samples_statistics(velocity_sampled_data_df_samples)),
    )
    # test conversion from hydraulic conductivity to intrinsic permeability
    water_vis = 0.001
    water_density = 1000
    gravitational_acceleration = sc.g
    convert_ratio = water_vis / (water_density * gravitational_acceleration)
    merged_hc_sampled_data_df = merge_property_value(hc_sampled_data_df)
    assert np.allclose(
        merged_hc_sampled_data_df[
            ["value", "value_std", "value_min", "value_max"]
        ].values[0],
        samples_statistics(hc_sampled_data_df_samples * convert_ratio),
    )

    merged_hc_truncnorm_df = merge_property_value(hc_truncnorm_df)
    assert np.allclose(
        merged_hc_truncnorm_df[["value", "value_std", "value_min", "value_max"]].values[
            0
        ],
        samples_statistics(hc_truncnorm_df_samples * convert_ratio),
    )

    # test merging with non-scalar types
    hc_nonscalar_df = test_masks_df.loc[(test_masks_df["type"] != "scalar") & (hc_mask)]
    hc_truncnorm_nonscalar_df = pd.concat(
        [hc_truncnorm_df, hc_nonscalar_df], ignore_index=True
    )

    additional_cols = pd.DataFrame(
        {
            "unit_base": [None, None, None, None],
            "variable_unit_base": [None, None, None, None],
            "agency": [None, None, None, None],
            "location": [None, None, None, None],
            "source": [None, None, None, None],
            "variable_name": [None, None, None, None],
            "variable_unit_str": [None, None, None, None],
            "description": [None, None, None, None],
            "simplified_lithology": [None, None, None, None],
        }
    )
    hc_truncnorm_nonscalar_df = pd.concat(
        [hc_truncnorm_nonscalar_df, additional_cols], axis=1
    )

    merged_hc_truncnorm_nonscalar_df = merge_property_value(hc_truncnorm_nonscalar_df)
    assert merged_hc_truncnorm_nonscalar_df.iloc[0].equals(
        merged_hc_truncnorm_df.iloc[0]
    )

    ip_nonscalar_df = hydraulic2intrinic(hc_nonscalar_df).copy()
    assert (
        ip_nonscalar_df["value"].values.tolist()
        == merged_hc_truncnorm_nonscalar_df.iloc[1:3]["value"].values.tolist()
    )

    print("All the tests for merging functions passed successfully!")


if __name__ == "__main__":
    test_merge_method()
