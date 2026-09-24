
#Geoclim steady state — weathering module (573 parameter combinations)

#Code sources: Yu Liu (2667781848@qq.com) 


import io
import os

import numpy as np
import pandas as pd
from netCDF4 import Dataset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIRECTORY = SCRIPT_DIR
OUTPUT_DIRECTORY = SCRIPT_DIR

FILENAME_AREA = "land_area.nc"
FILENAME_CELL_AREA = "cell_area.nc"  
FILENAME_RUNOFF = "runoff.nc"
FILENAME_TEMP = "temp.nc"
FILENAME_SLOPE = "slope.nc"


MAX_ALLOWED_INACC = 1.0e-6
DEFFILLVAL = 1.0e36
FILL_ABS_THRESHOLD = 1.0e30 



KE = 0.0030713

DEFAULT_EROSION_PARAMS = {
    "ke": KE,
    "a": 0.5,
    "b": 1.0,
}
DEFAULT_MODEL_PARAMS = {
    "kd": 0.0005,
    "kw": 1,
    "krp": 0.01,
    "sigma": -0.4,
    "Ea": 42000,
    "R_gas": 8.314472,
    "T0": 286.0,
    "h0": 2.73,
}
# Default lithology CaMg: water, metamorphics, felsic, intermediates, mafic, carbonates, sediments
DEFAULT_LITHOLOGY_PARAMS = np.array([0, 2500, 1521, 4759, 10317, 0, 2000])

# Fixed global lithology area fractions (no lith.nc).
#   0 water/ice, 1 变质岩, 2 长英质岩, 3 中性岩, 4 镁铁质岩, 5 碳酸盐, 6 硅质碎屑沉积物
FIXED_LITHOLOGY_FRACTIONS = np.array(
    [
        0.0,    # water/ice
        0.144,  # metamorphics 变质岩 14.4%
        0.074,  # felsic 长英质岩 7.4%
        0.023,  # intermediates 中性岩 2.3%
        0.053,  # mafic 镁铁质岩 5.3%
        0.090,  # carbonates 碳酸盐 9%
        0.616,  # sediments 硅质碎屑岩沉积物 61.6%
    ],
    dtype=float,
)
LITHOLOGY_CLASS_NAMES = [
    "water/ice",
    "变质岩",
    "长英质岩",
    "中性岩",
    "镁铁质岩",
    "碳酸盐",
    "硅质碎屑沉积物",
]


def load_params_from_csv(csv_source=None):
    """Load parameter sets from embedded CSV text (or optional external file)."""
    if csv_source and os.path.exists(csv_source):
        df = pd.read_csv(csv_source)
        print(f"从CSV文件读取参数: {csv_source}")
    else:
        df = pd.read_csv(io.StringIO(EMBEDDED_PARAMS_CSV))
        print("从脚本内嵌参数读取 ")

    print(f"共找到 {len(df)} 套参数组合")
    print(f"全局侵蚀系数 KE = {KE}")
    print(f"参数列名: {list(df.columns)}")

    # ke is a global editable parameter (KE); do not read it from the table
    erosion_param_names = ["a", "b"]
    model_param_names = ["kd", "kw", "krp", "sigma", "Ea", "R_gas", "T0", "h0"]

    params_list = []
    for idx, row in df.iterrows():
        erosion_params = DEFAULT_EROSION_PARAMS.copy()
        erosion_params["ke"] = KE  # always use the top-level KE
        model_params = DEFAULT_MODEL_PARAMS.copy()
        lithology_params = DEFAULT_LITHOLOGY_PARAMS.copy()

        for param_name in erosion_param_names:
            if param_name in df.columns:
                erosion_params[param_name] = float(row[param_name])

        for param_name in model_param_names:
            if param_name in df.columns:
                model_params[param_name] = float(row[param_name])

        for i in range(1, 7):
            col_name = f"CaMg_{i}"
            if col_name in df.columns:
                lithology_params[i] = float(row[col_name])

        if not isinstance(lithology_params, np.ndarray):
            lithology_params = np.array(lithology_params)

        params_list.append(
            {
                "erosion_params": erosion_params,
                "model_params": model_params,
                "lithology_params": lithology_params,
                "row_index": idx,
            }
        )

    return params_list


def _read_fill_value(nc_var, default=DEFFILLVAL):
    for attr in ("_FillValue", "missing_value", "fill_value"):
        if attr in nc_var.ncattrs():
            try:
                return float(nc_var.getncattr(attr))
            except (TypeError, ValueError):
                pass
    return float(default)


def is_missing_value(arr, fill_value=None):
    """True where values are non-finite or equal/near the NetCDF fill sentinel."""
    a = np.asarray(arr, dtype=float)
    mask = ~np.isfinite(a) | (np.abs(a) >= FILL_ABS_THRESHOLD)
    if fill_value is not None and np.isfinite(fill_value):
        mask |= np.isclose(a, fill_value, rtol=0.0, atol=0.0) | (a == fill_value)
    return mask


def sanitize_field(arr, fill_value=None, replace_with=np.nan):
    """Replace missing/fill values with replace_with."""
    a = np.array(arr, dtype=float, copy=True)
    a[is_missing_value(a, fill_value)] = replace_with
    return a


def check_continental_missing_values(
    land_area,
    named_fields,
    remove_erratic=True,
):

    land_area_corr = np.array(land_area, dtype=float, copy=True)
    land_mask = land_area_corr > 0
    missing_any = np.zeros_like(land_area_corr, dtype=bool)
    reports = []

    tot_land_area = float(np.nansum(land_area_corr))
    nland = int(np.count_nonzero(land_mask))

    print("=" * 70)
    print("陆地格点缺测检查 (Fortran check_continental_cells)")
    print("=" * 70)

    for name, field in named_fields.items():
        miss = land_mask & is_missing_value(field)
        nerr = int(np.count_nonzero(miss))
        area_err = float(np.nansum(land_area_corr[miss])) if nerr > 0 else 0.0
        reports.append((name, nerr, area_err))
        missing_any |= miss
        if nerr > 0:
            print(f"  WARNING: found missing values on continental cells of variable {name}")
            print(f"    异常格点数:     {nerr}")
            print(f"    占陆地格点比例: {nerr / max(nland, 1):.6e}")
            print(f"    异常面积 (m2):  {area_err:.6e}")
            print(
                f"    占陆地总面积:   "
                f"{area_err / tot_land_area if tot_land_area > 0 else 0:.6e}"
            )

    n_total = int(np.count_nonzero(missing_any))
    if n_total > 0 and remove_erratic:
        land_area_corr[missing_any] = 0.0
        print(f"  已清除缺测陆地格点: {n_total}（land_area = 0）")
    elif n_total == 0:
        print("  通过检查: 陆地格点无温度/径流/坡度缺测")
    print()

    return land_area_corr, n_total


def load_climate_data(directory, filename_runoff, filename_temp):
    f_runoff = Dataset(os.path.join(directory, filename_runoff))
    rnf_fill = _read_fill_value(f_runoff["rnf"])
    R = sanitize_field(f_runoff["rnf"][:], fill_value=rnf_fill, replace_with=np.nan)
    f_runoff.close()
    R = np.where(np.isnan(R), np.nan, np.maximum(R, 0.0))

    f_temp = Dataset(os.path.join(directory, filename_temp))
    tmp_fill = _read_fill_value(f_temp["tmp"])
    T_celsius = np.array(f_temp["tmp"][:], dtype=float)
    tmp_missing = is_missing_value(T_celsius, tmp_fill)
    T = T_celsius + 273.15
    T[tmp_missing] = np.nan
    f_temp.close()

    print(f"径流数据形状：{R.shape}")
    print(f"温度数据形状：{T.shape}")
    print(
        f"径流范围(有效值): {np.nanmin(R):.4f} - {np.nanmax(R):.4f} m/yr "
        f"| 缺测格点: {int(np.count_nonzero(np.isnan(R)))}"
    )
    print(
        f"温度范围(有效值): {(np.nanmin(T) - 273.15):.2f} - {(np.nanmax(T) - 273.15):.2f} C "
        f"| 缺测格点: {int(np.count_nonzero(np.isnan(T)))}"
    )

    return R, T


def load_topography_data(directory, filename_area, filename_slope):
    f_area = Dataset(os.path.join(directory, filename_area))
    area = np.array(f_area["area"][:], dtype=float)
    f_area.close()

    f_slope = Dataset(os.path.join(directory, filename_slope))
    slope_fill = _read_fill_value(f_slope["slope"])
    slope = sanitize_field(f_slope["slope"][:], fill_value=slope_fill, replace_with=np.nan)
    f_slope.close()

    print(f"陆地面积形状：{area.shape}")
    print(
        f"坡度形状：{slope.shape} | 有效范围: {np.nanmin(slope):.6e} - {np.nanmax(slope):.6e} "
        f"| 缺测格点: {int(np.count_nonzero(np.isnan(slope)))}"
    )

    return area, slope


def load_cell_area(directory, filename_cell_area):
    """Load full geometric grid-cell area (m^2), corresponding to Fortran cell_area."""
    path = os.path.join(directory, filename_cell_area)
    if not os.path.exists(path):
        # fallback name if user renamed the file
        alt = os.path.join(directory, "cell_area.nc")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(
                f"找不到网格面积文件: {filename_cell_area} 或 cell_area.nc"
            )

    f_cell = Dataset(path)
    cell_area = np.array(f_cell["area"][:], dtype=float)
    f_cell.close()

    print(f"网格面积(cell_area)形状：{cell_area.shape}")
    print(
        f"网格面积范围：{np.nanmin(cell_area):.6e} - {np.nanmax(cell_area):.6e} m^2"
    )
    return cell_area


def check_continental_lithology(
    cell_area,
    land_area,
    lith_frac,
    max_allowed_inacc=MAX_ALLOWED_INACC,
    remove_erratic=True,
):

    land_area_corr = np.array(land_area, dtype=float, copy=True)
    lith_sum = np.sum(lith_frac, axis=0)

    land_mask = land_area_corr > 0
    areadiff = np.zeros_like(land_area_corr, dtype=float)
    areadiff[land_mask] = (
        np.abs(lith_sum[land_mask] * cell_area[land_mask] - land_area_corr[land_mask])
        / land_area_corr[land_mask]
    )

    erratic = land_mask & (areadiff > max_allowed_inacc)
    nerr = int(np.count_nonzero(erratic))
    tot_land_area = float(np.nansum(land_area_corr))
    area_err = float(np.nansum(land_area_corr[erratic])) if nerr > 0 else 0.0
    max_area_diff = float(np.nanmax(areadiff[land_mask])) if np.any(land_mask) else 0.0
    nland = int(np.count_nonzero(land_mask))

    print("=" * 70)
    print("岩性–面积一致性检查 (Fortran check_continental_lithology)")
    print("=" * 70)
    print(f"  阈值 MAX_ALLOWED_INACC = {max_allowed_inacc}")
    if nerr > 0:
        print("  WARNING: inconsistency of lithological class fraction and continental area.")
        print(f"  异常格点数:               {nerr}")
        print(f"  占陆地格点比例:           {nerr / max(nland, 1):.6e}")
        print(f"  异常格点总面积 (m2):      {area_err:.6e}")
        print(f"  占陆地总面积比例:         {area_err / tot_land_area if tot_land_area > 0 else 0:.6e}")
        print(f"  最大相对偏差:             {max_area_diff:.6e}")
        if remove_erratic:
            land_area_corr[erratic] = 0.0
            print("  已清除异常点: land_area = 0（这些点不再参与总通量积分）")
    else:
        print("  通过检查: 未发现岩性–面积不一致格点")
    print()

    return land_area_corr, nerr


def load_raw_lithology_frac(directory, filename_lithology, skip_water=True):
   
    lithology_path = os.path.join(directory, filename_lithology)
    f_lithology = Dataset(lithology_path)
    dum_lithology = np.array(f_lithology["frac"][:], dtype=float)
    f_lithology.close()

    if skip_water:
        # Fortran: ignore first lithology class (water/ice)
        dum_lithology = dum_lithology[1:, :, :]

    print(f"原始岩性分数形状（用于一致性检查）: {dum_lithology.shape}")
    return dum_lithology


def calculate_erosion(R, slope, erosion_params, verbose=True):
    ke = erosion_params["ke"]
    a = erosion_params["a"]
    b = erosion_params["b"]

    runoff = np.array(R, dtype=float, copy=True)
    slope_safe = np.array(slope, dtype=float, copy=True)
   
    invalid = is_missing_value(runoff) | is_missing_value(slope_safe)
    runoff = np.where(invalid, 0.0, np.maximum(runoff, 0.0))
    slope_safe = np.where(invalid, 0.0, np.maximum(slope_safe, 0.0))

    erosion_rate = ke * np.power(runoff, a) * np.power(slope_safe, b)
    E = np.where(runoff > 0, erosion_rate, 0.0)
    if verbose:
        print(f"  侵蚀率形状: {E.shape}")
        print(f"  侵蚀率范围: {np.nanmin(E):.6e} - {np.nanmax(E):.6e} m/yr")

    return E


def build_uniform_lithology_weighted(shape, lithology_params, verbose=True):
   
    fracs = np.array(FIXED_LITHOLOGY_FRACTIONS, dtype=float)
    params = np.array(lithology_params, dtype=float)
    if fracs.shape != params.shape:
        raise ValueError(
            f"岩性比例与 CaMg 参数长度不一致: {fracs.shape} vs {params.shape}"
        )

    s = fracs.sum()
    if s <= 0:
        raise ValueError("FIXED_LITHOLOGY_FRACTIONS 之和必须 > 0")
    fracs = fracs / s

    L = float(np.dot(fracs, params))
    lithology_weighted = np.full(shape, L, dtype=float)

    if verbose:
        print("使用全球固定岩性比例（不读取 lith.nc）:")
        for name, f, camg in zip(LITHOLOGY_CLASS_NAMES, fracs, params):
            print(f"  {name}: {f * 100:.1f}% × CaMg={camg:g}")
        print(f"  加权岩性 L = {L:.6f}")
        print(f"  加权岩性场形状: {lithology_weighted.shape}（空间均匀）")

    return lithology_weighted


def calculate_weathering_flux(R, T, E, lithology_weighted, area, params, verbose=True):
    kd = params["kd"]
    kw = params["kw"]
    krp = params["krp"]
    sigma = params["sigma"]
    Ea = params["Ea"]
    R_gas = params["R_gas"]
    T0 = params["T0"]
    h0 = params["h0"]

    valid = (
        (area > 0)
        & np.isfinite(R)
        & np.isfinite(T)
        & np.isfinite(E)
        & (R > 0)
        & (E > 0)
        & (T > 0)
    )

    weathering_flux_2D = np.zeros_like(R, dtype=float)
    if np.any(valid):
        Rv = R[valid]
        Tv = T[valid]
        Ev = E[valid]
        Lv = lithology_weighted[valid]

        P0 = krp * Rv * np.exp(-Ea / R_gas * (1.0 / Tv - 1.0 / T0))
        h = np.maximum(0.0, h0 * np.log(np.maximum(P0 / Ev, 1e-10)))
        K_dis = kd * (1.0 - np.exp(-kw * Rv)) * np.exp(
            -Ea / R_gas * (1.0 / Tv - 1.0 / T0)
        )
        xi = -K_dis / (sigma + 1.0) * np.power(np.maximum(h / Ev, 1e-10), sigma + 1.0)
        weathering_flux_2D[valid] = Lv * Ev * (1.0 - np.exp(xi))

    total_flux = np.nansum(weathering_flux_2D * area)

    if verbose:
        print(f"  全球总风化通量: {total_flux:.2e} mol/yr")
        land = area > 0
        if np.any(land):
            print(
                f"  单位面积通量范围(陆地): {np.nanmin(weathering_flux_2D[land]):.6e} - "
                f"{np.nanmax(weathering_flux_2D[land]):.6e} mol/(m^2·yr)"
            )

    return weathering_flux_2D, total_flux


def save_all_results(
    all_weathering_flux, all_total_flux, erosion_flux_first, R, T, area, output_dir, all_params
):
    output_file = os.path.join(output_dir, "A_weathering_flux_result.nc")

    f_out = Dataset(output_file, "w", format="NETCDF4")

    f_out.description = (
        f"Multiple parameter sets weathering flux results. "
        f"Total {len(all_params)} parameter sets."
    )

    ntime, nlat, nlon = all_weathering_flux.shape

    f_out.createDimension("time", ntime)
    f_out.createDimension("lat", nlat)
    f_out.createDimension("lon", nlon)

    lat_var = f_out.createVariable("lat", np.float32, ("lat",))
    lon_var = f_out.createVariable("lon", np.float32, ("lon",))
    time_var = f_out.createVariable("time", np.int32, ("time",))

    flux_var = f_out.createVariable("weathering_rate", np.float32, ("time", "lat", "lon"))
    flux_var.units = "mol/(m^2·yr)"
    flux_var.long_name = "Silicate weathering rate"

    erosion_flux_var = f_out.createVariable("erosion_flux", np.float32, ("lat", "lon"))
    erosion_flux_var.units = "Gt/yr"
    erosion_flux_var.long_name = "Erosion flux (first parameter set only)"

    runoff_var = f_out.createVariable("runoff", np.float32, ("lat", "lon"))
    runoff_var.units = "m/yr"
    runoff_var.long_name = "Runoff"

    temp_var = f_out.createVariable("temperature", np.float32, ("lat", "lon"))
    temp_var.units = "C"
    temp_var.long_name = "Temperature"

    area_var = f_out.createVariable("area", np.float32, ("lat", "lon"))
    area_var.units = "m^2"
    area_var.long_name = "Land area"

    total_flux_var = f_out.createVariable("total_flux", np.float32, ("time",))
    total_flux_var.units = "mol/yr"
    total_flux_var.long_name = "Global total silicate weathering flux"

    lat_var[:] = np.linspace(-90 + 0.25, 90, nlat)
    lon_var[:] = np.linspace(-180 + 0.25, 180, nlon)
    time_var[:] = np.arange(ntime)

    flux_var[:] = all_weathering_flux
    erosion_flux_var[:] = erosion_flux_first
    runoff_var[:] = R
    temp_var[:] = T
    area_var[:] = area
    total_flux_var[:] = all_total_flux

    f_out.close()

    print(f"所有结果已保存到: {output_file}")


def main():
    print("=" * 70)
    print("geoclim-steady-state----YU Liu")
    print("=" * 70)
    print()

    if not os.path.exists(INPUT_DIRECTORY):
        print(f"错误: 数据目录不存在: {INPUT_DIRECTORY}")
        print("请修改代码中的 INPUT_DIRECTORY 路径")
        return

    if not os.path.exists(OUTPUT_DIRECTORY):
        print(f"警告: 输出目录不存在，将创建: {OUTPUT_DIRECTORY}")
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    try:
        print("=" * 70)
        print("读取参数组合")
        print("=" * 70)
        params_list = load_params_from_csv()
        total_param_sets = len(params_list)
        print(f"共需要处理 {total_param_sets} 套参数组合\n")

        print("=" * 70)
        print("读取输入数据")
        print("=" * 70)
        R, T = load_climate_data(INPUT_DIRECTORY, FILENAME_RUNOFF, FILENAME_TEMP)
        area, slope = load_topography_data(INPUT_DIRECTORY, FILENAME_AREA, FILENAME_SLOPE)

        area, n_miss_err = check_continental_missing_values(
            area,
            {"temperature": T, "runoff": R, "slope": slope},
            remove_erratic=True,
        )
        print(f"检查后陆地格点数: {int(np.count_nonzero(area > 0))}")
        print(f"  缺测清除: {n_miss_err}")
        print("  岩性: 使用全球固定比例（已取消 lith.nc / 岩性–面积一致性检查）\n")

        all_results = []
        all_weathering_flux_list = []
        all_total_flux_list = []
        erosion_flux_first = None  # save erosion flux for first parameter set only

        for param_idx, param_set in enumerate(params_list):
            is_first = param_idx == 0

            if is_first:
                print("\n" + "=" * 70)
                print(f"处理参数组合 {param_idx + 1}/{total_param_sets}")
                print("=" * 70)
            else:
                if (param_idx + 1) % 50 == 0 or param_idx == total_param_sets - 1:
                    print(f"处理参数组合 {param_idx + 1}/{total_param_sets}...")

            erosion_params = param_set["erosion_params"]
            model_params = param_set["model_params"]
            lithology_params = param_set["lithology_params"]

            if is_first:
                print(
                    f"\n侵蚀参数: ke={erosion_params['ke']}, "
                    f"a={erosion_params['a']}, b={erosion_params['b']}"
                )
                print(
                    f"风化参数: kd={model_params['kd']}, kw={model_params['kw']}, "
                    f"krp={model_params['krp']}, sigma={model_params['sigma']}"
                )
                print(f"岩性参数: {lithology_params[1:7]}")

            if is_first:
                print("\n计算侵蚀率")
            E = calculate_erosion(R, slope, erosion_params, verbose=is_first)

            # erosion flux = erosion rate * land area * 2.5 / 1e9
            erosion_flux_2D = E * area * 2.5 / 1e9
            total_erosion_flux = np.nansum(erosion_flux_2D)

            if is_first:
                erosion_flux_first = erosion_flux_2D.copy()
                print(f"  侵蚀通量形状: {erosion_flux_2D.shape}")
                print(f"  全球总侵蚀通量: {total_erosion_flux:.6e} Gt/yr")
                print(
                    f"  单位面积侵蚀通量范围: {np.nanmin(erosion_flux_2D):.6e} - "
                    f"{np.nanmax(erosion_flux_2D):.6e} Gt/(m^2·yr)"
                )

            if is_first:
                print("\n构建均匀岩性权重")
            lithology_weighted = build_uniform_lithology_weighted(
                R.shape, lithology_params, verbose=is_first
            )

            if is_first:
                print("\n计算风化通量")
            weathering_flux_2D, total_flux = calculate_weathering_flux(
                R, T, E, lithology_weighted, area, model_params, verbose=is_first
            )

            all_weathering_flux_list.append(weathering_flux_2D)
            all_total_flux_list.append(total_flux)

            all_results.append(
                {
                    "param_index": param_idx,
                    "total_flux": total_flux,
                    "total_erosion_flux": total_erosion_flux,
                    "erosion_params": erosion_params,
                    "model_params": model_params,
                    "lithology_params": lithology_params,
                }
            )

            if is_first:
                print(f"\n参数组合 {param_idx + 1} 完成")
                print(
                    f"  全球总风化通量: {total_flux:.2e} mol/yr "
                    f"({total_flux / 1e12:.2f} Tmol/yr)"
                )
                print(f"  全球总侵蚀通量: {total_erosion_flux:.6e} Gt/yr")

        print("\n" + "=" * 70)
        print("保存所有结果到NetCDF文件")
        print("=" * 70)

        all_weathering_flux = np.array(all_weathering_flux_list)
        all_total_flux = np.array(all_total_flux_list)

        save_all_results(
            all_weathering_flux,
            all_total_flux,
            erosion_flux_first,
            R,
            T,
            area,
            OUTPUT_DIRECTORY,
            all_results,
        )

        all_spatial_sum_flux = []
        for flux_2d in all_weathering_flux_list:
            spatial_sum = np.nansum(flux_2d * area)
            all_spatial_sum_flux.append(spatial_sum)

        all_spatial_sum_flux = np.array(all_spatial_sum_flux)
        mean_spatial_sum = np.nanmean(all_spatial_sum_flux)
        min_spatial_sum = np.nanmin(all_spatial_sum_flux)
        max_spatial_sum = np.nanmax(all_spatial_sum_flux)

        print("=" * 70)
        print(f"\n共处理了 {total_param_sets} 套参数组合")
        print(f"所有结果已保存到: {OUTPUT_DIRECTORY}")
        print("\n" + "-" * 70)
        print("风化通量:")
        print("-" * 70)
        print(
            f"  平均风化通量: {mean_spatial_sum:.2e} mol/yr "
            f"({mean_spatial_sum / 1e12:.2f} Tmol/yr)"
        )
        print(
            f"  最小风化通量: {min_spatial_sum:.2e} mol/yr "
            f"({min_spatial_sum / 1e12:.2f} Tmol/yr)"
        )
        print(
            f"  最大风化通量: {max_spatial_sum:.2e} mol/yr "
            f"({max_spatial_sum / 1e12:.2f} Tmol/yr)"
        )
        print("\n" + "-" * 70)
        print("侵蚀通量:")
        print("-" * 70)
        if erosion_flux_first is not None:
            total_erosion_first = np.nansum(erosion_flux_first)
            print(f"  全球总侵蚀通量: {total_erosion_first:.6e} Gt/yr")
            print(
                f"  单位面积侵蚀通量范围: {np.nanmin(erosion_flux_first):.6e} - "
                f"{np.nanmax(erosion_flux_first):.6e} Gt/(m^2·yr)"
            )
        print("-" * 70)

        summary_file = os.path.join(OUTPUT_DIRECTORY, "A_results_summary.csv")
        summary_df = pd.DataFrame(
            [
                {
                    "param_index": r["param_index"],
                    "total_flux_mol_yr": r["total_flux"],
                    "total_flux_Tmol_yr": r["total_flux"] / 1e12,
                    "total_erosion_flux_Gt_yr": r["total_erosion_flux"],
                    "ke": r["erosion_params"]["ke"],
                    "a": r["erosion_params"]["a"],
                    "b": r["erosion_params"]["b"],
                    "kd": r["model_params"]["kd"],
                    "kw": r["model_params"]["kw"],
                    "krp": r["model_params"]["krp"],
                    "sigma": r["model_params"]["sigma"],
                    "CaMg_1": r["lithology_params"][1],
                    "CaMg_2": r["lithology_params"][2],
                    "CaMg_3": r["lithology_params"][3],
                    "CaMg_4": r["lithology_params"][4],
                    "CaMg_5": r["lithology_params"][5],
                    "CaMg_6": r["lithology_params"][6],
                }
                for r in all_results
            ]
        )
        summary_df.to_csv(summary_file, index=False)
        print(f"结果摘要已保存到: {summary_file}")

    except FileNotFoundError as e:
        print("\n错误: 找不到数据文件或CSV参数文件")
        print(f"  {e}")
        import traceback

        traceback.print_exc()
    except Exception as e:
        print("\n错误: 计算过程中出现异常")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()



# ---------------------------------------------------------------------------
# Parameter combinations (573 sets; moved to file bottom for easier editing above)
# ---------------------------------------------------------------------------
# Embedded from test_params_573.csv (573 parameter sets)
EMBEDDED_PARAMS_CSV = """
a,b,krp,Ea_rp,T0_rp,h0,kd,kw,Ea,T0,sigma,CaMg_1,CaMg_2,CaMg_3,CaMg_4,CaMg_5,CaMg_6,R2
0.5,1,0.01,42000,286,2.73,0.0005,1,42000,286,-0.4,2500,1521,4759,10317,0,2000,0.570427846
0.5,1,0.005,42000,286,2.73,2.00E-05,1,42000,286,-0.1,3500,1521,4759,10317,0,3000,0.50093338
0.5,1,0.015,42000,286,2.73,1.00E-05,0.5,42000,286,0.1,1500,1521,4759,10317,0,1000,0.505323317
0.5,1,0.015,42000,286,2.73,2.00E-05,1,42000,286,-0.1,3000,1521,4759,10317,0,1500,0.500234954
0.5,1,0.015,42000,286,2.73,0.01,0.02,42000,286,-0.4,3500,1521,4759,10317,0,3000,0.501209782
0.5,1,0.015,42000,286,2.73,0.0005,0.01,42000,286,0,2500,1521,4759,10317,0,2000,0.513129835
0.5,1,0.01,42000,286,2.73,0.001,1,42000,286,-0.4,1500,1521,4759,10317,0,1000,0.500493658
0.5,1,0.015,42000,286,2.73,1.00E-05,1,42000,286,0,3000,1521,4759,10317,0,2500,0.513631878
0.5,1,0.015,42000,286,2.73,0.01,0.05,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.502834868
0.5,1,0.015,42000,286,2.73,0.0002,0.05,42000,286,-0.1,3500,1521,4759,10317,0,3000,0.501762649
0.5,1,0.01,42000,286,2.73,0.01,0.001,42000,286,0,2000,1521,4759,10317,0,1500,0.522361523
0.5,1,0.003,42000,286,2.73,0.0005,1,42000,286,-0.4,4000,1521,4759,10317,0,3000,0.504081559
0.5,1,0.005,42000,286,2.73,5.00E-05,1,42000,286,-0.1,3000,1521,4759,10317,0,1500,0.507294065
0.5,1,0.015,42000,286,2.73,0.0002,0.1,42000,286,-0.1,3000,1521,4759,10317,0,2500,0.503372026
0.5,1,0.005,42000,286,2.73,1.00E-05,0.5,42000,286,0.1,2000,1521,4759,10317,0,1500,0.506210154
0.5,1,0.015,42000,286,2.73,0.0001,0.5,42000,286,-0.2,4000,1521,4759,10317,0,2000,0.524319467
0.5,1,0.015,42000,286,2.73,0.0005,1,42000,286,-0.4,4000,1521,4759,10317,0,1500,0.512022535
0.5,1,0.01,42000,286,2.73,0.0002,0.2,42000,286,-0.1,1500,1521,4759,10317,0,1000,0.5020226
0.5,1,0.005,42000,286,2.73,1.00E-05,1,42000,286,0,2500,1521,4759,10317,0,2000,0.505136779
0.5,1,0.01,42000,286,2.73,5.00E-05,1,42000,286,-0.2,4000,1521,4759,10317,0,2000,0.512730177
0.5,1,0.015,42000,286,2.73,5.00E-05,1,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.506939253
0.5,1,0.005,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3500,1521,4759,10317,0,2500,0.502112059
0.5,1,0.01,42000,286,2.73,0.002,0.2,42000,286,-0.4,3500,1521,4759,10317,0,2000,0.500482722
0.5,1,0.003,42000,286,2.73,5.00E-05,1,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.511028151
0.5,1,0.01,42000,286,2.73,5.00E-05,1,42000,286,-0.2,3500,1521,4759,10317,0,3000,0.548941862
0.5,1,0.005,42000,286,2.73,0.0005,1,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.509797802
0.5,1,0.015,42000,286,2.73,0.001,0.5,42000,286,-0.4,3000,1521,4759,10317,0,1500,0.510225434
0.5,1,0.015,42000,286,2.73,1.00E-05,0.5,42000,286,0,4000,1521,4759,10317,0,2500,0.50181505
0.5,1,0.015,42000,286,2.73,0.0005,1,42000,286,-0.4,4000,1521,4759,10317,0,2500,0.513092659
0.5,1,0.015,42000,286,2.73,5.00E-05,1,42000,286,-0.1,1500,1521,4759,10317,0,1000,0.518135596
0.5,1,0.015,42000,286,2.73,0.0002,0.2,42000,286,-0.2,4000,1521,4759,10317,0,3000,0.507723486
0.5,1,0.01,42000,286,2.73,5.00E-05,1,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.531660519
0.5,1,0.01,42000,286,2.73,5.00E-05,0.2,42000,286,0,2500,1521,4759,10317,0,2000,0.514404734
0.5,1,0.015,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3000,1521,4759,10317,0,1500,0.506782778
0.5,1,0.015,42000,286,2.73,1.00E-05,1,42000,286,0,2000,1521,4759,10317,0,1500,0.54768565
0.5,1,0.015,42000,286,2.73,1.00E-05,0.2,42000,286,0.1,3000,1521,4759,10317,0,2000,0.500744664
0.5,1,0.005,42000,286,2.73,0.0001,1,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.515866368
0.5,1,0.015,42000,286,2.73,1.00E-05,0.5,42000,286,0.1,2000,1521,4759,10317,0,1500,0.514250427
0.5,1,0.01,42000,286,2.73,0.0002,0.2,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.502091912
0.5,1,0.01,42000,286,2.73,0.0001,1,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.502520534
0.5,1,0.01,42000,286,2.73,2.00E-05,1,42000,286,-0.1,3000,1521,4759,10317,0,2500,0.546545707
0.5,1,0.015,42000,286,2.73,1.00E-05,1,42000,286,0,3000,1521,4759,10317,0,2000,0.53451505
0.5,1,0.015,42000,286,2.73,0.0005,0.1,42000,286,-0.2,4000,1521,4759,10317,0,2000,0.500269463
0.5,1,0.005,42000,286,2.73,0.001,1,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.52065413
0.5,1,0.01,42000,286,2.73,0.002,0.02,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.509771659
0.5,1,0.01,42000,286,2.73,0.001,0.5,42000,286,-0.4,3000,1521,4759,10317,0,2500,0.500001196
0.5,1,0.003,42000,286,2.73,0.0005,1,42000,286,-0.4,2500,1521,4759,10317,0,2000,0.50386747
0.5,1,0.015,42000,286,2.73,0.005,0.005,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.502866425
0.5,1,0.005,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.508877471
0.5,1,0.015,42000,286,2.73,5.00E-05,1,42000,286,-0.2,4000,1521,4759,10317,0,2500,0.548152859
0.5,1,0.015,42000,286,2.73,0.0005,0.5,42000,286,-0.4,3000,1521,4759,10317,0,2500,0.533122029
0.5,1,0.015,42000,286,2.73,1.00E-05,0.2,42000,286,0.1,3000,1521,4759,10317,0,2500,0.508350946
0.5,1,0.015,42000,286,2.73,0.005,0.001,42000,286,0,3000,1521,4759,10317,0,2500,0.515380283
0.5,1,0.01,42000,286,2.73,0.0001,1,42000,286,-0.2,2000,1521,4759,10317,0,1000,0.502492586
0.5,1,0.015,42000,286,2.73,0.0005,0.005,42000,286,0.1,2000,1521,4759,10317,0,1500,0.512185657
0.5,1,0.005,42000,286,2.73,0.002,0.2,42000,286,-0.4,3000,1521,4759,10317,0,2500,0.506432315
0.5,1,0.015,42000,286,2.73,0.0001,0.5,42000,286,-0.2,4000,1521,4759,10317,0,2500,0.524903946
0.5,1,0.01,42000,286,2.73,0.002,0.2,42000,286,-0.4,2500,1521,4759,10317,0,1500,0.500411533
0.5,1,0.015,42000,286,2.73,0.0005,0.1,42000,286,-0.2,3000,1521,4759,10317,0,1500,0.503333792
0.5,1,0.01,42000,286,2.73,0.0005,1,42000,286,-0.4,3500,1521,4759,10317,0,3000,0.512097436
0.5,1,0.005,42000,286,2.73,0.0002,0.5,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.501537572
0.5,1,0.015,42000,286,2.73,0.0005,1,42000,286,-0.4,4000,1521,4759,10317,0,2000,0.528389349
0.5,1,0.015,42000,286,2.73,0.0001,1,42000,286,-0.2,3000,1521,4759,10317,0,1500,0.521127446
0.5,1,0.01,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.537839228
0.5,1,0.005,42000,286,2.73,0.0002,0.5,42000,286,-0.2,3000,1521,4759,10317,0,1500,0.50453705
0.5,1,0.01,42000,286,2.73,1.00E-05,1,42000,286,0,4000,1521,4759,10317,0,2000,0.505275597
0.5,1,0.005,42000,286,2.73,0.0001,1,42000,286,-0.2,4000,1521,4759,10317,0,2000,0.504486045
0.5,1,0.005,42000,286,2.73,0.0005,1,42000,286,-0.4,3000,1521,4759,10317,0,2500,0.554309651
0.5,1,0.015,42000,286,2.73,0.0005,1,42000,286,-0.4,2500,1521,4759,10317,0,1500,0.547780939
0.5,1,0.005,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3500,1521,4759,10317,0,3000,0.50789425
0.5,1,0.015,42000,286,2.73,1.00E-05,0.5,42000,286,0,3500,1521,4759,10317,0,3000,0.51369513
0.5,1,0.015,42000,286,2.73,0.0005,0.5,42000,286,-0.4,4000,1521,4759,10317,0,3000,0.534529588
0.5,1,0.015,42000,286,2.73,0.001,0.5,42000,286,-0.4,2500,1521,4759,10317,0,2000,0.529204896
0.5,1,0.01,42000,286,2.73,0.0001,0.2,42000,286,-0.1,3500,1521,4759,10317,0,2000,0.510794849
0.5,1,0.01,42000,286,2.73,0.0002,0.2,42000,286,-0.2,4000,1521,4759,10317,0,2500,0.50841473
0.5,1,0.005,42000,286,2.73,0.0005,1,42000,286,-0.4,3500,1521,4759,10317,0,2000,0.535711656
0.5,1,0.005,42000,286,2.73,5.00E-05,1,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.511575254
0.5,1,0.01,42000,286,2.73,2.00E-05,0.5,42000,286,0,3000,1521,4759,10317,0,1500,0.508707635
0.5,1,0.01,42000,286,2.73,2.00E-05,1,42000,286,-0.1,4000,1521,4759,10317,0,3000,0.519815558
0.5,1,0.005,42000,286,2.73,0.005,0.1,42000,286,-0.4,2500,1521,4759,10317,0,2000,0.501876156
0.5,1,0.01,42000,286,2.73,2.00E-05,1,42000,286,-0.1,4000,1521,4759,10317,0,2500,0.526095557
0.5,1,0.003,42000,286,2.73,5.00E-05,1,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.51285636
0.5,1,0.015,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.528500448
0.5,1,0.01,42000,286,2.73,0.0001,1,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.560565933
0.5,1,0.015,42000,286,2.73,1.00E-05,0.5,42000,286,0,3000,1521,4759,10317,0,2500,0.520243493
0.5,1,0.015,42000,286,2.73,1.00E-05,1,42000,286,0,3500,1521,4759,10317,0,1500,0.509623603
0.5,1,0.005,42000,286,2.73,0.0001,0.5,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.51701956
0.5,1,0.01,42000,286,2.73,0.002,0.02,42000,286,-0.2,3500,1521,4759,10317,0,2500,0.509621268
0.5,1,0.005,42000,286,2.73,0.0001,0.5,42000,286,-0.1,1500,1521,4759,10317,0,1000,0.500680314
0.5,1,0.01,42000,286,2.73,2.00E-05,1,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.52715448
0.5,1,0.01,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,3000,1521,4759,10317,0,2500,0.515494876
0.5,1,0.005,42000,286,2.73,5.00E-05,0.2,42000,286,0,2500,1521,4759,10317,0,2000,0.505383378
0.5,1,0.01,42000,286,2.73,0.0001,0.5,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.54193607
0.5,1,0.01,42000,286,2.73,0.0002,0.1,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.503453882
0.5,1,0.015,42000,286,2.73,5.00E-05,1,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.567963616
0.5,1,0.003,42000,286,2.73,0.0001,0.5,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.504722619
0.5,1,0.005,42000,286,2.73,0.0002,1,42000,286,-0.2,1500,1521,4759,10317,0,1000,0.505378491
0.5,1,0.015,42000,286,2.73,0.0001,1,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.522197706
0.5,1,0.01,42000,286,2.73,2.00E-05,0.5,42000,286,0,3500,1521,4759,10317,0,2000,0.500440915
0.5,1,0.005,42000,286,2.73,5.00E-05,0.1,42000,286,0.1,2000,1521,4759,10317,0,1500,0.500402354
0.5,1,0.015,42000,286,2.73,0.01,0.002,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.514264757
0.5,1,0.015,42000,286,2.73,0.002,0.02,42000,286,-0.2,3500,1521,4759,10317,0,3000,0.502265275
0.5,1,0.01,42000,286,2.73,0.0001,0.5,42000,286,-0.2,4000,1521,4759,10317,0,3000,0.511585109
0.5,1,0.003,42000,286,2.73,0.0001,1,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.508591462
0.5,1,0.015,42000,286,2.73,0.0002,0.5,42000,286,-0.2,2500,1521,4759,10317,0,1500,0.501869419
0.5,1,0.005,42000,286,2.73,0.001,0.5,42000,286,-0.4,2500,1521,4759,10317,0,2000,0.534475589
0.5,1,0.003,42000,286,2.73,0.0002,1,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.50839725
0.5,1,0.015,42000,286,2.73,0.001,0.5,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.538406109
0.5,1,0.005,42000,286,2.73,5.00E-05,1,42000,286,-0.2,3500,1521,4759,10317,0,3000,0.513666726
0.5,1,0.015,42000,286,2.73,0.005,0.005,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.513351006
0.5,1,0.005,42000,286,2.73,0.01,0.01,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.506185103
0.5,1,0.003,42000,286,2.73,0.0005,1,42000,286,-0.4,4000,1521,4759,10317,0,2500,0.505212965
0.5,1,0.005,42000,286,2.73,1.00E-05,1,42000,286,0,3000,1521,4759,10317,0,2500,0.507913978
0.5,1,0.01,42000,286,2.73,1.00E-05,1,42000,286,0,3500,1521,4759,10317,0,2500,0.507676472
0.5,1,0.003,42000,286,2.73,0.0002,1,42000,286,-0.2,1500,1521,4759,10317,0,1000,0.500710766
0.5,1,0.005,42000,286,2.73,0.0001,1,42000,286,-0.2,3500,1521,4759,10317,0,1500,0.508365454
0.5,1,0.015,42000,286,2.73,2.00E-05,1,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.512048821
0.5,1,0.01,42000,286,2.73,0.001,0.5,42000,286,-0.4,3500,1521,4759,10317,0,2000,0.505411927
0.5,1,0.015,42000,286,2.73,0.001,0.2,42000,286,-0.4,4000,1521,4759,10317,0,3000,0.504316495
0.5,1,0.01,42000,286,2.73,1.00E-05,0.5,42000,286,0.1,2500,1521,4759,10317,0,1500,0.505003331
0.5,1,0.005,42000,286,2.73,0.001,0.5,42000,286,-0.4,2500,1521,4759,10317,0,1500,0.50440514
0.5,1,0.01,42000,286,2.73,2.00E-05,1,42000,286,0,1500,1521,4759,10317,0,1000,0.514929586
0.5,1,0.01,42000,286,2.73,2.00E-05,1,42000,286,0,2000,1521,4759,10317,0,1500,0.510263398
0.5,1,0.005,42000,286,2.73,0.0001,1,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.534881929
0.5,1,0.005,42000,286,2.73,0.002,0.2,42000,286,-0.4,2500,1521,4759,10317,0,2000,0.511227808
0.5,1,0.01,42000,286,2.73,0.0005,0.5,42000,286,-0.4,4000,1521,4759,10317,0,2500,0.504914751
0.5,1,0.01,42000,286,2.73,1.00E-05,1,42000,286,0,2500,1521,4759,10317,0,1500,0.51945319
0.5,1,0.01,42000,286,2.73,2.00E-05,1,42000,286,-0.1,4000,1521,4759,10317,0,2000,0.509830779
0.5,1,0.015,42000,286,2.73,2.00E-05,0.02,42000,286,0.3,2000,1521,4759,10317,0,1500,0.50010659
0.5,1,0.015,42000,286,2.73,0.0005,1,42000,286,-0.4,3000,1521,4759,10317,0,2500,0.543659807
0.5,1,0.015,42000,286,2.73,2.00E-05,0.1,42000,286,0.1,2500,1521,4759,10317,0,2000,0.512138791
0.5,1,0.01,42000,286,2.73,0.005,0.05,42000,286,-0.4,3000,1521,4759,10317,0,2500,0.504684832
0.5,1,0.005,42000,286,2.73,0.0001,1,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.545462127
0.5,1,0.01,42000,286,2.73,0.0001,0.5,42000,286,-0.1,1500,1521,4759,10317,0,1000,0.504199277
0.5,1,0.01,42000,286,2.73,0.0002,0.2,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.502865697
0.5,1,0.005,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,3000,1521,4759,10317,0,2500,0.513341542
0.5,1,0.01,42000,286,2.73,0.01,0.005,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.5030515
0.5,1,0.015,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,3500,1521,4759,10317,0,1500,0.508728481
0.5,1,0.01,42000,286,2.73,0.0001,0.2,42000,286,-0.1,3500,1521,4759,10317,0,2500,0.504152998
0.5,1,0.015,42000,286,2.73,0.002,0.02,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.509496625
0.5,1,0.01,42000,286,2.73,2.00E-05,0.5,42000,286,0,2500,1521,4759,10317,0,2000,0.532044531
0.5,1,0.003,42000,286,2.73,0.0001,1,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.508547947
0.5,1,0.01,42000,286,2.73,2.00E-05,0.2,42000,286,0.1,2500,1521,4759,10317,0,1500,0.504912453
0.5,1,0.005,42000,286,2.73,2.00E-05,1,42000,286,0,2000,1521,4759,10317,0,1500,0.529864301
0.5,1,0.01,42000,286,2.73,1.00E-05,1,42000,286,0,2500,1521,4759,10317,0,2000,0.545940297
0.5,1,0.015,42000,286,2.73,5.00E-05,1,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.541135159
0.5,1,0.015,42000,286,2.73,5.00E-05,0.1,42000,286,0,3500,1521,4759,10317,0,2500,0.507404013
0.5,1,0.01,42000,286,2.73,5.00E-05,0.1,42000,286,0.1,2000,1521,4759,10317,0,1500,0.501982971
0.5,1,0.015,42000,286,2.73,5.00E-05,0.2,42000,286,-0.1,3500,1521,4759,10317,0,3000,0.502058706
0.5,1,0.005,42000,286,2.73,0.0005,0.05,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.508082396
0.5,1,0.01,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.545889527
0.5,1,0.015,42000,286,2.73,0.0001,0.2,42000,286,-0.1,3000,1521,4759,10317,0,1500,0.51005632
0.5,1,0.005,42000,286,2.73,0.001,1,42000,286,-0.4,1500,1521,4759,10317,0,1000,0.50730848
0.5,1,0.01,42000,286,2.73,0.0001,1,42000,286,-0.2,3500,1521,4759,10317,0,1500,0.512438431
0.5,1,0.005,42000,286,2.73,0.001,0.5,42000,286,-0.4,3500,1521,4759,10317,0,2500,0.50163331
0.5,1,0.005,42000,286,2.73,5.00E-05,1,42000,286,-0.1,1500,1521,4759,10317,0,1000,0.50200774
0.5,1,0.015,42000,286,2.73,5.00E-05,0.2,42000,286,0,2500,1521,4759,10317,0,1500,0.516402936
0.5,1,0.01,42000,286,2.73,0.01,0.005,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.512489304
0.5,1,0.005,42000,286,2.73,0.0005,1,42000,286,-0.4,4000,1521,4759,10317,0,2500,0.531398952
0.5,1,0.005,42000,286,2.73,2.00E-05,0.5,42000,286,0,2500,1521,4759,10317,0,2000,0.511081114
0.5,1,0.015,42000,286,2.73,0.0001,1,42000,286,-0.2,1500,1521,4759,10317,0,1000,0.523340411
0.5,1,0.005,42000,286,2.73,0.0005,1,42000,286,-0.4,3500,1521,4759,10317,0,3000,0.529323879
0.5,1,0.01,42000,286,2.73,0.0002,0.2,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.527179862
0.5,1,0.01,42000,286,2.73,5.00E-05,1,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.505796923
0.5,1,0.01,42000,286,2.73,0.0001,0.2,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.535841775
0.5,1,0.015,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.537056368
0.5,1,0.015,42000,286,2.73,2.00E-05,1,42000,286,-0.1,3500,1521,4759,10317,0,2000,0.537469956
0.5,1,0.015,42000,286,2.73,0.0002,0.2,42000,286,-0.2,4000,1521,4759,10317,0,2000,0.505651735
0.5,1,0.003,42000,286,2.73,0.0005,1,42000,286,-0.4,3000,1521,4759,10317,0,2500,0.523056189
0.5,1,0.005,42000,286,2.73,0.0005,1,42000,286,-0.4,2500,1521,4759,10317,0,2000,0.55417646
0.5,1,0.015,42000,286,2.73,0.005,0.01,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.534582754
0.5,1,0.01,42000,286,2.73,5.00E-05,1,42000,286,-0.1,2000,1521,4759,10317,0,1000,0.505521052
0.5,1,0.015,42000,286,2.73,1.00E-05,1,42000,286,0,4000,1521,4759,10317,0,2000,0.503688228
0.5,1,0.015,42000,286,2.73,2.00E-05,1,42000,286,-0.1,3500,1521,4759,10317,0,2500,0.545055632
0.5,1,0.01,42000,286,2.73,0.0002,0.5,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.533581038
0.5,1,0.003,42000,286,2.73,0.001,1,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.525640389
0.5,1,0.005,42000,286,2.73,0.001,0.5,42000,286,-0.4,3500,1521,4759,10317,0,2000,0.50951786
0.5,1,0.01,42000,286,2.73,0.0005,0.02,42000,286,0,2500,1521,4759,10317,0,2000,0.50019163
0.5,1,0.015,42000,286,2.73,0.005,0.05,42000,286,-0.4,3500,1521,4759,10317,0,3000,0.506999375
0.5,1,0.01,42000,286,2.73,0.001,0.02,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.512716786
0.5,1,0.015,42000,286,2.73,5.00E-05,0.05,42000,286,0.1,2500,1521,4759,10317,0,1500,0.500463874
0.5,1,0.01,42000,286,2.73,0.0005,1,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.547815976
0.5,1,0.015,42000,286,2.73,5.00E-05,0.1,42000,286,0,3000,1521,4759,10317,0,2000,0.503408635
0.5,1,0.003,42000,286,2.73,2.00E-05,1,42000,286,0,2000,1521,4759,10317,0,1500,0.503003704
0.5,1,0.015,42000,286,2.73,5.00E-05,1,42000,286,-0.2,4000,1521,4759,10317,0,3000,0.541576342
0.5,1,0.015,42000,286,2.73,0.01,0.002,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.516743301
0.5,1,0.01,42000,286,2.73,0.0002,0.5,42000,286,-0.2,1500,1521,4759,10317,0,1000,0.509418173
0.5,1,0.005,42000,286,2.73,0.0005,0.2,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.519249572
0.5,1,0.015,42000,286,2.73,0.0002,0.1,42000,286,-0.1,3500,1521,4759,10317,0,2000,0.506636637
0.5,1,0.015,42000,286,2.73,1.00E-05,1,42000,286,0,2500,1521,4759,10317,0,1500,0.535815918
0.5,1,0.01,42000,286,2.73,5.00E-05,0.1,42000,286,0,3000,1521,4759,10317,0,2500,0.500172184
0.5,1,0.015,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3500,1521,4759,10317,0,3000,0.518890822
0.5,1,0.01,42000,286,2.73,5.00E-05,0.05,42000,286,0.1,2500,1521,4759,10317,0,2000,0.508669563
0.5,1,0.01,42000,286,2.73,0.0005,1,42000,286,-0.4,3000,1521,4759,10317,0,1500,0.530335386
0.5,1,0.015,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.546818771
0.5,1,0.01,42000,286,2.73,1.00E-05,1,42000,286,0,3000,1521,4759,10317,0,1500,0.508602967
0.5,1,0.015,42000,286,2.73,0.01,0.005,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.506585104
0.5,1,0.015,42000,286,2.73,2.00E-05,1,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.506825167
0.5,1,0.01,42000,286,2.73,1.00E-05,1,42000,286,0,2000,1521,4759,10317,0,1500,0.529306348
0.5,1,0.01,42000,286,2.73,5.00E-05,1,42000,286,-0.2,3500,1521,4759,10317,0,2500,0.54644968
0.5,1,0.015,42000,286,2.73,2.00E-05,1,42000,286,-0.1,3500,1521,4759,10317,0,3000,0.531034828
0.5,1,0.01,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.52401704
0.5,1,0.015,42000,286,2.73,2.00E-05,1,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.556693086
0.5,1,0.015,42000,286,2.73,0.0005,0.1,42000,286,-0.2,3500,1521,4759,10317,0,2500,0.503230206
0.5,1,0.01,42000,286,2.73,0.002,0.2,42000,286,-0.4,3000,1521,4759,10317,0,2500,0.504421416
0.5,1,0.015,42000,286,2.73,5.00E-05,0.05,42000,286,0.1,3000,1521,4759,10317,0,2000,0.504555331
0.5,1,0.015,42000,286,2.73,0.0002,0.05,42000,286,0,2500,1521,4759,10317,0,1500,0.507348726
0.5,1,0.015,42000,286,2.73,0.0005,1,42000,286,-0.4,3000,1521,4759,10317,0,2000,0.55674889
0.5,1,0.015,42000,286,2.73,1.00E-05,1,42000,286,0,3000,1521,4759,10317,0,1500,0.523039436
0.5,1,0.01,42000,286,2.73,5.00E-05,0.2,42000,286,0,2000,1521,4759,10317,0,1500,0.529844669
0.5,1,0.015,42000,286,2.73,2.00E-05,0.5,42000,286,0,3000,1521,4759,10317,0,1500,0.513635221
0.5,1,0.015,42000,286,2.73,0.0001,0.02,42000,286,0.1,3000,1521,4759,10317,0,2500,0.502222239
0.5,1,0.005,42000,286,2.73,0.0002,0.5,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.53451638
0.5,1,0.01,42000,286,2.73,1.00E-05,1,42000,286,0,3000,1521,4759,10317,0,2000,0.53283017
0.5,1,0.015,42000,286,2.73,0.0002,0.2,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.532319228
0.5,1,0.01,42000,286,2.73,0.0005,1,42000,286,-0.4,3500,1521,4759,10317,0,2000,0.546111352
0.5,1,0.01,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,4000,1521,4759,10317,0,2000,0.502213452
0.5,1,0.015,42000,286,2.73,2.00E-05,0.5,42000,286,0,2500,1521,4759,10317,0,2000,0.526372812
0.5,1,0.005,42000,286,2.73,2.00E-05,1,42000,286,0,2500,1521,4759,10317,0,1500,0.512362999
0.5,1,0.005,42000,286,2.73,0.0001,1,42000,286,-0.2,2500,1521,4759,10317,0,1500,0.533995291
0.5,1,0.015,42000,286,2.73,1.00E-05,1,42000,286,0,2500,1521,4759,10317,0,2000,0.549356674
0.5,1,0.01,42000,286,2.73,5.00E-05,1,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.526272517
0.5,1,0.01,42000,286,2.73,2.00E-05,1,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.534674632
0.5,1,0.015,42000,286,2.73,2.00E-05,1,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.547460549
0.5,1,0.01,42000,286,2.73,0.0001,1,42000,286,-0.2,2500,1521,4759,10317,0,1500,0.545260487
0.5,1,0.005,42000,286,2.73,0.0002,0.2,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.516137713
0.5,1,0.01,42000,286,2.73,0.001,0.5,42000,286,-0.4,2500,1521,4759,10317,0,1500,0.52370857
0.5,1,0.015,42000,286,2.73,0.0002,0.2,42000,286,-0.2,3500,1521,4759,10317,0,2500,0.528668044
0.5,1,0.01,42000,286,2.73,2.00E-05,1,42000,286,-0.1,3500,1521,4759,10317,0,2000,0.518806395
0.5,1,0.005,42000,286,2.73,5.00E-05,1,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.541076621
0.5,1,0.005,42000,286,2.73,0.0005,1,42000,286,-0.4,2500,1521,4759,10317,0,1500,0.505240995
0.5,1,0.01,42000,286,2.73,2.00E-05,0.5,42000,286,0,2000,1521,4759,10317,0,1500,0.535079882
0.5,1,0.01,42000,286,2.73,2.00E-05,0.5,42000,286,0,2500,1521,4759,10317,0,1500,0.522316274
0.5,1,0.01,42000,286,2.73,1.00E-05,0.5,42000,286,0.1,2000,1521,4759,10317,0,1500,0.522973124
0.5,1,0.015,42000,286,2.73,0.01,0.005,42000,286,-0.2,2500,1521,4759,10317,0,1500,0.511395581
0.5,1,0.015,42000,286,2.73,2.00E-05,1,42000,286,-0.1,4000,1521,4759,10317,0,2000,0.526912652
0.5,1,0.005,42000,286,2.73,5.00E-05,1,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.524502253
0.5,1,0.01,42000,286,2.73,0.0005,1,42000,286,-0.4,3500,1521,4759,10317,0,2500,0.538714306
0.5,1,0.01,42000,286,2.73,0.0005,0.1,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.533098529
0.5,1,0.01,42000,286,2.73,1.00E-05,1,42000,286,0,3500,1521,4759,10317,0,2000,0.519224906
0.5,1,0.003,42000,286,2.73,0.0002,0.5,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.505505606
0.5,1,0.015,42000,286,2.73,0.0001,0.5,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.520703533
0.5,1,0.015,42000,286,2.73,5.00E-05,0.2,42000,286,0,2500,1521,4759,10317,0,2000,0.503368642
0.5,1,0.015,42000,286,2.73,0.0001,0.5,42000,286,-0.2,2500,1521,4759,10317,0,1500,0.514442431
0.5,1,0.005,42000,286,2.73,0.0001,1,42000,286,-0.2,3000,1521,4759,10317,0,1500,0.52153466
0.5,1,0.015,42000,286,2.73,1.00E-05,1,42000,286,0,3500,1521,4759,10317,0,2000,0.51924464
0.5,1,0.01,42000,286,2.73,0.0002,0.2,42000,286,-0.2,3500,1521,4759,10317,0,3000,0.517105891
0.5,1,0.015,42000,286,2.73,2.00E-05,0.2,42000,286,0.1,2000,1521,4759,10317,0,1500,0.520359564
0.5,1,0.01,42000,286,2.73,0.0002,0.05,42000,286,0,2500,1521,4759,10317,0,1500,0.50894112
0.5,1,0.015,42000,286,2.73,0.002,0.02,42000,286,-0.2,4000,1521,4759,10317,0,2500,0.504446078
0.5,1,0.01,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,3500,1521,4759,10317,0,1500,0.502256665
0.5,1,0.015,42000,286,2.73,2.00E-05,1,42000,286,-0.1,3000,1521,4759,10317,0,2500,0.556545315
0.5,1,0.015,42000,286,2.73,0.002,0.2,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.518809617
0.5,1,0.01,42000,286,2.73,1.00E-05,1,42000,286,0,3000,1521,4759,10317,0,2500,0.522506502
0.5,1,0.01,42000,286,2.73,0.0001,0.2,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.512199761
0.5,1,0.005,42000,286,2.73,0.001,1,42000,286,-0.4,2500,1521,4759,10317,0,1500,0.500669922
0.5,1,0.003,42000,286,2.73,0.0001,1,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.52004031
0.5,1,0.01,42000,286,2.73,2.00E-05,1,42000,286,-0.1,3500,1521,4759,10317,0,2500,0.536562887
0.5,1,0.015,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,3000,1521,4759,10317,0,1500,0.523217242
0.5,1,0.01,42000,286,2.73,0.0005,1,42000,286,-0.4,3500,1521,4759,10317,0,1500,0.519840709
0.5,1,0.003,42000,286,2.73,0.001,1,42000,286,-0.4,2500,1521,4759,10317,0,1500,0.50904825
0.5,1,0.01,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.531773662
0.5,1,0.015,42000,286,2.73,2.00E-05,0.5,42000,286,0,2500,1521,4759,10317,0,1500,0.529166006
0.5,1,0.005,42000,286,2.73,0.0001,1,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.549425818
0.5,1,0.015,42000,286,2.73,2.00E-05,1,42000,286,-0.1,4000,1521,4759,10317,0,2500,0.533135194
0.5,1,0.015,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,3000,1521,4759,10317,0,2500,0.502631165
0.5,1,0.005,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.519593982
0.5,1,0.015,42000,286,2.73,1.00E-05,0.5,42000,286,0,3500,1521,4759,10317,0,2500,0.511282116
0.5,1,0.015,42000,286,2.73,5.00E-05,1,42000,286,-0.2,4000,1521,4759,10317,0,2000,0.532444991
0.5,1,0.015,42000,286,2.73,0.0001,1,42000,286,-0.2,3500,1521,4759,10317,0,1500,0.502729144
0.5,1,0.015,42000,286,2.73,0.0001,1,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.502732965
0.5,1,0.015,42000,286,2.73,2.00E-05,0.5,42000,286,0,3000,1521,4759,10317,0,2000,0.50903388
0.5,1,0.01,42000,286,2.73,0.001,0.02,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.514525322
0.5,1,0.01,42000,286,2.73,5.00E-05,1,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.519886835
0.5,1,0.01,42000,286,2.73,5.00E-05,1,42000,286,-0.2,4000,1521,4759,10317,0,2500,0.537676519
0.5,1,0.01,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.545877153
0.5,1,0.005,42000,286,2.73,0.001,0.5,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.512849641
0.5,1,0.005,42000,286,2.73,0.0002,0.5,42000,286,-0.2,2500,1521,4759,10317,0,1500,0.519942738
0.5,1,0.01,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,3000,1521,4759,10317,0,1500,0.514942992
0.5,1,0.01,42000,286,2.73,2.00E-05,0.5,42000,286,0,3000,1521,4759,10317,0,2000,0.516432826
0.5,1,0.01,42000,286,2.73,5.00E-05,1,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.554611814
0.5,1,0.015,42000,286,2.73,2.00E-05,0.1,42000,286,0.1,3000,1521,4759,10317,0,2000,0.501069183
0.5,1,0.01,42000,286,2.73,0.0002,0.02,42000,286,0.1,2000,1521,4759,10317,0,1500,0.514148866
0.5,1,0.01,42000,286,2.73,0.0001,1,42000,286,-0.2,3000,1521,4759,10317,0,1500,0.529130177
0.5,1,0.015,42000,286,2.73,5.00E-05,1,42000,286,-0.2,3500,1521,4759,10317,0,2500,0.558336643
0.5,1,0.015,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.549967147
0.5,1,0.015,42000,286,2.73,5.00E-05,1,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.549115934
0.5,1,0.015,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.544296842
0.5,1,0.01,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.526900988
0.5,1,0.015,42000,286,2.73,1.00E-05,0.2,42000,286,0.1,2500,1521,4759,10317,0,2000,0.511238072
0.5,1,0.01,42000,286,2.73,0.0001,1,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.520734992
0.5,1,0.01,42000,286,2.73,0.0005,0.5,42000,286,-0.4,3500,1521,4759,10317,0,3000,0.53035567
0.5,1,0.015,42000,286,2.73,2.00E-05,0.5,42000,286,0,2000,1521,4759,10317,0,1500,0.543940188
0.5,1,0.015,42000,286,2.73,5.00E-05,1,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.556169596
0.5,1,0.005,42000,286,2.73,0.0001,1,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.51985106
0.5,1,0.01,42000,286,2.73,2.00E-05,1,42000,286,-0.1,3500,1521,4759,10317,0,3000,0.531281379
0.5,1,0.015,42000,286,2.73,0.002,0.02,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.528295249
0.5,1,0.01,42000,286,2.73,0.0001,1,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.538587472
0.5,1,0.015,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,3500,1521,4759,10317,0,2000,0.512267893
0.5,1,0.015,42000,286,2.73,0.002,0.2,42000,286,-0.4,3000,1521,4759,10317,0,2000,0.511882586
0.5,1,0.005,42000,286,2.73,0.0005,1,42000,286,-0.4,3000,1521,4759,10317,0,2000,0.54539857
0.5,1,0.015,42000,286,2.73,2.00E-05,1,42000,286,-0.1,4000,1521,4759,10317,0,3000,0.518217915
0.5,1,0.015,42000,286,2.73,0.005,0.005,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.529622327
0.5,1,0.015,42000,286,2.73,0.0001,1,42000,286,-0.2,2500,1521,4759,10317,0,1500,0.539041302
0.5,1,0.005,42000,286,2.73,5.00E-05,1,42000,286,-0.2,4000,1521,4759,10317,0,3000,0.506408931
0.5,1,0.015,42000,286,2.73,0.0001,1,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.556219051
0.5,1,0.015,42000,286,2.73,1.00E-05,0.5,42000,286,0,4000,1521,4759,10317,0,3000,0.503147011
0.5,1,0.015,42000,286,2.73,0.005,0.001,42000,286,0,3000,1521,4759,10317,0,2000,0.503691255
0.5,1,0.005,42000,286,2.73,0.0005,0.2,42000,286,-0.2,2500,1521,4759,10317,0,1500,0.502714414
0.5,1,0.01,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,3500,1521,4759,10317,0,2000,0.517167062
0.5,1,0.015,42000,286,2.73,0.005,0.05,42000,286,-0.4,3500,1521,4759,10317,0,2500,0.507531194
0.5,1,0.01,42000,286,2.73,0.0005,0.2,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.509018833
0.5,1,0.015,42000,286,2.73,0.0001,0.2,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.542174025
0.5,1,0.01,42000,286,2.73,0.0001,1,42000,286,-0.2,1500,1521,4759,10317,0,1000,0.512216875
0.5,1,0.01,42000,286,2.73,5.00E-05,1,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.526303086
0.5,1,0.005,42000,286,2.73,0.0005,1,42000,286,-0.4,4000,1521,4759,10317,0,2000,0.52533511
0.5,1,0.01,42000,286,2.73,5.00E-05,1,42000,286,-0.2,4000,1521,4759,10317,0,3000,0.539062319
0.5,1,0.005,42000,286,2.73,0.0005,0.1,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.501244028
0.5,1,0.01,42000,286,2.73,0.0002,0.1,42000,286,-0.1,3000,1521,4759,10317,0,2500,0.510373391
0.5,1,0.015,42000,286,2.73,0.0001,0.2,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.528256124
0.5,1,0.01,42000,286,2.73,0.0001,0.2,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.523619843
0.5,1,0.005,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.5101248
0.5,1,0.01,42000,286,2.73,0.002,0.2,42000,286,-0.4,2500,1521,4759,10317,0,2000,0.526859885
0.5,1,0.01,42000,286,2.73,5.00E-05,0.2,42000,286,0,2500,1521,4759,10317,0,1500,0.515231027
0.5,1,0.005,42000,286,2.73,5.00E-05,0.5,42000,286,-0.1,3500,1521,4759,10317,0,2500,0.500578786
0.5,1,0.015,42000,286,2.73,5.00E-05,1,42000,286,-0.2,3500,1521,4759,10317,0,3000,0.552768366
0.5,1,0.01,42000,286,2.73,0.005,0.005,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.513002964
0.5,1,0.015,42000,286,2.73,0.0001,0.2,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.532494915
0.5,1,0.005,42000,286,2.73,0.0002,0.5,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.518748455
0.5,1,0.01,42000,286,2.73,2.00E-05,0.2,42000,286,0.1,2000,1521,4759,10317,0,1500,0.521107542
0.5,1,0.01,42000,286,2.73,0.0001,0.5,42000,286,-0.2,4000,1521,4759,10317,0,2000,0.514001959
0.5,1,0.01,42000,286,2.73,0.0001,0.5,42000,286,-0.2,4000,1521,4759,10317,0,2500,0.52353788
0.5,1,0.01,42000,286,2.73,0.0002,0.5,42000,286,-0.2,2500,1521,4759,10317,0,1500,0.515169005
0.5,1,0.01,42000,286,2.73,0.005,0.005,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.52656863
0.5,1,0.015,42000,286,2.73,0.0001,0.1,42000,286,0,2000,1521,4759,10317,0,1500,0.527818101
0.5,1,0.015,42000,286,2.73,0.0001,1,42000,286,-0.2,2000,1521,4759,10317,0,1000,0.511348035
0.5,1,0.015,42000,286,2.73,0.001,0.005,42000,286,0,3500,1521,4759,10317,0,2500,0.503850232
0.5,1,0.015,42000,286,2.73,0.0002,0.2,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.515286419
0.5,1,0.015,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.535856086
0.5,1,0.01,42000,286,2.73,0.002,0.02,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.502048001
0.5,1,0.015,42000,286,2.73,0.001,0.5,42000,286,-0.4,3000,1521,4759,10317,0,2000,0.512718556
0.5,1,0.015,42000,286,2.73,0.0002,0.5,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.522140003
0.5,1,0.01,42000,286,2.73,5.00E-05,1,42000,286,-0.1,1500,1521,4759,10317,0,1000,0.521668383
0.5,1,0.01,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3500,1521,4759,10317,0,2500,0.534958172
0.5,1,0.015,42000,286,2.73,0.001,0.02,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.529355867
0.5,1,0.015,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.550108563
0.5,1,0.01,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.533397279
0.5,1,0.01,42000,286,2.73,0.01,0.005,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.516584108
0.5,1,0.015,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3500,1521,4759,10317,0,2500,0.537723974
0.5,1,0.003,42000,286,2.73,0.0005,1,42000,286,-0.4,3500,1521,4759,10317,0,2500,0.51444447
0.5,1,0.015,42000,286,2.73,5.00E-05,0.1,42000,286,0,2500,1521,4759,10317,0,2000,0.512251074
0.5,1,0.005,42000,286,2.73,0.001,0.5,42000,286,-0.4,3000,1521,4759,10317,0,2000,0.522368304
0.5,1,0.005,42000,286,2.73,0.0005,1,42000,286,-0.4,3500,1521,4759,10317,0,2500,0.543123849
0.5,1,0.015,42000,286,2.73,2.00E-05,1,42000,286,0,1500,1521,4759,10317,0,1000,0.508446176
0.5,1,0.015,42000,286,2.73,0.0001,0.5,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.557009629
0.5,1,0.015,42000,286,2.73,0.0001,0.2,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.521810395
0.5,1,0.01,42000,286,2.73,0.0002,0.1,42000,286,-0.1,3500,1521,4759,10317,0,2000,0.505562194
0.5,1,0.005,42000,286,2.73,0.0001,0.1,42000,286,0,2500,1521,4759,10317,0,2000,0.501734383
0.5,1,0.015,42000,286,2.73,0.002,0.001,42000,286,0.1,3000,1521,4759,10317,0,2000,0.500360271
0.5,1,0.015,42000,286,2.73,2.00E-05,0.2,42000,286,0.1,2500,1521,4759,10317,0,1500,0.502131098
0.5,1,0.015,42000,286,2.73,0.0005,0.5,42000,286,-0.4,3500,1521,4759,10317,0,2500,0.527836552
0.5,1,0.01,42000,286,2.73,0.0001,0.2,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.503495117
0.5,1,0.01,42000,286,2.73,0.0001,0.2,42000,286,-0.1,3000,1521,4759,10317,0,2500,0.518270648
0.5,1,0.015,42000,286,2.73,0.0001,0.2,42000,286,-0.1,3500,1521,4759,10317,0,2000,0.513803115
0.5,1,0.005,42000,286,2.73,0.002,0.2,42000,286,-0.4,3000,1521,4759,10317,0,2000,0.501381548
0.5,1,0.015,42000,286,2.73,0.0002,0.1,42000,286,-0.1,3000,1521,4759,10317,0,1500,0.507289336
0.5,1,0.015,42000,286,2.73,5.00E-05,0.1,42000,286,0,3000,1521,4759,10317,0,2500,0.518551702
0.5,1,0.01,42000,286,2.73,0.01,0.002,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.502440285
0.5,1,0.01,42000,286,2.73,0.01,0.002,42000,286,-0.1,3000,1521,4759,10317,0,2500,0.501607569
0.5,1,0.01,42000,286,2.73,0.0002,0.2,42000,286,-0.2,3500,1521,4759,10317,0,2500,0.518085691
0.5,1,0.015,42000,286,2.73,0.0002,0.2,42000,286,-0.2,4000,1521,4759,10317,0,2500,0.517598182
0.5,1,0.015,42000,286,2.73,0.01,0.002,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.529417366
0.5,1,0.01,42000,286,2.73,0.005,0.01,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.504361661
0.5,1,0.015,42000,286,2.73,0.0001,0.2,42000,286,-0.1,3000,1521,4759,10317,0,2500,0.513122566
0.5,1,0.01,42000,286,2.73,0.0001,0.5,42000,286,-0.2,3500,1521,4759,10317,0,3000,0.523941131
0.5,1,0.015,42000,286,2.73,0.0002,0.1,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.519730516
0.5,1,0.01,42000,286,2.73,0.005,0.005,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.512260688
0.5,1,0.015,42000,286,2.73,0.0002,0.2,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.53921513
0.5,1,0.01,42000,286,2.73,0.0005,0.1,42000,286,-0.2,3500,1521,4759,10317,0,2500,0.507722374
0.5,1,0.015,42000,286,2.73,0.0002,0.1,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.521682478
0.5,1,0.015,42000,286,2.73,0.0002,0.2,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.52424764
0.5,1,0.015,42000,286,2.73,0.0001,0.5,42000,286,-0.2,4000,1521,4759,10317,0,3000,0.5052324
0.5,1,0.015,42000,286,2.73,0.0002,0.1,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.536221206
0.5,1,0.003,42000,286,2.73,0.0005,1,42000,286,-0.4,3500,1521,4759,10317,0,3000,0.514388895
0.5,1,0.005,42000,286,2.73,0.0005,1,42000,286,-0.4,4000,1521,4759,10317,0,3000,0.516700008
0.5,1,0.015,42000,286,2.73,5.00E-05,0.2,42000,286,0,2000,1521,4759,10317,0,1500,0.533024861
0.5,1,0.01,42000,286,2.73,0.0002,0.2,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.509283673
0.5,1,0.005,42000,286,2.73,0.001,0.5,42000,286,-0.4,3000,1521,4759,10317,0,2500,0.515743192
0.5,1,0.015,42000,286,2.73,0.0005,0.1,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.517961309
0.5,1,0.01,42000,286,2.73,0.002,0.02,42000,286,-0.2,3500,1521,4759,10317,0,3000,0.503098898
0.5,1,0.015,42000,286,2.73,2.00E-05,0.1,42000,286,0.1,3000,1521,4759,10317,0,2500,0.505379859
0.5,1,0.015,42000,286,2.73,0.01,0.005,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.520711156
0.5,1,0.005,42000,286,2.73,0.002,0.02,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.504753965
0.5,1,0.015,42000,286,2.73,0.0002,0.5,42000,286,-0.2,1500,1521,4759,10317,0,1000,0.512869623
0.5,1,0.015,42000,286,2.73,0.01,0.001,42000,286,0,2000,1521,4759,10317,0,1500,0.521891369
0.5,1,0.015,42000,286,2.73,0.002,0.02,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.519695058
0.5,1,0.01,42000,286,2.73,0.0002,0.2,42000,286,-0.2,4000,1521,4759,10317,0,3000,0.506375966
0.5,1,0.015,42000,286,2.73,0.0005,0.1,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.514237932
0.5,1,0.015,42000,286,2.73,0.0002,0.2,42000,286,-0.2,3500,1521,4759,10317,0,3000,0.519754469
0.5,1,0.015,42000,286,2.73,0.002,0.02,42000,286,-0.2,3500,1521,4759,10317,0,2500,0.516607855
0.5,1,0.015,42000,286,2.73,0.01,0.005,42000,286,-0.2,3000,1521,4759,10317,0,1500,0.50006935
0.5,1,0.01,42000,286,2.73,0.005,0.1,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.509947721
0.5,1,0.01,42000,286,2.73,0.0005,0.1,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.521011084
0.5,1,0.015,42000,286,2.73,0.01,0.005,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.507774873
0.5,1,0.015,42000,286,2.73,0.01,0.005,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.521533923
0.5,1,0.01,42000,286,2.73,0.01,0.005,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.528459365
0.5,1,0.015,42000,286,2.73,0.002,0.01,42000,286,-0.1,3000,1521,4759,10317,0,1500,0.503956886
0.5,1,0.015,42000,286,2.73,0.0005,0.1,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.523320682
0.5,1,0.01,42000,286,2.73,0.0005,0.1,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.509939426
0.5,1,0.01,42000,286,2.73,0.002,0.02,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.519861172
0.5,1,0.01,42000,286,2.73,0.0005,0.1,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.521875797
0.5,1,0.01,42000,286,2.73,0.0005,0.1,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.502995847
0.5,1,0.015,42000,286,2.73,0.0005,0.005,42000,286,0.1,2500,1521,4759,10317,0,2000,0.517274267
0.5,1,0.015,42000,286,2.73,0.002,0.02,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.529065898
0.5,1,0.015,42000,286,2.73,0.001,0.2,42000,286,-0.4,3500,1521,4759,10317,0,3000,0.509851349
0.5,1,0.015,42000,286,2.73,0.0005,0.1,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.527744452
0.5,1,0.015,42000,286,2.73,0.0005,0.1,42000,286,-0.2,2500,1521,4759,10317,0,1500,0.513941347
0.5,1,0.01,42000,286,2.73,0.0002,0.1,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.531826379
0.5,1,0.01,42000,286,2.73,0.0005,1,42000,286,-0.4,4000,1521,4759,10317,0,1500,0.508556185
0.5,1,0.015,42000,286,2.73,0.0005,0.1,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.540606434
0.5,1,0.01,42000,286,2.73,0.0005,1,42000,286,-0.4,4000,1521,4759,10317,0,2000,0.532984555
0.5,1,0.015,42000,286,2.73,0.0001,0.02,42000,286,0.1,2500,1521,4759,10317,0,2000,0.512111917
0.5,1,0.015,42000,286,2.73,0.0005,1,42000,286,-0.4,3500,1521,4759,10317,0,1500,0.524747314
0.5,1,0.01,42000,286,2.73,0.001,0.02,42000,286,-0.1,3500,1521,4759,10317,0,2000,0.500626034
0.5,1,0.01,42000,286,2.73,0.0005,1,42000,286,-0.4,4000,1521,4759,10317,0,2500,0.52444957
0.5,1,0.01,42000,286,2.73,0.0005,0.05,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.50017946
0.5,1,0.015,42000,286,2.73,0.0005,1,42000,286,-0.4,3500,1521,4759,10317,0,2000,0.54282929
0.5,1,0.01,42000,286,2.73,0.0002,0.1,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.518981777
0.5,1,0.015,42000,286,2.73,0.0005,1,42000,286,-0.4,3500,1521,4759,10317,0,2500,0.528571556
0.5,1,0.015,42000,286,2.73,0.0005,1,42000,286,-0.4,3000,1521,4759,10317,0,1500,0.53674897
0.5,1,0.015,42000,286,2.73,0.001,0.02,42000,286,-0.1,3500,1521,4759,10317,0,2000,0.500175017
0.5,1,0.01,42000,286,2.73,0.0005,1,42000,286,-0.4,3000,1521,4759,10317,0,2000,0.558660677
0.5,1,0.01,42000,286,2.73,0.0005,1,42000,286,-0.4,3000,1521,4759,10317,0,2500,0.552538238
0.5,1,0.01,42000,286,2.73,0.0002,0.1,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.512885306
0.5,1,0.01,42000,286,2.73,0.0001,0.1,42000,286,0,2500,1521,4759,10317,0,2000,0.506844468
0.5,1,0.01,42000,286,2.73,0.0005,1,42000,286,-0.4,2500,1521,4759,10317,0,1500,0.539784254
0.5,1,0.015,42000,286,2.73,0.0005,1,42000,286,-0.4,2500,1521,4759,10317,0,2000,0.569951272
0.5,1,0.005,42000,286,2.73,0.001,0.1,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.512688988
0.5,1,0.015,42000,286,2.73,0.0002,0.1,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.531139398
0.5,1,0.015,42000,286,2.73,0.0005,1,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.557484251
0.5,1,0.015,42000,286,2.73,0.002,0.1,42000,286,-0.4,3500,1521,4759,10317,0,3000,0.505671418
0.5,1,0.01,42000,286,2.73,0.001,0.5,42000,286,-0.4,3000,1521,4759,10317,0,1500,0.510623436
0.5,1,0.01,42000,286,2.73,0.001,0.5,42000,286,-0.4,3000,1521,4759,10317,0,2000,0.5211009
0.5,1,0.015,42000,286,2.73,0.001,0.5,42000,286,-0.4,2500,1521,4759,10317,0,1500,0.52486876
0.5,1,0.01,42000,286,2.73,0.0005,0.5,42000,286,-0.4,4000,1521,4759,10317,0,3000,0.524098714
0.5,1,0.01,42000,286,2.73,0.001,0.5,42000,286,-0.4,2500,1521,4759,10317,0,2000,0.536173855
0.5,1,0.015,42000,286,2.73,0.0005,0.5,42000,286,-0.4,4000,1521,4759,10317,0,2500,0.521692058
0.5,1,0.01,42000,286,2.73,0.01,0.002,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.526891038
0.5,1,0.01,42000,286,2.73,0.001,0.5,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.535600056
0.5,1,0.015,42000,286,2.73,0.0005,0.5,42000,286,-0.4,3500,1521,4759,10317,0,3000,0.541944285
0.5,1,0.01,42000,286,2.73,0.0005,0.5,42000,286,-0.4,3500,1521,4759,10317,0,2500,0.509817113
0.5,1,0.01,42000,286,2.73,0.0005,0.5,42000,286,-0.4,3000,1521,4759,10317,0,2500,0.513808232
0.5,1,0.01,42000,286,2.73,0.0001,0.1,42000,286,0,2000,1521,4759,10317,0,1500,0.526488725
0.5,1,0.01,42000,286,2.73,0.002,0.2,42000,286,-0.4,3000,1521,4759,10317,0,2000,0.514042905
0.5,1,0.015,42000,286,2.73,0.002,0.002,42000,286,0.1,2000,1521,4759,10317,0,1500,0.508915015
0.5,1,0.015,42000,286,2.73,0.002,0.2,42000,286,-0.4,2500,1521,4759,10317,0,1500,0.507929435
0.5,1,0.015,42000,286,2.73,0.002,0.2,42000,286,-0.4,2500,1521,4759,10317,0,2000,0.526120643
0.5,1,0.01,42000,286,2.73,0.002,0.2,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.509634452
0.5,1,0.01,42000,286,2.73,0.005,0.05,42000,286,-0.4,3500,1521,4759,10317,0,3000,0.503007278
0.5,1,0.015,42000,286,2.73,0.005,0.05,42000,286,-0.4,3000,1521,4759,10317,0,2500,0.516143373
0.5,1,0.015,42000,286,2.73,0.0001,0.1,42000,286,-0.1,3500,1521,4759,10317,0,3000,0.502195899
0.5,1,0.015,42000,286,2.73,0.01,0.001,42000,286,-0.1,3500,1521,4759,10317,0,3000,0.501001823
0.5,1,0.015,42000,286,2.73,0.0001,0.05,42000,286,0,3000,1521,4759,10317,0,2000,0.503692409
0.5,1,0.015,42000,286,2.73,0.0001,0.05,42000,286,0,3000,1521,4759,10317,0,2500,0.517095719
0.5,1,0.015,42000,286,2.73,0.0001,0.05,42000,286,0,3500,1521,4759,10317,0,2500,0.505667899
0.5,1,0.01,42000,286,2.73,0.0005,0.005,42000,286,0.1,2500,1521,4759,10317,0,2000,0.507788066
0.5,1,0.015,42000,286,2.73,0.0005,0.005,42000,286,0.1,3000,1521,4759,10317,0,2000,0.502434674
0.5,1,0.015,42000,286,2.73,0.0005,0.005,42000,286,0.1,2500,1521,4759,10317,0,1500,0.50032865
0.5,1,0.01,42000,286,2.73,0.0001,0.1,42000,286,0,2500,1521,4759,10317,0,1500,0.511243889
0.5,1,0.015,42000,286,2.73,5.00E-05,0.05,42000,286,0.1,2500,1521,4759,10317,0,2000,0.519145362
0.5,1,0.015,42000,286,2.73,5.00E-05,0.05,42000,286,0.1,2000,1521,4759,10317,0,1500,0.512029989
0.5,1,0.01,42000,286,2.73,0.01,0.001,42000,286,0,2500,1521,4759,10317,0,1500,0.506485392
0.5,1,0.015,42000,286,2.73,0.0001,0.1,42000,286,0,2500,1521,4759,10317,0,1500,0.51056503
0.5,1,0.01,42000,286,2.73,0.001,0.05,42000,286,-0.2,3500,1521,4759,10317,0,2500,0.503062632
0.5,1,0.01,42000,286,2.73,0.0002,0.05,42000,286,0,2000,1521,4759,10317,0,1500,0.524504012
0.5,1,0.015,42000,286,2.73,0.01,0.001,42000,286,0,2500,1521,4759,10317,0,1500,0.504006534
0.5,1,0.015,42000,286,2.73,0.0002,0.05,42000,286,0,2000,1521,4759,10317,0,1500,0.52492001
0.5,1,0.015,42000,286,2.73,0.0001,0.05,42000,286,0,2500,1521,4759,10317,0,2000,0.512853734
0.5,1,0.005,42000,286,2.73,0.005,0.005,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.505873711
0.5,1,0.01,42000,286,2.73,0.001,0.05,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.519215465
0.5,1,0.01,42000,286,2.73,0.001,0.05,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.516652375
0.5,1,0.01,42000,286,2.73,0.0005,0.05,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.514484379
0.5,1,0.015,42000,286,2.73,0.001,0.05,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.510331493
0.5,1,0.01,42000,286,2.73,0.0005,0.05,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.5169045
0.5,1,0.015,42000,286,2.73,0.001,0.05,42000,286,-0.2,3000,1521,4759,10317,0,1500,0.50173502
0.5,1,0.01,42000,286,2.73,0.001,0.05,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.530781083
0.5,1,0.01,42000,286,2.73,0.0005,0.05,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.528478838
0.5,1,0.015,42000,286,2.73,0.001,0.05,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.524163478
0.5,1,0.015,42000,286,2.73,0.001,0.05,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.512721953
0.5,1,0.015,42000,286,2.73,0.0005,0.05,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.516370871
0.5,1,0.01,42000,286,2.73,0.001,0.05,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.503146465
0.5,1,0.015,42000,286,2.73,0.001,0.05,42000,286,-0.2,2500,1521,4759,10317,0,1500,0.512720104
0.5,1,0.015,42000,286,2.73,0.0005,0.05,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.507512042
0.5,1,0.015,42000,286,2.73,0.0005,0.05,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.532329246
0.5,1,0.015,42000,286,2.73,0.001,0.05,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.522498212
0.5,1,0.01,42000,286,2.73,0.001,0.05,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.506952111
0.5,1,0.01,42000,286,2.73,0.01,0.05,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.505770457
0.5,1,0.015,42000,286,2.73,0.001,0.05,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.537366791
0.5,1,0.015,42000,286,2.73,0.005,0.1,42000,286,-0.4,2000,1521,4759,10317,0,1500,0.507613256
0.5,1,0.005,42000,286,2.73,0.002,0.05,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.509144091
0.5,1,0.01,42000,286,2.73,0.0002,0.05,42000,286,0,2500,1521,4759,10317,0,2000,0.502748887
0.5,1,0.015,42000,286,2.73,0.0005,0.02,42000,286,-0.1,3500,1521,4759,10317,0,3000,0.501336857
0.5,1,0.015,42000,286,2.73,0.0001,0.02,42000,286,0.1,3000,1521,4759,10317,0,2000,0.500577582
0.5,1,0.015,42000,286,2.73,0.002,0.001,42000,286,0.1,3000,1521,4759,10317,0,2500,0.501368245
0.5,1,0.015,42000,286,2.73,0.002,0.001,42000,286,0.1,2500,1521,4759,10317,0,2000,0.512005417
0.5,1,0.01,42000,286,2.73,0.0005,0.02,42000,286,0,2500,1521,4759,10317,0,1500,0.507460919
0.5,1,0.01,42000,286,2.73,0.0005,0.02,42000,286,0,2000,1521,4759,10317,0,1500,0.523215399
0.5,1,0.015,42000,286,2.73,0.0005,0.02,42000,286,0,2500,1521,4759,10317,0,1500,0.505324697
0.5,1,0.015,42000,286,2.73,0.0005,0.02,42000,286,0,2000,1521,4759,10317,0,1500,0.523087752
0.5,1,0.01,42000,286,2.73,0.002,0.002,42000,286,0.1,2000,1521,4759,10317,0,1500,0.513319117
0.5,1,0.01,42000,286,2.73,0.01,0.002,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.513431012
0.5,1,0.01,42000,286,2.73,0.001,0.02,42000,286,-0.1,3000,1521,4759,10317,0,2500,0.503293639
0.5,1,0.01,42000,286,2.73,0.001,0.02,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.502699238
0.5,1,0.01,42000,286,2.73,0.001,0.02,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.527871935
0.5,1,0.015,42000,286,2.73,0.0002,0.02,42000,286,0.1,2000,1521,4759,10317,0,1500,0.510071729
0.5,1,0.01,42000,286,2.73,0.01,0.002,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.512590236
0.5,1,0.015,42000,286,2.73,0.001,0.02,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.515699346
0.5,1,0.015,42000,286,2.73,0.001,0.02,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.517362556
0.5,1,0.015,42000,286,2.73,0.001,0.02,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.530738829
0.5,1,0.015,42000,286,2.73,0.01,0.002,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.528868642
0.5,1,0.015,42000,286,2.73,0.001,0.01,42000,286,-0.1,3500,1521,4759,10317,0,3000,0.501166887
0.5,1,0.015,42000,286,2.73,0.0002,0.01,42000,286,0.1,3000,1521,4759,10317,0,2000,0.50046806
0.5,1,0.015,42000,286,2.73,0.0002,0.01,42000,286,0.1,3000,1521,4759,10317,0,2500,0.501777792
0.5,1,0.015,42000,286,2.73,0.0002,0.01,42000,286,0.1,2500,1521,4759,10317,0,2000,0.512060688
0.5,1,0.015,42000,286,2.73,0.0005,0.01,42000,286,0,3000,1521,4759,10317,0,2000,0.503712407
0.5,1,0.015,42000,286,2.73,0.0005,0.01,42000,286,0,3000,1521,4759,10317,0,2500,0.515716966
0.5,1,0.015,42000,286,2.73,0.005,0.001,42000,286,0,2500,1521,4759,10317,0,2000,0.513166403
0.5,1,0.015,42000,286,2.73,0.001,0.02,42000,286,-0.1,3000,1521,4759,10317,0,1500,0.504366647
0.5,1,0.01,42000,286,2.73,0.001,0.01,42000,286,0,2500,1521,4759,10317,0,1500,0.506951161
0.5,1,0.01,42000,286,2.73,0.001,0.01,42000,286,0,2000,1521,4759,10317,0,1500,0.522769637
0.5,1,0.015,42000,286,2.73,0.001,0.01,42000,286,0,2500,1521,4759,10317,0,1500,0.504634426
0.5,1,0.015,42000,286,2.73,0.001,0.01,42000,286,0,2000,1521,4759,10317,0,1500,0.522461542
0.5,1,0.005,42000,286,2.73,0.005,0.02,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.506934939
0.5,1,0.01,42000,286,2.73,0.002,0.01,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.513921519
0.5,1,0.01,42000,286,2.73,0.002,0.01,42000,286,-0.1,3000,1521,4759,10317,0,2500,0.502361158
0.5,1,0.01,42000,286,2.73,0.002,0.01,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.502559421
0.5,1,0.015,42000,286,2.73,0.01,0.002,42000,286,-0.1,3000,1521,4759,10317,0,1500,0.503621985
0.5,1,0.01,42000,286,2.73,0.002,0.01,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.527331115
0.5,1,0.01,42000,286,2.73,0.002,0.01,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.512650504
0.5,1,0.015,42000,286,2.73,0.002,0.01,42000,286,-0.1,3000,1521,4759,10317,0,2000,0.514906349
0.5,1,0.015,42000,286,2.73,0.002,0.01,42000,286,-0.1,2500,1521,4759,10317,0,1500,0.517022444
0.5,1,0.015,42000,286,2.73,0.002,0.01,42000,286,-0.1,2500,1521,4759,10317,0,2000,0.530008659
0.5,1,0.015,42000,286,2.73,0.002,0.01,42000,286,-0.1,2000,1521,4759,10317,0,1500,0.529089073
0.5,1,0.015,42000,286,2.73,0.0005,0.01,42000,286,0,3500,1521,4759,10317,0,2500,0.504064237
0.5,1,0.01,42000,286,2.73,0.01,0.005,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.504025345
0.5,1,0.01,42000,286,2.73,0.005,0.01,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.516887556
0.5,1,0.01,42000,286,2.73,0.005,0.01,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.512963064
0.5,1,0.015,42000,286,2.73,0.005,0.01,42000,286,-0.2,3500,1521,4759,10317,0,2000,0.507012126
0.5,1,0.015,42000,286,2.73,0.002,0.005,42000,286,-0.1,3500,1521,4759,10317,0,3000,0.5010766
0.5,1,0.015,42000,286,2.73,0.005,0.01,42000,286,-0.2,3000,1521,4759,10317,0,1500,0.500265024
0.5,1,0.01,42000,286,2.73,0.005,0.01,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.528728378
0.5,1,0.015,42000,286,2.73,0.005,0.01,42000,286,-0.2,3000,1521,4759,10317,0,2000,0.521105438
0.5,1,0.015,42000,286,2.73,0.005,0.01,42000,286,-0.2,3000,1521,4759,10317,0,2500,0.508335317
0.5,1,0.01,42000,286,2.73,0.005,0.01,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.503072852
0.5,1,0.015,42000,286,2.73,0.005,0.01,42000,286,-0.2,2500,1521,4759,10317,0,1500,0.511553276
0.5,1,0.015,42000,286,2.73,0.01,0.005,42000,286,-0.2,2500,1521,4759,10317,0,2000,0.534222758
0.5,1,0.015,42000,286,2.73,0.005,0.01,42000,286,-0.2,2000,1521,4759,10317,0,1500,0.521651531
0.5,1,0.015,42000,286,2.73,0.005,0.001,42000,286,0,3500,1521,4759,10317,0,2500,0.503676849
0.5,1,0.015,42000,286,2.73,0.001,0.005,42000,286,0,3000,1521,4759,10317,0,2000,0.503701837
0.5,1,0.015,42000,286,2.73,0.001,0.005,42000,286,0,3000,1521,4759,10317,0,2500,0.515531119
0.5,1,0.015,42000,286,2.73,0.001,0.005,42000,286,0,2500,1521,4759,10317,0,2000,0.513151325
0.5,1,0.01,42000,286,2.73,0.002,0.005,42000,286,0,2500,1521,4759,10317,0,1500,0.506693199
0.5,1,0.01,42000,286,2.73,0.002,0.005,42000,286,0,2000,1521,4759,10317,0,1500,0.522543711
0.5,1,0.015,42000,286,2.73,0.002,0.005,42000,286,0,2500,1521,4759,10317,0,1500,0.504286364
0.5,1,0.015,42000,286,2.73,0.002,0.005,42000,286,0,2000,1521,4759,10317,0,1500,0.522145541
0.5,1,0.015,42000,286,2.73,0.005,0.002,42000,286,-0.1,3500,1521,4759,10317,0,3000,0.501020722
0.5,1,0.015,42000,286,2.73,0.001,0.002,42000,286,0.1,3000,1521,4759,10317,0,2000,0.500372678
0.5,1,0.015,42000,286,2.73,0.001,0.002,42000,286,0.1,3000,1521,4759,10317,0,2500,0.501414213
0.5,1,0.015,42000,286,2.73,0.001,0.002,42000,286,0.1,2500,1521,4759,10317,0,2000,0.512011999
0.5,1,0.01,42000,286,2.73,0.005,0.002,42000,286,0,2500,1521,4759,10317,0,1500,0.506537478
0.5,1,0.01,42000,286,2.73,0.005,0.002,42000,286,0,2000,1521,4759,10317,0,1500,0.522407207
0.5,1,0.015,42000,286,2.73,0.005,0.002,42000,286,0,2500,1521,4759,10317,0,1500,0.504076609
0.5,1,0.015,42000,286,2.73,0.005,0.002,42000,286,0,2000,1521,4759,10317,0,1500,0.521955017
"""

if __name__ == "__main__":
    main()
