# from numba import njit
import numpy as np
from numpy.random import default_rng
rng = default_rng()



def Pcentroid_Rsampling(P, Rname, Rdir):
    import xarray as xr
    import rioxarray as rio
    r = rio.open_rasterio(f"{Rdir}/{Rname}")
    x_indexer = xr.DataArray(P.geometry.centroid.x, dims=["point"])
    y_indexer = xr.DataArray(P.geometry.centroid.y, dims=["point"])
	# extract raster value at points
    Rvalues = r.sel(x=x_indexer, y=y_indexer, method="nearest")
    P['pgas'] = Rvalues.values.flatten()
    return P



def join_Pcentroid_at_Pl(Pc, Pccols, Pl, Plcols):
    import geopandas as gpd
    c = gpd.GeoDataFrame(Pc, crs=Pc.crs, geometry=Pc.geometry.centroid)
    j = gpd.sjoin(c[Pccols], Pl[Plcols], how="left")
    j = gpd.GeoDataFrame(j, geometry=Pc.geometry, crs=Pc.crs)
    return j


# def join_Pcentroid_at_Polygons_all_columns(Pc, Pl):
# 	import geopandas as gpd
# 	c = gpd.GeoDataFrame(Pc, crs=Pc.crs, geometry=Pc.geometry.centroid)
# 	j = gpd.sjoin(c, Pl, how="left")
# 	j = gpd.GeoDataFrame(j, geometry=Pc.geometry, crs=Pc.crs)
# 	return j
#
#
#
# @njit()
# def mh_cascade_scenario(EXP_IDS, PPGA_SU, SI_SU, FLOWR_MEAN, FLOWR_STD):
#
#     a =  np.zeros(EXP_IDS.shape[0], np.float64)
#
#     for i,_ in enumerate(EXP_IDS):
#         rng = np.random.uniform(0, 1)
#         if PPGA_SU[i] > rng:
#             rng = np.random.uniform(0, 1)
#             if np.random.normal(SI_SU[i], SI_SU[i]*0.1) > rng:
#                 rng = np.random.uniform(0, 1)
#                 if np.random.normal(FLOWR_MEAN[i], FLOWR_STD[i]) > rng:
#                     a[i] = 1
#     return a


