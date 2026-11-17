#%%
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from pathlib import Path
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import norm, expon, lognorm

from mods_geom_ops import Pcentroid_Rsampling

# Set plotting style
plt.style.use(['science', 'no-latex'])

# Load data paths and datasets
current_dir = Path(__file__).parent.resolve()
print(os.getcwd())

#%%
datadir = current_dir / 'data/shp'
rasdir = current_dir / 'data/tif/test_gorkha'  # Gorkha testing
resdir = current_dir /  'data/results/test_gorkha/xgboost'

print('list of shapefiles')
print(os.listdir(datadir))

list_rasters = [file for file in os.listdir(rasdir) if file.lower().endswith(".tif")]
print(f'this is the list of rasters : \n{list_rasters}')


#%%----------------------------------------------------------------

# extract PGA value at slope units centroid
slp = gpd.read_file(f"{datadir}/su_updt.shp")
list_rasters = [file for file in os.listdir(rasdir) if file.lower().endswith(".tif")]
slp = Pcentroid_Rsampling(slp, list_rasters[0], rasdir)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
slp.plot("pgas", cmap='viridis', legend=True, 
         legend_kwds={"label": "PGAS", "orientation": "horizontal"}, 
         ax=ax)

# attach the number of E4 landslides
e4 = gpd.read_file(f"{datadir}/E4/Epoch04_FINAL_TopologyFixed.shp")
e4points = gpd.GeoDataFrame(e4[['E4', 'id']], geometry=e4.geometry.centroid)
join = gpd.sjoin(slp, e4points, how="left")

# Group by 'su_id', sum 'E4', and keep the maximum 'PGA'
grouped_data = join.groupby('su_id').agg({'E4': 'sum', 'pgas': 'max'}).reset_index()

# Rename the aggregated columns if desired
grouped_data.rename(columns={'E4': 'E4_sum', 'pgas': 'max_PGA'}, inplace=True)

# Extract the 'geometry' column from the existing GeoDataFrame
grouped_data = grouped_data.merge(slp, on="su_id", how="left")
gdf = gpd.GeoDataFrame(grouped_data, geometry=grouped_data['geometry'])


#%% plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
gdf.plot("E4_sum", cmap='viridis', legend=True, 
         legend_kwds={"label": "E4 sum", "orientation": "horizontal"},
         ax=ax)
plt.show()

#%%
data = gdf[['pgas', 'E4_sum']].copy()

# Split the data into input features (X) and the target variable (y)
X = data[['pgas']]
y = np.where(data['E4_sum']>0, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)


#%% Create the histogram
# Extract the relevant data
pgas_data = data.pgas[data.E4_sum > 0]
# plt.hist(pgas_data, bins='auto', density=True, alpha=0.7)

# plt.savefig(resdir /'FIG_PGAs_whereE4_histogram.png', dpi=300)
# plt.close()
# plt.show()

#%% Fit the CDF
mu, std = norm.fit(pgas_data)
shape, loc, scale = lognorm.fit(pgas_data, floc=0)
loc_exp, scale_exp = expon.fit(pgas_data)

x = np.linspace(pgas_data.min(), pgas_data.max(), 100)

cdf = norm.cdf(x, mu, std)
cdf_log = lognorm.cdf(x, shape, loc=loc, scale=scale)
cdf_expo = expon.cdf(x, loc=loc_exp, scale=scale_exp)

# Plot the CDF fit
plt.plot(x, cdf, color='red', linestyle='-', label='CDT Normal fit')
# plt.plot(x, cdf_log, 'b-', label='CDF_log Fit')
# plt.plot(x, cdf_expo, 'g-', label='CDF_expo Fit')

plt.scatter(data.pgas, data.E4_sum, c='none', edgecolors='black', marker='o')
plt.hist(pgas_data, bins=x, density=True, alpha=0.7, label='Distribution of PGAs \ninduced landslide')

# # Plot test sampling
# plt.scatter(x, norm(mu, scale=std).cdf(x), label='Sample', c="b", s=10)
# plt.scatter(x, lognorm.cdf(x, shape, loc=loc, scale=scale), label='Sample', c="g", s=10)
# plt.scatter(x, expon.cdf(x, loc=loc_exp, scale=scale_exp), label='Sample', c="r", s=10)

plt.ylabel('Probability')
plt.xlabel('PGA(g)')
plt.ylim(0, 1)

# add legend with bbox
plt.legend(loc='upper left', framealpha=1, frameon=True, edgecolor='black', facecolor='white')
# for text in legend.get_texts():
#     text.set_color("black")

plt.savefig(resdir /'FIG_Probability_PGAs.png', dpi=300)
plt.close()
# plt.show()

# Print the fit parameters
print("Mean:", mu)
print("Standard Deviation:", std)


# %%
from sklearn.metrics import roc_curve, roc_auc_score

# Predict on the test set
# Get predicted probabilities for the positive class
y_prob = logistic_model.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, lw=2, label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label='Random case')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
plt.legend(loc='lower right')

plt.savefig(resdir /'FIG_ROCcurveLogisticRegression.png', dpi=300)
plt.close()
# plt.show()