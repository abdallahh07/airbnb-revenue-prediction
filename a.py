import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. loading the data 
listing = pd.read_csv("listings.csv", on_bad_lines="skip", low_memory=False)
past_rates=pd.read_csv("past_rates.csv",on_bad_lines="skip",low_memory=False)

# 2. adusting the data to analyze it  
rates=past_rates.copy()
rates["Date"]=pd.to_datetime(past_rates["date"])
rates["Year"]=rates["Date"].dt.year
rates["Month"]=rates["Date"].dt.month


# 3. analyze the past rates
def analyze(rates):
  fig,ax =plt.subplots(3,3,figsize=(18,15))
  ax=ax.flatten()
  
  # 3.1 monthly revenue distribution 
  monthly_rev= rates.groupby("Month")["revenue"].mean()
  sns.barplot(x=monthly_rev.index,
               y=monthly_rev.values,
               ax=ax[0],
               palette="magma")
  ax[0].set_title("Monthly revenue distribution")
  
  # 3.2 Revenue by largest 15 City
  cit_rev = rates.groupby("city")["revenue"].sum().sort_values(ascending=False).head(15)
  sns.barplot(
    x = cit_rev.values,
    y = cit_rev.index,
    ax = ax[1], 
    palette = "magma")
  ax[1].set_title("Top 15 Cities by Revenue")
  
  # 3.3 Revenue by Largest 15 country
  count_rev = rates.groupby("country")["revenue"].sum().sort_values(ascending=False).head(15)
  sns.barplot(
    x=count_rev.values,
    y=count_rev.index,
    palette="magma",
    ax=ax[2])
  ax[2].set_title("Top 15 Country by Revenue")
  
  # 3.4 Occupancy distribution
  monthly_occ = rates.groupby("Month")["occupancy"].mean().reset_index()
  sns.lineplot(data=monthly_occ,
              x="Month",
              y="occupancy",
              ax=ax[3],
              palette="magma")
  ax[3].set_title("Average Occupancy by Month")
  
  # 3.5 Occupancy by country 
  country_occ = rates.groupby("country")["revenue"].sum().sort_values(ascending=False).head(15)
  sns.barplot(
    x = country_occ.values,
    y = country_occ.index,
    ax = ax[4], 
    palette = "magma")
  ax[4].set_title("Occupancy by country")
  
  # 3.6 Monthly reserved days
  monthly_reserved = rates.groupby("Month")["reserved_days"].mean()
  sns.barplot(x=monthly_reserved.index,
                y=monthly_reserved.values,
                ax=ax[5],
                palette="magma")
  ax[5].set_title("Average Reserved Days by Month")
  ax[5].set_xlabel("Month")
  
  # 3.7 Monthly booking lead time
  monthly_lead = rates.groupby("Month")["booking_lead_time_avg"].mean()
  sns.lineplot(x=monthly_lead.index,
                 y=monthly_lead.values,
                 ax=ax[6],
                 palette="magma")
  ax[6].set_title("Booking Lead Time by Month")
  ax[6].set_xlabel("Month")
  
  # 3.8 Monthly length of stay
  monthly_stay = rates.groupby("Month")["length_of_stay_avg"].mean()
  sns.lineplot(x=monthly_stay.index,
                 y=monthly_stay.values,
                 ax=ax[7])
  ax[7].set_title("Length of Stay by Month")
  ax[7].set_xlabel("Month")
  
  sns.histplot(data=rates,
              x="revenue",
              bins=50,
              kde=True,
              ax=ax[8])
  ax[8].set_title("Revenue Distribution")
  
  plt.tight_layout()
  plt.show()
analyze(rates)

# 4. analyzing the listing 
cols_to_drop = [
    "listing_id", "host_id", "cover_photo_url",
    "ttm_revenue_native", "ttm_avg_rate_native",
    "ttm_revpar_native", "ttm_adjusted_revpar_native",
    "ttm_total_days", "l90d_revenue_native",
    "l90d_avg_rate_native", "l90d_revpar_native",
    "l90d_adjusted_revpar_native", "l90d_total_days"
]
f = listing.drop(columns=cols_to_drop, errors='ignore')
valid_rooms = ["entire_home", "private_room", "shared_room", "hotel_room"]
f=f[f["room_type"].isin(valid_rooms)]

def analyze_listing(f):
    fig, ax = plt.subplots(3, 3, figsize=(18, 15))
    ax = ax.flatten()
    
    # 1. Revenue distribution
    sns.histplot(data=f, x="ttm_revenue", bins=50, kde=True, ax=ax[0], color="purple")
    ax[0].set_title("Revenue Distribution")
    ax[0].set_xlabel("TTM Revenue")
    
    # 2. Revenue by room type
    room_rev = f.groupby("room_type")["ttm_revenue"].mean().sort_values(ascending=True)
    sns.barplot(x=room_rev.values, y=room_rev.index, ax=ax[1], palette="magma")
    ax[1].set_title("Revenue by Room Type")
    ax[1].set_xlabel("Average Revenue")
    
    # 3. Revenue by country top 15
    country_rev = f.groupby("country")["ttm_revenue"].mean().sort_values(ascending=True).head(15)
    sns.barplot(x=country_rev.values, y=country_rev.index, ax=ax[2], palette="magma")
    ax[2].set_title("Revenue by Country")
    ax[2].set_xlabel("Average Revenue")
    
    # 4. Occupancy distribution
    sns.histplot(data=f, x="ttm_occupancy", bins=50, kde=True, ax=ax[3], color="purple")
    ax[3].set_title("Occupancy Distribution")
    ax[3].set_xlabel("Occupancy Rate")
    
    # 5. Rating vs Revenue
    sns.scatterplot(data=f, x="rating_overall", y="ttm_revenue", ax=ax[4], alpha=0.3, color="purple")
    ax[4].set_title("Rating vs Revenue")
    ax[4].set_xlabel("Overall Rating")
    ax[4].set_ylabel("Revenue")
    
    # 6. Bedrooms vs Revenue
    bed_rev = f.groupby("bedrooms")["ttm_revenue"].mean().sort_values(ascending=True).head(10)
    sns.barplot(x=bed_rev.values, y=bed_rev.index.astype(str), ax=ax[5], palette="magma")
    ax[5].set_title("Revenue by Bedrooms")
    ax[5].set_xlabel("Average Revenue")
    ax[5].set_ylabel("Bedrooms")
    
    # 7. Superhost vs Revenue
    super_rev = f.groupby("superhost")["ttm_revenue"].mean()
    sns.barplot(x=super_rev.index.astype(str), y=super_rev.values, ax=ax[6], palette="magma")
    ax[6].set_title("Superhost vs Revenue")
    ax[6].set_xlabel("Is Superhost")
    ax[6].set_ylabel("Average Revenue")
    
    # 8. Revenue by number of guests capacity
    guest_rev = f.groupby("guests")["ttm_revenue"].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=guest_rev.index.astype(str),
            y=guest_rev.values,
            ax=ax[7],
            palette="magma",
            order=guest_rev.index.astype(str))
    ax[7].set_title("Revenue by Guest Capacity")
    ax[7].set_xlabel("Number of Guests")
    ax[7].set_ylabel("Average Revenue")
    
    # 9. Correlation heatmap
    numeric_cols = f.select_dtypes(include='number')
    sns.heatmap(numeric_cols.corr(), annot=False, cmap="magma", ax=ax[8])
    ax[8].set_title("Correlation Heatmap")
    
    plt.tight_layout()
    plt.show()

analyze_listing(f)

# 5. Clean Listing data
print (f.isna().sum())
  # 5.1 missing columns
N_col= ["photos_count","num_reviews","guests","bedrooms","beds","baths","min_nights"
        ,"cleaning_fee","extra_guest_fee","rating_overall"
        ,"rating_accuracy","rating_checkin","rating_cleanliness"
        ,"rating_communication","rating_location","rating_value"]
bol_col=["registration","professional_management","instant_book"]
text_cols=["amenities","cancellation_policy"]
  # 5.2 number columns imputer
for col in N_col:
  f[col]=pd.to_numeric(f[col],errors="coerce")
num_imputer=SimpleImputer(strategy="mean")
f[N_col]=num_imputer.fit_transform(f[N_col])
  # 5.3 boolean columns imputer
for col in bol_col:
  f[col] = f[col].map({"true":1,"false":0,
                  True:1,False:0})
bool_imputer = SimpleImputer(strategy="constant",fill_value=0)
f[bol_col]=bool_imputer.fit_transform(f[bol_col])
  # 5.4 text columns imputer
text_imputer = SimpleImputer(strategy="most_frequent")
f[text_cols] = text_imputer.fit_transform(f[text_cols])

print(f.isna().sum())

# 6. Feature Engeneering 
  # 6.1 wifi 
f["wifi"]=f["amenities"].apply(lambda x:1 if "wifi" in str(x) else 0)

  # 6.2 Kitchen
f["Kitchen"]=f["amenities"].apply(lambda x:1 if "Kitchen" in str(x) else 0)

# 6.3 Property size score
f["size_score"] = f["bedrooms"] + f["baths"] + f["beds"]

# 6.4 Amenity count
f["amenity_count"] = f["amenities"].apply(lambda x: len(str(x).split(",")) if pd.notna(x) else 0)

# 7. splitting the data
drop_cols = [col for col in f.columns if col.startswith("ttm_") and col != "ttm_revenue"]
drop_cols += [col for col in f.columns if col.startswith("l90d_")]
f = f.drop(drop_cols,axis=1)
x = f.drop("ttm_revenue",axis=1)
y = f["ttm_revenue"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# 8. Preprocessing the data
one_hot = OneHotEncoder(handle_unknown="ignore")
categorical= x.select_dtypes(include="object").columns.tolist()
transformer=ColumnTransformer([("one_hot",
                                one_hot,
                                categorical,)],
                              remainder="passthrough")
x_train = transformer.fit_transform(x_train)
x_test = transformer.transform(x_test)

# 9. the model
models = {
    "CatBoostRegressor": CatBoostRegressor(verbose=0),
    "XGBRegressor": XGBRegressor(n_jobs=-1),
    "LinearRegression": LinearRegression(n_jobs=-1),
    "Ridge": Ridge()
}

# 10. Evaluate the model
results = []
name = None
best_pipe = None
best_model = None
r2_ = -999
meansquarederror = 0
rootmeansquarederror = 0 

for model,m in models.items():
  pipe= Pipeline([("scaler",StandardScaler(with_mean=False)),
                  ("poly",PolynomialFeatures(degree=1)),
                  ("Model",m),])
  pipe.fit(x_train,y_train)
  y_pred =pipe.predict(x_test)
  
  r2 = r2_score(y_test,y_pred)
  mean = mean_squared_error(y_test,y_pred)
  root = np.sqrt(mean)
  
  print(f"------{model}------")
  print("R2 Score",r2)
  print("Mean_Squared_Error",mean)
  print("Root_Mean_Squared_Error",root)
  
  results.append({
    "Name":model,
    "Model":m,
    "R2":r2,
    "Mean_Squared_Error":mean,
    "Root_Mean_Squared_Error":root})
  
  if r2 > r2_ :
    name = model
    best_pipe = pipe
    best_model = m
    r2_= r2
    meansquarederror = mean
    rootmeansquarederror = root
  
print("---------------")   
print("The best Model is:",best_model)
print("R2:",r2_)
print("Mean_Squared_Error:",meansquarederror)
print("Root_Mean_Squared_Error:",rootmeansquarederror)

# 11. analyze the model
y_pred_best = best_pipe.predict(x_test)

  # 11.1 Actual vs predicted
plt.scatter(y_test, y_pred_best, alpha=0.3)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted")
plt.show()

  # 11.2 Residuals
residuals = y_test - y_pred_best
plt.hist(residuals, bins=50)
plt.title("Residuals Distribution")
plt.show()

  # 11.3 Result table
results_df = pd.DataFrame(results)
print(results_df[["Name", "R2", "Mean_Squared_Error", "Root_Mean_Squared_Error"]].sort_values("R2", ascending=False))
