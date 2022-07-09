##############################################
# CLTV Prediction with BG-NBD and Gamma-Gamma
##############################################

###################
# Business Problem
###################
"""
FLO wants to set a roadmap for sales and marketing activities.
In the medium-long-term plan of the company, it is necessary to be able to predict
the potential value that existing customers will bring to the company in the future.
"""

###################
# Features
###################

# Total Features   : 12
# Total Row        : 19.945
# csv File Size    : 2.7MB

"""
- master_id                         : Unique Customer Number
- order_channel                     : Which channel of the shopping platform is used 
                                      (Android, IOS, Desktop, Mobile)
- last_order_channel                : The channel where the most recent purchase was made
- first_order_date                  : Date of the customer's first purchase
- last_order_channel                : Customer's previous shopping history
- last_order_date_offline           : The date of the last purchase made by the customer on the offline platform
- order_num_total_ever_online       : Total number of purchases made by the customer on the online platform
- order_num_total_ever_offline      : Total number of purchases made by the customer on the offline platform
- customer_value_total_ever_offline : Total fees paid for the customer's offline purchases
- customer_value_total_ever_online  :  Total fees paid for the customer's online purchases
- interested_in_categories_12       : List of categories the customer has shopped in the last 12 months
"""

# ----------------------------------------------------------------------------------------------------------------
import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.mode.chained_assignment = None

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


# Outlier Analysis

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)


columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
           "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)


# Omnichannel means that customers shop from both online and offline platforms.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# We convert the column types from object to datetime format.

# first_order_date                   object ---------datetime
# last_order_date                    object ---------datetime
# last_order_date_online             object ---------datetime
# last_order_date_offline            object ---------datetime

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()


# Creating the CLTV Data Structure

df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)

cltv_df = pd.DataFrame({"customer_id": df["master_id"],
                        "recency_cltv_weekly": ((df["last_order_date"] - df["first_order_date"]).dt.days)/7,
                        "T_weekly": ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7,
                        "frequency": df["order_num_total"],
                        "monetary_cltv_avg": df["customer_value_total"] / df["order_num_total"]})

cltv_df.head()

"""
               customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg
0  cc29****-****-****-****               17.000    30.571      5.000            187.874 
1  f431****-****-****-****              209.857   224.857     21.000             95.883
2  69b6****-****-****-****               52.286    78.857      5.000            117.064
3  1854****-****-****-****                1.571    20.857      2.000             60.985
4  d6ea****-****-****-****               83.143    95.429      2.000            104.990  
"""


# Establishment of BG/NBD, Gamma-Gamma Models

# BG/NBD Model
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# <lifetimes.BetaGeoFitter: fitted with 19945 subjects, a: 0.00, alpha: 76.17, b: 0.00, r: 3.66>


# Expected purchases from customers within 3 months;
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])


# Expected purchases from customers within 6 months;
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

cltv_df.head()
"""
               customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month
0  cc29****-****-****-****               17.000    30.571      5.000            187.874              0.974              1.948 
1  f431****-****-****-****              209.857   224.857     21.000             95.883              0.983              1.966
2  69b6****-****-****-****               52.286    78.857      5.000            117.064              0.671              1.341
3  1854****-****-****-****                1.571    20.857      2.000             60.985              0.700              1.401
4  d6ea****-****-****-****               83.143    95.429      2.000            104.990              0.396              0.792
             
"""


# Gamma-Gamma Model
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
# <lifetimes.GammaGammaFitter: fitted with 19945 subjects, p: 4.15, q: 0.47, v: 4.08>

# Expected Average Profit
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])


# Calculation of CLTV with BG/NBD and Gamma-Gamma Models  - (6 MONTHS)

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,             # 6 MONTHS
                                   freq="W",           # T's frequency information (T_weekly)
                                   discount_rate=0.01) # discounts available
cltv_df["cltv"] = cltv

cltv_df.sort_values("cltv", ascending=False).head(10)


# Creating Segments by CLTV

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_df.head()

cltv_df.groupby("cltv_segment").agg( {"count", "mean", "sum"})

"""
             recency_cltv_weekly                 T_weekly                \
                             sum count    mean        sum count    mean   
cltv_segment                                                              
D                     693193.857  4987 139.000 808807.714  4987 162.183   
C                     461850.857  4986  92.630 562512.143  4986 112.818   
B                     408794.000  4986  81.988 500228.000  4986 100.327   
A                     336191.714  4986  67.427 411592.857  4986  82.550   
             
             frequency                  monetary_cltv_avg                \
                   sum count  mean               sum count    mean   
cltv_segment                                                         
D            18795.000  4987 3.769        464547.046  4987  93.152   
C            21962.000  4986 4.405        627181.647  4986 125.789   
B            25392.000  4986 5.093        800933.959  4986 160.637   
A            33140.000  4986 6.647       1140952.075  4986 228.831   
             exp_sales_3_month             exp_sales_6_month              \
                        
                           sum count  mean               sum count  mean   
cltv_segment                                                               
D                     2039.164  4987 0.409          4078.328  4987 0.818   
C                     2619.885  4986 0.525          5239.769  4986 1.051   
B                     2997.110  4986 0.601          5994.219  4986 1.202   
A                     3854.313  4986 0.773          7708.626  4986 1.546   
             
             exp_average_value                      cltv                
                           sum count    mean         sum count    mean  
cltv_segment                                                            
D                   492172.441  4987  98.691  400657.955  4987  80.340  
C                   659401.453  4986 132.251  689621.178  4986 138.312  
B                   837650.882  4986 168.001  994870.784  4986 199.533  
A                  1186787.639  4986 238.024 1806505.089  4986 362.316  

"""

