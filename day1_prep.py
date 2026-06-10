import pandas as pd
import numpy as np
df = pd.read_csv("Nassau Candy Distributor (1).csv")
print("Total rows    :", len(df))
print("Total columns :", len(df.columns))
print("Missing values:")
print(df.isnull().sum())
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
df['Ship Date Fixed'] = pd.to_datetime({
    'year':  df['Order Date'].dt.year,
    'month': df['Ship Date'].dt.month,
    'day':   df['Ship Date'].dt.day
})
df.loc[df['Ship Date Fixed'] < df['Order Date'],
       'Ship Date Fixed'] += pd.DateOffset(years=1)
df['Lead Time'] = (df['Ship Date Fixed'] - df['Order Date']).dt.days

print("Lead Time min :", df['Lead Time'].min())
print("Lead Time max :", df['Lead Time'].max())
print("Lead Time avg :", df['Lead Time'].mean().round(1))
factory_map = {
    'Wonka Bar - Nutty Crunch Surprise'  : "Lot's O' Nuts",
    'Wonka Bar - Fudge Mallows'          : "Lot's O' Nuts",
    'Wonka Bar -Scrumdiddlyumptious'     : "Lot's O' Nuts",
    'Wonka Bar - Milk Chocolate'          : "Wicked Choccy's",
    'Wonka Bar - Triple Dazzle Caramel'  : "Wicked Choccy's",
    'Laffy Taffy'                        : 'Sugar Shack',
    'SweeTARTS'                          : 'Sugar Shack',
    'Nerds'                              : 'Sugar Shack',
    'Fun Dip'                            : 'Sugar Shack',
    'Fizzy Lifting Drinks'               : 'Sugar Shack',
    'Everlasting Gobstopper'             : 'Secret Factory',
    'Lickable Wallpaper'                 : 'Secret Factory',
    'Wonka Gum'                          : 'Secret Factory',
    'Hair Toffee'                        : 'The Other Factory',
    'Kazookles'                          : 'The Other Factory',
}
df['Factory'] = df['Product Name'].map(factory_map)

print("Orders per factory:")
print(df['Factory'].value_counts())
print("Missing:", df['Factory'].isnull().sum())
factory_coords = {
    "Lot's O' Nuts"    : (32.881893, -111.768036),
    "Wicked Choccy's"  : (32.076176,  -81.088371),
    'Sugar Shack'      : (48.119140,  -96.181150),
    'Secret Factory'   : (41.446333,  -90.565487),
    'The Other Factory': (35.117500,  -89.971107),
}
df['Factory Lat'] = df['Factory'].map(lambda f: factory_coords[f][0])
df['Factory Lon'] = df['Factory'].map(lambda f: factory_coords[f][1])

print(df[['Factory','Factory Lat','Factory Lon']].drop_duplicates().to_string())
df['Gross Margin %']   = (df['Gross Profit'] / df['Sales'] * 100).round(2)
df['Profit per Unit']  = (df['Gross Profit'] / df['Units']).round(2)
df['Revenue per Unit'] = (df['Sales']        / df['Units']).round(2)

print("KPI sample (first 3 rows):")
print(df[['Product Name','Gross Margin %',
          'Profit per Unit','Revenue per Unit']].head(3).to_string())
lt_mean = df['Lead Time'].mean()
lt_std  = df['Lead Time'].std()

df_clean = df[
    (df['Lead Time'] >= lt_mean - 3 * lt_std) &
    (df['Lead Time'] <= lt_mean + 3 * lt_std)
].copy()
df_clean.to_csv('nassau_clean.csv', index=False)

print("Original rows  :", len(df))
print("Clean rows     :", len(df_clean))
print("Rows removed   :", len(df) - len(df_clean))
print("Total columns  :", len(df_clean.columns))
print("nassau_clean.csv saved in your project folder!")