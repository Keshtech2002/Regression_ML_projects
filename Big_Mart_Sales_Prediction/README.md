# Medical_INsurance_Cost_Prediction

### ABOUT THE PROJECT
Taking some inferences from the [Big Mart Sales Datasets](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data) and building a regression model to predict.
#### Concepts mentioned here were learnt and utilized in this project

#### Exploratory Data Analysis (EDA)
- Countplot
- Distribution Plot

#### Regression
- XGBoost Regression

#### Other concepts
- Correlation; Positive and Negative

### Files
| File | |

### ABOUT THE DATA
- **Number of Instances**: 8523
- **Number of Features**: 11 features.
    - Missing values occur
        - Item_Weight 1463
        - Outlet_Size 2410
    - 7 categorical features : 'Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
- **Target**: 'Item Outlet Sales'

- **Features Info**
- 'Item_Identifier' : Codes to identify the items {FD___: Food, DR___: Drink, NC___: Non Consumable}
- 'Item_Weight' : Weight of item in kg
- 'Item_Fat_Content' : Fat content (Categorical)
- 'Item_Visibility' : How much often do the mart have them
- 'Item_Type' : The type of product 
- 'Item_MRP': Maximum Rated Price
- 'Outlet_Identifier' : Identification of outlet stores where we find the item
- 'Outlet_Establishment_Year' : When the outlet was established
- 'Outlet_Size' : Size of outlet
- 'Outlet_Location_Type' : Kind of place where the outlet is located
- 'Outlet_Type' : Type of outlet
- 'Item_Outlet_Sales' : Sales of item in the outlet