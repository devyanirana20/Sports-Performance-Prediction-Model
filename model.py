import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

teams=pd.read_csv("C:\\Users\\aditi\\Desktop\\PYTHON\\final.csv")


reg = LinearRegression()

train=teams[teams["year"]<2012].copy()
test=teams[teams["year"]>=2012].copy()

predictors=["athletes","prev_medals"]
target="medals"

reg.fit(train[predictors],train["medals"])

predictions=reg.predict(test[predictors])

test["predictions"]=predictions

test.loc[test["predictions"]<0,"predictions"]=0
test["predictions"]=test["predictions"].round()

#Now we will find the error in Our Model 
from sklearn.metrics import mean_absolute_error

error=mean_absolute_error(test["medals"],test["predictions"])

errors=(test["medals"]-test["predictions"]).abs()

error_by_team=errors.groupby(test["team"]).mean()

medals_by_team=test["medals"].groupby(test["team"]).mean()

error_ratio=error_by_team/medals_by_team

error_ratio =error_ratio[~pd.isnull(error_ratio)]   #Show only those Values 
#Whose Error Rtion is Not Null



#Now We Want Clean up the Data that is data 
#So We wll Clean up the inf values
error_ratio=error_ratio[np.isfinite(error_ratio)]

#Calculating Accuracy 
sum=0

for i in range(len(error_ratio)):
    sum+=error_ratio.iloc[i]
sum/=len(error_ratio)

accuracy=sum*100



