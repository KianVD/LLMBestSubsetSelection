#Toxic dataset gemini
from google import genai
from openai import OpenAI
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import gurobipy as gp
from gurobipy import GRB
from sklearn.preprocessing import StandardScaler
import time
import random
from gurobipy import quicksum
from sklearn.metrics import r2_score, mean_squared_error
random.seed(0)

# Set GEMINI api key
load_dotenv(dotenv_path=".env")
apikey = os.getenv("GEMAPIKEY")
os.environ['GEMINI_API_KEY'] = apikey
client = genai.Client()
clientOpenAI = OpenAI(
    api_key = os.getenv("GPTAPIKEY")
)

options = {
"WLSACCESSID":os.getenv("WLSACCESSID"),
"WLSSECRET":os.getenv("WLSSECRET"),
"LICENSEID":int(os.getenv("LICENSEID")),
}

env = gp.Env(params=options)

def GetLLMFeaturesGemini(contextFilepath, featuresToGet, features):
    #now feed headers and context to chatgpt and ask it to return which n features to include in readable format
    n = featuresToGet # f string doesn't work for some reason
    with open(contextFilepath,"r") as f:
        context = f.read()
    #get full response
    start = time.perf_counter()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""{context}\nYour Task:
                            Please print only a list of the available features in the order of their significance to predicting the desired variable, listing the most significant first, based on the above data. 
                           This list should be in a csv format, seperating features with a comma then a space, maintaining the exact feature names including capitalization.
                            For example, when given a list of features: FeaTure2, feature1, ftr3 : you would return the following: feature1, FeaTure2, ftr3, etc. in that format, ordered by significance.
                           These features should be selected based on their relevance and likelyhood to predict the variable given by and using the context. 
                           At least {n} of the available features should be returned. The only available features to be picked are given by the user, following this message.\n{", ".join(features)}""",
    )
    end = time.perf_counter()

    #get chosen features
    LLMfeatures = response.text

    #print(LLMfeatures)
    finalFeatures = LLMfeatures.split(", ")

    return finalFeatures,end -start

def NarrowDownDFLLM(df,contextFilePath, featuresToGet):
    headers = df.columns.tolist()

    #get features chosen by llm
    newHeaders,time = GetLLMFeaturesGemini(contextFilePath, featuresToGet,headers)

    valid_cols = list()
    for col in newHeaders:
        if col in df.columns and col not in valid_cols:
            valid_cols.append(col)
    valid_cols = valid_cols[:featuresToGet] #cut off any extra columns if llm included too many (they are ranked in order of importance so least important get cut off first )
    return df[valid_cols].copy(),time

def miqp(features, response, non_zero, verbose=False):
    """
    Deploy and optimize the MIQP formulation of L0-Regression.
    """
    assert isinstance(non_zero, (int, np.integer))
    # Create a Gurobi environment and a model object
    with gp.Model("", env=env) as regressor:
        samples, dim = features.shape
        assert samples == response.shape[0]
        assert non_zero <= dim

        # Append a column of ones to the feature matrix to account for the y-intercept
        X = np.concatenate([features, np.ones((samples, 1))], axis=1)  

        # Decision variables
        norm_0 = regressor.addVar(lb=0, ub=non_zero, name="norm")
        beta = regressor.addMVar((dim + 1,), lb=-GRB.INFINITY, name="beta") # Weights
        intercept = beta[dim] # Last decision variable captures the y-intercept

        regressor.setObjective(beta.T @ X.T @ X @ beta
                               - 2*response.T @ X @ beta
                               + np.dot(response, response), GRB.MINIMIZE)

        # Budget constraint based on the L0-norm
        regressor.addGenConstrNorm(norm_0, beta[:-1], which=0, name="budget")

        if not verbose:
            regressor.params.OutputFlag = 0
        regressor.params.timelimit = 60
        regressor.params.mipgap = 0.001
        regressor.optimize()

        coeff = np.array([beta[i].X for i in range(dim)])
        return intercept.X, coeff

# Define functions necessary to perform hyper-parameter tuning via cross-validation

def split_folds(features, response, train_mask):
    """
    Assign folds to either train or test partitions based on train_mask.
    """
    xtrain = features[train_mask,:]
    xtest = features[~train_mask,:]
    ytrain = response[train_mask]
    ytest = response[~train_mask]
    return xtrain, xtest, ytrain, ytest

def cross_validate(features, response, non_zero, folds, standardize, seed):
    """
    Train an L0-Regression for each fold and report the cross-validated MSE.
    """
    if seed is not None:
        np.random.seed(seed)
    samples, dim = features.shape
    assert samples == response.shape[0]
    fold_size = int(np.ceil(samples / folds))
    # Randomly assign each sample to a fold
    shuffled = np.random.choice(samples, samples, replace=False)
    mse_cv = 0
    # Exclude folds from training, one at a time, 
    #to get out-of-sample estimates of the MSE
    for fold in range(folds):
        idx = shuffled[fold * fold_size : min((fold + 1) * fold_size, samples)]
        train_mask = np.ones(samples, dtype=bool)
        train_mask[idx] = False
        xtrain, xtest, ytrain, ytest = split_folds(features, response, train_mask)
        if standardize:
            scaler = StandardScaler()
            scaler.fit(xtrain)
            xtrain = scaler.transform(xtrain)
            xtest = scaler.transform(xtest)
        intercept, beta = miqp(xtrain, ytrain, non_zero)
        ypred = np.dot(xtest, beta) + intercept
        mse_cv += mean_squared_error(ytest, ypred) / folds
    # Report the average out-of-sample MSE
    return mse_cv

def L0_regression(features, response, maxfeatures,folds=5, standardize=False, seed=None):
    """
    Select the best L0-Regression model by performing grid search on the budget.
    """
    dim = features.shape[1]
    best_mse = np.inf
    best = 0
    #Find highest possible features
    max_k = dim if maxfeatures >= dim else maxfeatures
    # Grid search to find best number of features to consider
    for i in range(1, max_k + 1):
        val = cross_validate(features, response, i, folds=folds,
                             standardize=standardize, seed=seed)
        if val < best_mse:
            best_mse = val
            best = i
    if standardize:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)
    intercept, beta = miqp(features, response, best)
    return intercept, beta


def TrainAppendResults(df,y,seed,results,model,BSSFeatureAmount):
    #split, standardize, train bss, and predict on specified df and seed, and append data to specified lists

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2,random_state = seed)

    #standardize test and train sep
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    start = time.perf_counter()
    intercept, coefficients = miqp(X_train_std, y_train.to_numpy(), min(BSSFeatureAmount,X_train_std.shape[1]))#uses featureAmount for k, or col dim if smaller
    #intercept, coefficients = L0_regression(X_train_std,y_train.to_numpy(),BSSFeatureAmount,standardize=True,seed=seed) #set seed and feature as max k 
    end = time.perf_counter()

    # Predict and evaluate (@ is matrix multiplication) #headers? array types?

    # Predict and evaluate (@ is matrix multiplication) #headers? array types?
    y_pred = (X_test_std @ coefficients) +intercept

    results[model]["r2"].append(r2_score(y_test, y_pred))
    results[model]["mse"].append(mean_squared_error(y_test, y_pred))

    #add training results
    y_pred = (X_train_std @ coefficients) +intercept

    results[f"{model}train"]["r2"].append(r2_score(y_train, y_pred))
    results[f"{model}train"]["mse"].append(mean_squared_error(y_train, y_pred))

    #return weights to use for matched feature comparison
    return coefficients,end -start

def match_features(givenFeatures,otherFeatures):
    """otherFeatures is the features that givenFeatures is being compared to (BSS)"""
    totalMatched = sum(1 for feature in givenFeatures if feature in otherFeatures)
    return totalMatched/len(givenFeatures)

def save_results(results,ModelName,p,k,trials):
    output = {
            'r2': results[ModelName]['r2'],
            'mse': results[ModelName]['mse'],
            'rmse (Spotify Streams)': np.sqrt(results[ModelName]["mse"])
        }
    if "LLM time" in results[ModelName]:
        output["LLM time (sec)"]= results[ModelName]['LLM time']
    if "training time" in results[ModelName]:
        output["training time (sec)"]= results[ModelName]['training time']
    if "features used" in results[ModelName]:
        output["feaures used"] =results[ModelName]["features used"]
    if ModelName in ["LLMtrain"]:
        output["features chosen by LLM"] = results[ModelName]["featuresChosenByLLM"] #extra column that tells how many features the llm returns (should be equal to features specified, but may not be if LLM didn't listen)
    if "matched features" in results[ModelName]:
        output["features matched to BSS"] = results[ModelName]["matched features"]
    if ModelName in ["LLMtrain","Randtrain"]:
        output["features specified"] = [p] *trials #make a TRIAL long list of the number 'feature'
    pd.DataFrame(output).to_csv(f'output{ModelName}p{p}k{k}.csv', index=True)

def run_trial(model,df,y,seed,DfFeatureAmount,results,SvmFeatureAmount,contextFile=None,otherFeatureNames=None):
    #1 get df for specific model
    
    match model:
        case "BSS":
            #original df
            currdf = df
        case "LLM":
            #get newdf with chosen columns using llm 
            currdf,LLMtime = NarrowDownDFLLM(df,contextFile,DfFeatureAmount) #here is where you specify how many features the LLM should choose
            #find number of features chosen by llm, make sure its not 0
            llmFeatureAmount = currdf.shape[1]
            print("Number of columsn:" ,llmFeatureAmount)
            if llmFeatureAmount < 1:
                print(f"LLM didn't give any features") #error
            results["LLMtrain"]["featuresChosenByLLM"].append(llmFeatureAmount)
        case "Rand":
            currdf = df[random.sample(df.columns.tolist(),DfFeatureAmount)].copy()

    #2 trainappend results

    Coef,trainTime = TrainAppendResults(currdf,y,seed,results,model,SvmFeatureAmount)
    #record time of whole trial
    
    results[f"{model}train"]["training time"].append(trainTime)
    if model == "LLM":
        results[f"{model}train"]["LLM time"].append(LLMtime)

    #find the number of features used
    totalfeaturesused = 0
    for i in range(len(Coef)):
        if Coef[i] != 0:
            totalfeaturesused +=1
    results[f"{model}train"]["features used"].append(totalfeaturesused)

    if model == "BSS":
        ChosenFeatureNames = list()
        for i in range(len(currdf.columns)):
            if Coef[i] != 0:
                ChosenFeatureNames.append(currdf.columns[i])
        return ChosenFeatureNames
    else:
        #find matched features with BSS
        if otherFeatureNames is not None:
            #do just for train
            if "matched features" not in results[f"{model}train"]:
                results[f"{model}train"]["matched features"] = list()
            results[f"{model}train"]["matched features"].append(match_features(currdf.columns,otherFeatureNames))
                
#--------------------------------------------------DATA CLEANING-------------------------------------------------------

#find dataset with 1000 features (genes?)
df = pd.read_csv("Spotify/Most Streamed Spotify Songs 2024.csv",encoding='ISO-8859-1') 
#drop rows where the target is na
df = df[~df["Spotify Streams"].isna()]


#drop stupid TIDAL colun (all nOne)
df = df.drop(columns="TIDAL Popularity")

#clean data and standardize
df["Release Date"] = pd.to_datetime(df["Release Date"])#turn to datetime Extract features like year, month, day, or even weekday

df['Year'] = df['Release Date'].dt.year
df['Month'] = df['Release Date'].dt.month
df['Day'] = df['Release Date'].dt.day
df['Weekday'] = df['Release Date'].dt.weekday

df = df.drop(columns="Release Date")

#df["ISRC"] = #sepearte into parts 
df["Country Code"] = df["ISRC"].str[:2]
df["Registrant Code"] = df["ISRC"].str[2:5]
df["Designation Code"] = df["ISRC"].str[7:12]

df = df.drop(columns="ISRC")
df = df.drop(columns="Designation Code")
#take out quotes and commas from following
for column in ["Spotify Streams","Spotify Playlist Count","Spotify Playlist Reach",
               "YouTube Views","YouTube Likes","TikTok Posts","TikTok Likes","TikTok Views",
               "YouTube Playlist Reach","AirPlay Spins","Deezer Playlist Reach","Pandora Streams",
               "Pandora Track Stations","Soundcloud Streams","Shazam Counts"]:
    df[column] = df[column].str.replace(',', '', regex=False).astype('Float64') #use Int64 for null values

#get numerilc cols 
numerical_cols = df.select_dtypes(include='number').columns.tolist()
#fillna for numerical
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

#convert categorical columns to numerical and fillna
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

for i in range(len(categorical_cols)):
    df[categorical_cols[i]] = df[categorical_cols[i]].astype(str)
    df[categorical_cols[i]] = df[categorical_cols[i]].fillna(df[categorical_cols[i]].mode().iloc[0])
    df[categorical_cols[i]] = le.fit_transform(df[categorical_cols[i]]) #label encoding
    #newdf = pd.concat([newdf.drop(columns=categorical_cols[i]),pd.get_dummies(newdf, columns=[categorical_cols[i]])],axis=1) #one hot encoding

#separate target and df
y = df["Spotify Streams"]
df = df.drop(columns=["Spotify Streams"])

#--------------------------------------------------MODEL TRAINING-------------------------------------------------------


TRIALS = 10 #this number of trials for each unique combination of feature amount and model type
DfFeatureAmount = 20 #list of features to try [10,15,20]
SvmFeatureAmount = 10

results = {
        'BSS' : {"r2":[],"mse":[]},
        'LLM' : {"r2":[],"mse":[]},
        'Rand' : {"r2":[],"mse":[]},
        'BSStrain' : {"r2":[],"mse":[],"training time": [],'features used':[]},
        'LLMtrain' : {"r2":[],"mse":[],"LLM time":[],"training time": [],'features used':[],"featuresChosenByLLM":[]},
        'Randtrain' : {"r2":[],"mse":[],"training time": [],'features used':[]}
    }


currTrial = 0
while currTrial < TRIALS:
    random.seed(currTrial)
    
    BSSChosenFeatureNames = run_trial("BSS",df,y,currTrial,DfFeatureAmount,results,SvmFeatureAmount) 

    #///////[LLM]\\\\\\\
    run_trial("LLM",df,y,currTrial,DfFeatureAmount,results,SvmFeatureAmount,contextFile="Spotify/contextSpotify.txt",otherFeatureNames=BSSChosenFeatureNames)


    #///////[Rand]\\\\\\\
    run_trial("Rand",df,y,currTrial,DfFeatureAmount,results,SvmFeatureAmount,otherFeatureNames=BSSChosenFeatureNames)
    
    currTrial += 1

for model in ["BSS","LLM","Rand"]:
    save_results(results,model,DfFeatureAmount,SvmFeatureAmount,TRIALS)
    save_results(results,f"{model}train",DfFeatureAmount,SvmFeatureAmount,TRIALS)