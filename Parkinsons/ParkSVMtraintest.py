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
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
from sklearn.ensemble import GradientBoostingClassifier

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


def GetLLMFeaturesOpenAI(contextFilepath, featuresToGet, features):
    #now feed headers and context to chatgpt and ask it to return which n features to include in readable format
    n = featuresToGet # f string doesn't work for some reason
    with open(contextFilepath,"r") as f:
        context = f.read()
    #get full response
    start = time.perf_counter()
    response = clientOpenAI.chat.completions.create(
                model = "gpt-3.5-turbo", #gpt-3.5-turbo, gpt-4o
                messages=[{"role":"developer","content": context + f"""Your Task:
                            Please print only a list of the available features in the order of their significance to predicting the desired variable, listing the most significant first, based on the above data. 
                           This list should be in a csv format, seperating features with a comma then a space, maintaining the exact feature names including capitalization.
                            For example, when given a list of features: FeaTure2, feature1, ftr3 : you would return the following: feature1, FeaTure2, ftr3, etc. in that format, ordered by significance.
                           These features should be selected based on their relevance and likelyhood to predict the variable given by and using the context. 
                           At least {n} of the available features should be returned. The only available features to be picked are given by the user, following this message."""},
                        {"role":"user","content":", ".join(features)},
                ],
            )
    end = time.perf_counter()
    #get chosen features
    LLMfeatures = response.choices[0].message.content

    #print(LLMfeatures)
    finalFeatures = LLMfeatures.split(", ")

    return finalFeatures,end -start

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

def gurobiSVM(X, y,k,gamma=1.0,M=1000,L0Regularization=False,sampleWeights =None):
    # Create a Gurobi environment and a model object
    with gp.Model("", env=env) as model:
        samples, features = X.shape
        assert samples == y.shape[0]

        #coefficients
        a = [model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"a{i}") for i in range(features)]
        
        # intercept
        beta = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta")


        #slack variables
        slack = [model.addVar(lb=0,name=f"xi{i}") for i in range(samples)]

        #l0 norm selectors
        if(L0Regularization):
            z = model.addVars(features, vtype=GRB.BINARY, name="z")
        
            # Link w and z constraints
            for j in range(features):
                model.addConstr(a[j] <= M * z[j])
                model.addConstr(a[j] >= -M * z[j])

            #costrain max k
            model.addConstr(quicksum(z[j] for j in range(features)) <= k, name="feature_budget")
        
        #constraints
        for i in range(samples):
            model.addConstr(
                y[i] * (quicksum(a[j]*X[i][j] for j in range(features)) - beta) >= 1 - slack[i],
                name=f"margin_{i}"
            )

        #objective
        if sampleWeights is not None:
            model.setObjective(
                quicksum(a[j]*a[j] for j in range(features)) + gamma * quicksum(sampleWeights[i] * slack[i] for i in range(samples)),
                GRB.MINIMIZE
            )
        else:
            model.setObjective(
                quicksum(a[j]*a[j] for j in range(features)) + gamma * quicksum(slack),
                GRB.MINIMIZE
            )

        model.setParam('OutputFlag', 0)
        model.params.timelimit = 60
        model.params.mipgap = 0.001

        model.optimize()
        #print(model.Status)
            
        equation = {}
        if model.SolCount > 0:
            equation['a'] = [a[j].X for j in range(features)]
            equation['beta'] = beta.X
        else:
            #didnt converge, return other apporximatination
            print("Optimization did not converge")
            equation['a'] = [1 for _ in range(features)]
            equation['beta'] = 1
        return equation

def split_folds(features, response, train_mask):
    """
    Assign folds to either train or test partitions based on train_mask.
    """
    xtrain = features[train_mask,:]
    xtest = features[~train_mask,:]
    ytrain = response[train_mask]
    ytest = response[~train_mask]
    return xtrain, xtest, ytrain, ytest

def cross_validate_resultsGBM(features, response, folds, standardize, seed,gamma=1,sampleWeights=None):
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
    acc = 0
    roc = 0
    f1 = 0
    # Exclude folds from training, one at a time, 
    #to get out-of-sample estimates of the roc
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
        if sampleWeights is not None:
            totalpos = sum(ytrain==1)
            totalneg = sum(ytrain==-1)
            posweight = (totalpos + totalneg)/(2*totalpos)
            negweight = (totalpos + totalneg)/(2*totalneg)
            weights = np.where(ytrain == 1,posweight,negweight) #n/2n(t)
            model = GradientBoostingClassifier(n_estimators=10,max_depth=3,learning_rate=1,max_features=100,random_state=seed)
            model = model.fit(xtrain,ytrain,sample_weight=weights)
        else:
            model = GradientBoostingClassifier(n_estimators=10,max_depth=3,learning_rate=1,max_features=100,random_state=seed)
            model = model.fit(xtrain,ytrain)
        
        ypred = model.predict(xtest)
        acc += accuracy_score(ytest,ypred)/folds
        roc += roc_auc_score(ytest, ypred) / folds
        f1 += f1_score(ytest,ypred)/folds
    # Report the average out-of-sample roc
    return acc,roc,f1

def cross_validate_results(features, response, k, folds, standardize, seed,gamma=1,sampleWeights=False):
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
    acc = 0
    roc = 0
    f1 = 0
    # Exclude folds from training, one at a time, 
    #to get out-of-sample estimates of the roc
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
        if (sampleWeights):
            totalpos = sum(ytrain==1)
            totalneg = sum(ytrain==-1)
            posweight = (totalpos + totalneg)/(2*totalpos)
            negweight = (totalpos + totalneg)/(2*totalneg)
            weights = np.where(ytrain == 1,posweight,negweight) #n/2n(t)
            equation = gurobiSVM(xtrain, ytrain,k,gamma=gamma,M=1000,L0Regularization=True,sampleWeights=weights)
        else:
            equation = gurobiSVM(xtrain, ytrain,k,gamma=gamma,M=1000,L0Regularization=True)
        ypred = findYPred(xtest,equation)
        acc += accuracy_score(ytest,ypred)/folds
        roc += roc_auc_score(ytest, ypred) / folds
        f1 += f1_score(ytest,ypred)/folds
    # Report the average out-of-sample roc
    return acc,roc,f1



def findYPred(X,equation):
    decision_scores = X @ equation["a"] - equation["beta"]  # (dot product of each row with a) - beta

    #convert to bipolar
    y_pred = (decision_scores > 0).astype(int)
    return np.where(y_pred == 0, -1, 1)


def TrainAppendResults(df,y,seed,results,model,SvmFeatureAmount):
    #split, standardize, train bss, and predict on specified df and seed, and append data to specified lists

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y,random_state = seed)

    #standardize test and train sep
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    #or with cross validation
    #equation = gurobiSVM(X_train_std, y_train.to_numpy(),k,gamma=1,M=1000,L0Regularization=True)#uses featureAmount for k, or col dim if smaller
    
    totalpos = sum(y_train==1)
    totalneg = sum(y_train==-1)
    posweight = (totalpos + totalneg)/(2*totalpos)
    negweight = (totalpos + totalneg)/(2*totalneg)
    weights = np.where(y_train == 1,posweight,negweight) #n/2n(t)
    start = time.perf_counter()
    equation = gurobiSVM(X_train_std, y_train.to_numpy(),SvmFeatureAmount,gamma=1,M=1000,L0Regularization=True,sampleWeights=weights)
    end = time.perf_counter()
    # Predict and evaluate (@ is matrix multiplication) #headers? array types?

    y_pred = findYPred(X_test_std,equation)

    results[model]["acc"].append(accuracy_score(y_test, y_pred))
    results[model]["roc"].append(roc_auc_score(y_test, y_pred))
    results[model]["f1"].append(f1_score(y_test, y_pred))

    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NonToxic','Toxic'])
    # disp.plot()
    # plt.show()

    #add training results
    y_pred = findYPred(X_train_std,equation)

    results[f"{model}train"]["acc"].append(accuracy_score(y_train, y_pred))
    results[f"{model}train"]["roc"].append(roc_auc_score(y_train, y_pred))
    results[f"{model}train"]["f1"].append(f1_score(y_train, y_pred))

    #return weights to use for matched feature comparison
    return equation["a"],end -start

def match_features(givenFeatures,otherFeatures):
    """otherFeatures is the features that givenFeatures is being compared to (SVM)"""
    totalMatched = sum(1 for feature in givenFeatures if feature in otherFeatures)
    return totalMatched/len(givenFeatures)

def save_results(results,ModelName,p,k,trials):
    output = {
            'acc': results[ModelName]['acc'],
            'roc': results[ModelName]['roc'],
            'f1': results[ModelName]["f1"],
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
        output["features matched to SVM"] = results[ModelName]["matched features"]
    if ModelName in ["LLMtrain","Randtrain"]:
        output["features specified"] = [p] *trials #make a TRIAL long list of the number 'feature'
    pd.DataFrame(output).to_csv(f'output{ModelName}p{p}k{k}.csv', index=True)

def run_trial(model,df,y,seed,DfFeatureAmount,results,SvmFeatureAmount,contextFile=None,otherFeatureNames=None):
    #1 get df for specific model
    
    match model:
        case "SVM":
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

    if model == "SVM":
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
df = pd.read_csv("Parkinsons/pd_speech_features.csv",header =1) 
#drop rows where the target is na
df = df[~df["class"].isna()]

#get numerilc cols 
numerical_cols = df.select_dtypes(include='number').columns.tolist()
#fillna for numerical
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
#separate target and df
y = df["class"]
df.drop("class",axis=1,inplace=True)

#turn categorical y into binary 1 for 1 and 0
y = pd.Series([-1 if cla == 1 else 1 for cla in y])

#--------------------------------------------------MODEL TRAINING-------------------------------------------------------


TRIALS = 10 #this number of trials for each unique combination of feature amount and model type
DfFeatureAmount = 200 #list of features to try [10,15,20]
SvmFeatureAmount = 100

results = {
    'SVM' : {"acc":[],"roc":[],"f1":[]},
    'LLM' : {"acc":[],"roc":[],"f1":[]},
    'Rand' : {"acc":[],"roc":[],"f1":[]},
    'SVMtrain' : {"acc":[],"roc":[],"f1":[],"training time": [],'features used':[]},
    'LLMtrain' : {"acc":[],"roc":[],"f1":[],"LLM time":[],"training time": [],'features used':[],"featuresChosenByLLM":[]},
    'Randtrain' : {"acc":[],"roc":[],"f1":[],"training time": [],'features used':[]}
}


currTrial = 0
while currTrial < TRIALS:
    random.seed(currTrial)
    
    SVMChosenFeatureNames = run_trial("SVM",df,y,currTrial,DfFeatureAmount,results,SvmFeatureAmount) 

    #///////[LLM]\\\\\\\
    run_trial("LLM",df,y,currTrial,DfFeatureAmount,results,SvmFeatureAmount,contextFile="Parkinsons/contextPark.txt",otherFeatureNames=SVMChosenFeatureNames)


    #///////[Rand]\\\\\\\
    run_trial("Rand",df,y,currTrial,DfFeatureAmount,results,SvmFeatureAmount,otherFeatureNames=SVMChosenFeatureNames)
    
    currTrial += 1

for model in ["SVM","LLM","Rand"]:
    save_results(results,model,DfFeatureAmount,SvmFeatureAmount,TRIALS)
    save_results(results,f"{model}train",DfFeatureAmount,SvmFeatureAmount,TRIALS)