# LLMBestSubsetSelection

*.env information*

    $Fill in GPTAPIKEY env file to use OpenAI api, and GEMAPIKEY to use Gemini api

    $You can find the three options to fill in for Gurobi license in your gurobi.lic file after downloading your gurobi license

*usage*

    $to test these files, first download the .csv file for the dataset from the link in the datasets.txt file. If it's unclear which dataset to use, the name I used is hardcoded in the .py files in the DATA CLEANING section

    $put the .csv in the correct folder

    $fill in required .env keys

    $specify in the top of the  MODEL TRAINING section how many trials you want to run, how many features the LLM and Random models should select in the first step, and how many features all models should select in the optimization step

    $change which models should run if desired

    $The LLMBSS framework will generate several .csv files, with a testing and training results file for each model

*more information*

    $The link to my LLMBSS article is here: [LLMBSSArticle](https://kianvd.github.io/LLMBSSArticle/LLMBSSArticle.pdf)
