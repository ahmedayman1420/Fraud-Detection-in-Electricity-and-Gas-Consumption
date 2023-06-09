# Fraud Detection in Electricity and Gas Consumption

# Problem Description
We know that the world is about to face a vital crisis in electricity, and gas as they’re considered our main source of power, regardless we are in deep into that problem right now. our main electricity provider wasn’t renewable, and still there’re many research and development in that area to find a new renewable one, like solar power, wind power, power walls, and many are still being developed. so from here and now we must adapt a solution to maximize our benefits, including power saving, enhance the power generation costs, and minimize factories waste. Unfortunately, all of them cost a fortune, though many people in this era should have a well conscious about that, but there’re bad shaming behaviour towards. many of them tried to steel, and misconfigure the costs, which increase our weird crisis. So we decided to extract them by building a Machine Learning model by analysing the data acquired along the way. in addition, as the data is huge, has a variety in date, locations, and for many population, we are about to use a big data analysis framework. may that help us to discover what ruins our world beloved system.

# Pipeline
Azure Synapse Resource is the main point at within the architecture it’s the service used to run the spark job in the cloud, it uses a Microsoft Data storage called Data Lake Storage Gen2 to store the dataset file, beside the Notebook codes files used to run the jobs and communicate with synapse. Step 1 User have to load the dataset file into data storage after creating one, then, after configuration, the storage service will give a connection string, in terms of file location just to use it inside notebook. Step 2 Instantiate a workspace within Synapse service, then create new pool to run our spark job on. Step 3 From Synapse workspace, Start a new Synapse studio, then upload your notebook after embedd the connection string to dataset. Step 4 Running the notebook will automatically run the service, service in turn will create jobs, splitting the data and send it to nodes to run the Machine learning algorithm, using the cloud spark service will lead us to run the job x100 times faster than local

# Data preprocessing and visualization
Dataset contains two tables: First is about Clients data and whether they have been labeled as fraud or not. Second table is the all invoices of these clients with information such as Consumption values in each consumption level (4 levels), Counter statutes, Reading Remarks, Number of months of the invoice and also whether the invoice is Electric or Gas. It was important to produce a dataset by joining the two tables and aggregating the invoices data of each client. We calculated 4 statistical numbers for each column and that is: Mean, Max, Min, Std. These 4 numbers will be calculated for each continuous value column for each client and for each type of invoice (ELEC or GAZ) so for example the column of reading_remarque will produce 8 columns in the new dataset, mean_elec_reading_remarque, max_elec_reading_remarque, min_elec_reading_remarque, std_elec_reading_remarque, mean_gaz_reading_remarque, max_gaz_reading_remarque, min_gaz_reading_remarque, std_gaz_reading_remarque. This was done on 6 continuous columns ( Reading Remarque, Months Number, Consumption Level 1, Consumption Level 2, Consumption Level 3, Consumption Level 4) to product 48 statistical columns for each client, finally some columns were dropped for not being important such as region or client id and so on. This produced a dataset of shape 135493 x 50 Finally another problem was that the data was very imbalanced where the number of frauds is as expected not very high!
![4eb282a2-599c-4cec-97b3-2ea054b330a3](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/d1ddc4e0-52cb-4356-922f-cdda7a7d3453)


# Extracting Insights from data

## Distribution of user categories: we have 3 categories (11, 12, 51), most of the clients are related to category 11.
![3a3b396c-5b82-487d-bd9c-acc5745dc0f8](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/16176e6d-605e-47d9-839d-736f31c64771)


## Electricity consumption for each category: category 51 consumes electricity more than category 11 & 12.
![ef5c63c3-48ad-4144-b4da-4d0d535e47cf](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/e0e574c3-7c05-4bee-954b-b917daaa733e)
![45959966-069b-4378-bfbf-ed6a822eb29b](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/8e6d6046-21e4-4087-b325-0249ed819440)
![432125d8-bddc-4f7c-9e87-e6059ca2e108](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/0770a4dd-0a3e-497f-bb7d-37cf36e0c696)
![98335349-0c8c-4a73-8bb8-0a21d46700f7](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/48a75299-2e9d-4fa0-81f2-e83fb2991fc1)


## Gaz consumption for each category: category 3 consumes more than category 1 & 2. In general, no category consumes more than level 2 in 
![2ae6e384-da42-4178-84b2-359bdb8e3516](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/f86aeb71-eea2-4d6c-b611-8db998b75788)
![bf4235dc-5541-40b3-9394-764ccf8d28ec](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/0c33372e-e735-4b53-9e6c-3dd59e4e4f5f)
![f2883ba3-7426-467b-92f0-c89202374166](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/cc6891e4-bc30-4cb1-8954-ef946e0b1d66)
![b47deb0f-f69c-44ba-9f20-444b857fb3c5](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/58049da4-c8e6-47fc-8467-0b0de2207af6)


## Average electricity & gaz month number
![a3922dca-612e-457e-98ba-6823bc8deebe](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/051f182e-e09e-4b59-99ce-8941a2f19d24)
![1ec4174b-06c6-4952-a165-93ec6cb88a3a](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/9746c5a9-5890-415a-abba-07ce8e011de3)

# Model and Classifier Tuning

| ID  | Classifier            | Oversampling | Hyperparameter tuning | Spark    | Cloud  |
| --- | --------------------- | ------------ | --------------------- | -------- | ------ |
| 0   | SVM                   | none         | C & Kernel & Gamma     | none     | none   |
| 1   | SVM                   | SMOTE        | none                  | none     | none   |
| 2   | Logistic Regression  | none         | none                  | none     | none   |
| 3   | Logistic Regression  | SMOTE        | none                  | none     | none   |
| 4   | Logistic Regression  | SMOTE        | Pyspark               | none     | none   |
| 5   | Logistic Regression  | SMOTE        | Pyspark               | Azure    |        |

# Results and evaluation

| ID  | Classifier            | Training time | Accuracy   |
| --- | --------------------- | ------------- | ---------- |
| 0   | SVM                   | 25 minutes    | 94.22 %    |
| 1   | SVM                   | 100 minutes   | 66.78 %    |
| 2   | Logistic Regression  | 1 minutes     | 94.20 %    |
| 3   | Logistic Regression  | 5 minutes     | 64.35 %    |
| 4   | Logistic Regression  | 1 minutes     | 68.66 %    |
| 5   | Logistic Regression  | 35 seconds    | 68.50 %    |










