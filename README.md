# Fraud Detection in Electricity and Gas Consumption

# Problem Description
We know that the world is about to face a vital crisis in electricity, and gas as they’re considered our main source of power, regardless we are in deep into that problem right now. our main electricity provider wasn’t renewable, and still there’re many research and development in that area to find a new renewable one, like solar power, wind power, power walls, and many are still being developed. so from here and now we must adapt a solution to maximize our benefits, including power saving, enhance the power generation costs, and minimize factories waste. Unfortunately, all of them cost a fortune, though many people in this era should have a well conscious about that, but there’re bad shaming behaviour towards. many of them tried to steel, and misconfigure the costs, which increase our weird crisis. So we decided to extract them by building a Machine Learning model by analysing the data acquired along the way. in addition, as the data is huge, has a variety in date, locations, and for many population, we are about to use a big data analysis framework. may that help us to discover what ruins our world beloved system.

# Pipeline
Azure Synapse Resource is the main point at within the architecture it’s the service used to run the spark job in the cloud, it uses a Microsoft Data storage called Data Lake Storage Gen2 to store the dataset file, beside the Notebook codes files used to run the jobs and communicate with synapse. Step 1 User have to load the dataset file into data storage after creating one, then, after configuration, the storage service will give a connection string, in terms of file location just to use it inside notebook. Step 2 Instantiate a workspace within Synapse service, then create new pool to run our spark job on. Step 3 From Synapse workspace, Start a new Synapse studio, then upload your notebook after embedd the connection string to dataset. Step 4 Running the notebook will automatically run the service, service in turn will create jobs, splitting the data and send it to nodes to run the Machine learning algorithm, using the cloud spark service will lead us to run the job x100 times faster than local

# Data preprocessing and visualization
Dataset contains two tables: First is about Clients data and whether they have been labeled as fraud or not. Second table is the all invoices of these clients with information such as Consumption values in each consumption level (4 levels), Counter statutes, Reading Remarks, Number of months of the invoice and also whether the invoice is Electric or Gas. It was important to produce a dataset by joining the two tables and aggregating the invoices data of each client. We calculated 4 statistical numbers for each column and that is: Mean, Max, Min, Std. These 4 numbers will be calculated for each continuous value column for each client and for each type of invoice (ELEC or GAZ) so for example the column of reading_remarque will produce 8 columns in the new dataset, mean_elec_reading_remarque, max_elec_reading_remarque, min_elec_reading_remarque, std_elec_reading_remarque, mean_gaz_reading_remarque, max_gaz_reading_remarque, min_gaz_reading_remarque, std_gaz_reading_remarque. This was done on 6 continuous columns ( Reading Remarque, Months Number, Consumption Level 1, Consumption Level 2, Consumption Level 3, Consumption Level 4) to product 48 statistical columns for each client, finally some columns were dropped for not being important such as region or client id and so on. This produced a dataset of shape 135493 x 50 Finally another problem was that the data was very imbalanced where the number of frauds is as expected not very high![b64c5b94-8386-40ff-b9c0-049fcdd9631d](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/d3a12c5a-cbce-4afa-94a0-11d3358fd753)

# Extracting Insights from data

## Distribution of user categories: we have 3 categories (11, 12, 51), most of the clients are related to category 11.
![773956ca-4a58-4d9a-8b2d-a8a8ca233af8](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/d86e8387-ac74-43fb-97c8-2724c8af2908)

## Electricity consumption for each category: category 51 consumes electricity more than category 11 & 12.
![ea7c675f-a807-49f7-a100-b2397e15d901](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/9abe80a8-fe78-4246-96c6-f6259dc1ea00)
![73fbbecc-18e8-4262-9655-e8b084a27c71](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/f845b869-dc3e-4837-a34f-5d05a77a9dee)
![5c1f4659-40bc-442a-823f-ded1dbc767e8](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/27532a89-e26b-43d7-b797-6af517f88553)
![f1a94454-44a1-4321-9beb-ce3ee90cdba6](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/a3e4e8e6-0533-4d38-a367-8d6ae39af3b3)

## Gaz consumption for each category: category 3 consumes more than category 1 & 2. In general, no category consumes more than level 2 in 
![2767a0e7-0a98-4048-b49f-cb927540336c](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/81f36633-d938-4601-bad3-c7a0257000b9)
![50449dbe-fc5c-4b15-9b48-fb3235acf9fc](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/85cc2c69-973f-4a43-83d9-a56ae3250c02)
![29bb23e5-1946-4a14-a900-5a6886cae87f](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/49466dda-b5bc-47b0-a697-c4536290c6c3)
![724606f4-79aa-4726-aec9-b6715504f230](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/0296d299-77cf-460c-93ed-941d75015a57)

## Average electricity & gaz month number
![2e835bf6-f143-4621-9a2a-d2bb61a8275b](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/e64040ff-ad6e-4244-8c63-95cbfc5c612a)
![b123e931-e13e-4429-9eeb-6f7b225ee60a](https://github.com/ahmedayman1420/Fraud-Detection-in-Electricity-and-Gas-Consumption/assets/76254195/a83f16e1-0203-431b-8136-61dbefb21b9f)

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










