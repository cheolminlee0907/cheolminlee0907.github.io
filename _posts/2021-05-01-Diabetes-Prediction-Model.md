---
layout: single
title:  "Diabetes Prediction Model"
categories: "R"
author_profile: false
sidebar: 
    nav: 'docs'
---

This post contains the prediction models for the Diabetes.

```{r}
library(tidyverse)
library(tidymodels)
library(rpart.plot)
library(ggridges)
```


# Introduction

According to the Centers for Disease Control and Prevention, there were 34.2 million people in the United States with a medical history of Diabetes. Even though there are numerous medications such as GLP-1 Agonist, SGLT2 inhibitors, insulin, etc., the incidence of diabetes is increasing. By 2050, the estimation of the diabetes prediction is 1 in 3 adults. There are many factors associated with diabetes such as obesity rates, ages, and others. This post will make prediction models of diabetes with the dataset from the Kaggle.[https://www.kaggle.com/mathchi/diabetes-data-set](https://www.kaggle.com/mathchi/diabetes-data-set) 
This data is originally from the National Institute of Diabetes and Digestive and Kidney Disease.

# Build the Prediction Model

## Data Processing

```{r,warning=FALSE,message=FALSE}
diabetes <- read_csv("diabetes.csv")
head(diabetes)
```

![diabetes1](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/diabetes1.png)

#### Modify the Outcome data with str_replace() 

![스크린샷 2022-07-08 오후 9.01.25](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.01.25.png)As you can see here, 8 factors possibly affect the diagnosis of diabetes. Since the outcome is shown as 1 and 0, I will change this to Positive and Negative for easier visualization.

```{r}
diabetes$Outcome <- str_replace(diabetes$Outcome,"1","Positive")
diabetes$Outcome <- str_replace(diabetes$Outcome, "0","Negative")
diabetes
```

```{r}
class(diabetes$Outcome)
```

![스크린샷 2022-07-08 오후 9.01.41](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.01.41.png)In addition, currently, the outcome is in *character*. However, to make it more suitable in the tidy model, it is better to change into *factor*.

```{r}
diabetes <- diabetes %>% 
  mutate(Outcome = factor(Outcome))
class(diabetes$Outcome)
```

![스크린샷 2022-07-08 오후 9.02.22](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.02.22.png)

## Relationship between the variables and the Outcome

#### 1. Glucose and Diabetes

```{r,message=FALSE,warning=FALSE}
ggplot(diabetes, aes(x = Glucose, y = Outcome))+
  ggridges::geom_density_ridges()+
  labs(x = "Glucose Level",y = "Result of Diabetes", title = "Relationship between Glucose Level and Diabetes",subtitle = "Normal Glucose Level: <140 mg/dL",caption = "Kaggle")
```

![스크린샷 2022-07-08 오후 9.02.38](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.02.38.png)

As you can see in the graph, one of the most contributing factors to diabetes is the glucose level. Also, the glucose level is one of the indicative measures of diabetes. As the density plot suggests, the positive group tends to show a higher glucose range compared to the negative group.

#### 2. BMI and Diabetes

```{r}
ggplot(diabetes, aes(x = BMI, fill = Outcome))+
  geom_density(alpha = 0.4)+
  labs(x = "BMI Level",y = "Density",title = "Relationship between BMI and Diabetes",subtitle = "Normal BMI Range = 18.5 ~ 24.9",caption = "Kaggle")
```

![스크린샷 2022-07-08 오후 9.02.48](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.02.48.png)Also, obesity is one of the risk factors for diabetes. Even though the BMI is not an accurate measure of obesity (i.e. people with high muscle mass), BMI is often used to represent the weight status. This density plot indicates the positive group tends to have a higher BMI compared to the negative groups. Therefore, there are possible relationships that obesity affects the diabetes diagnosis.

#### 3. Insulin and the Diabetes

```{r,warning=FALSE,message=FALSE}
ggplot(diabetes, aes(x = Insulin, y = Outcome))+
  ggridges::geom_density_ridges()+
  labs(x = "Insulin Level",y = "Result of Diabetes", title = "Relationship between Insulin Level and Diabetes",caption = "Kaggle")
```

![스크린샷 2022-07-08 오후 9.03.24](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.03.24.png)

Insulin is the key component of diabetes. Insulin is required in our body because insulin uses glucose to generate energy. Also, insulin is the factor that differentiates diabetes. Type 1 Diabetic patients do not generate insulin, therefore they require insulin medications. Type 2 diabetic patients generate insulin but the insulin does not function well. This dataset did not differentiate the types of diabetes. However, the positive groups that have a higher level of insulin might be an indication of Type 2 Diabetic patients. 

## Bulid a Classification Tree Model

### Using BMI and Glucose only

```{r}
set.seed(11)
diabetes_split <- initial_split(diabetes, prop = 0.8)
diabetes_train <- training(diabetes_split)
diabetes_test <- testing(diabetes_split)
```

I used the 20:80 methods to divide the training and the testing set.

```{r}
diabetes_fit <- decision_tree() %>% 
  set_engine(engine = "rpart") %>% 
  set_mode(mode ="classification") %>% 
  fit(Outcome ~ Glucose+BMI, data = diabetes_train)
```

I used the *rpart* engine to build the classification model and the outcome is a categorical variable, I used classification as a mode to fit the model.


#### Classification Tree Model

```{r}
rpart.plot(diabetes_fit$fit, roundint = FALSE)
```

![스크린샷 2022-07-08 오후 9.03.47](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.03.47.png)

The first diverging point of the classification tree model is the most influencing variable of the model. The first division occurs based on the glucose level of 144. Even though the diagnosis of glucose varies based on the fasting levels, post-meal, etc., 144 is above the normal range. Also, the model demonstrates that the glucose level is more affecting the model than the BMI. 

#### Test the Model with Testing set

```{r}
diabetes_pred <- diabetes_fit %>% 
  predict(diabetes_test)

diabetes_test %>% select(Glucose,BMI,Outcome) %>% 
  mutate(Outcome_Pred = diabetes_pred$.pred_class)
```

#### ![스크린샷 2022-07-08 오후 9.03.59](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.03.59.png)Evaluate the Model

##### Heatmap

```{r}
diabetes_test %>% select(Glucose,BMI,Outcome) %>% 
  mutate(Outcome_Pred = diabetes_pred$.pred_class) %>% 
  conf_mat(truth = Outcome,estimate = Outcome_Pred) %>% 
  autoplot(type = "heatmap")
```

![스크린샷 2022-07-08 오후 9.04.24](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.04.24.png)

##### Accuracy

```{r}
diabetes_test %>% select(Glucose,BMI,Outcome) %>% 
  mutate(Outcome_Pred = diabetes_pred$.pred_class) %>% 
  accuracy(truth = Outcome,estimate = Outcome_Pred)
```

![스크린샷 2022-07-08 오후 9.04.36](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.04.36.png)

This model shows the 71% accuracy of the diabetic prediction when using 2 variables (BMI and Glucose).

### Using Pregnancy, Glucose, Blood Pressure, SkinThickness, Insulin, BMI, Diabetes Pedigree Function, Age 

```{r}
diabetes_all_fit <- decision_tree() %>% 
  set_engine(engine = "rpart") %>% 
  set_mode(mode ="classification") %>% 
  fit(Outcome ~ Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction+Age, data = diabetes_train)

rpart.plot(diabetes_all_fit$fit,roundint = FALSE)
```

![스크린샷 2022-07-08 오후 9.04.50](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.04.50.png)

Glucose is the most influencing factor in this case as well. Just like the previous model, a glucose level of 144 is the most influencing factor of the classification of the diabetic model. Also, BMI is the second leading factor of this model. It can be inferred that obesity is the possible leading risk factor of diabetes as well. Lastly, the diabetes pedigree function, which indicates the genetic factor, also plays a significant role in this model. To conclude, this model portrays various risk factors for diabetes. According to this model, it is important for some individual, who are obese, family history of diabetes, and who have the high glucose level to be cautious about the possibilities of diabetes. 

#### Evaluate the Model

##### HeatMap

```{r}
diabetes_all_pred <- diabetes_all_fit %>% 
  predict(diabetes_test)

diabetes_test %>% mutate(Outcome_Pred=diabetes_all_pred$.pred_class) %>% 
  conf_mat(truth = Outcome, estimate = Outcome_Pred) %>% 
  autoplot(type = "heatmap")
```

![스크린샷 2022-07-08 오후 9.05.02](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.05.02.png)

##### Accuracy

```{r}
diabetes_test %>% mutate(Outcome_Pred=diabetes_all_pred$.pred_class) %>% 
  accuracy(truth = Outcome, estimate = Outcome_Pred)
```

![스크린샷 2022-07-08 오후 9.05.36](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.05.36.png)

As you can see here, the accuracy went up to 75% when using all the variables listed in the dataset. Therefore, we can conclude that other factors may also affect the diabetes diagnosis.

## Using Random Forest Model for Prediction

```{r}
set.seed(11)
diabetes_rf <- rand_forest() %>% 
  set_engine(engine = "randomForest") %>% 
  set_mode(mode = "classification") %>% 
  fit(Outcome ~ Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction+Age, data = diabetes_train)

diabetes_rf_pred <- diabetes_rf %>% 
  predict(new_data = diabetes_test)
```

#### Evluate the Model

##### HeatMap

```{r}
diabetes_test %>% mutate(Outcome_pred = diabetes_rf_pred$.pred_class) %>% 
  conf_mat(truth = Outcome,estimate = Outcome_pred) %>% 
  autoplot(type = "heatmap")
```

![스크린샷 2022-07-08 오후 9.05.50](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.05.50.png)

##### Accuracy

```{r}
diabetes_test %>% mutate(Outcome_pred = diabetes_rf_pred$.pred_class) %>% 
  accuracy(truth = Outcome,estimate = Outcome_pred)
```
![스크린샷 2022-07-08 오후 9.06.01](/Users/cheolmin/Documents/blog/cheolminlee0907-github-blog/cheolminlee0907.github.io/images/2021-05-01-Diabetes-Prediction-Model/스크린샷 2022-07-08 오후 9.06.01.png)

The Random Forest model shows a 73.9% accuracy compared to the classification model. In this case, the classification model shows the better result from an accuracy standpoint. 

# Conclusion

There is an increasing number of diabetes cases. Surprisingly, some patients do not know that they have diabetes until they visit the hospital for other medical conditions. Disease deteriorates when the treatment gets delayed or detected in a later period. Building the precise prediction models may impact the early diagnose of the diseases and this is not limited to diabetes. Building the precise models may alleviate the conditions through early detection and this may be the new approach and a game-changer in the medical field. Finding the risk factors of the disease and establishing the relationship can be beneficial for the patients in the future.
