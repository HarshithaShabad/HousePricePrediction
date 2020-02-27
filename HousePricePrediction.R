#install.packages("nnet")

rm(list = ls())

library(randomForest) 
library(neuralnet)
library(tidyverse)
library(forcats)
library(kernlab)
library(ggplot2) 
library(xgboost)
library(rpart)
library(tidyr)
library(dplyr) 
library(psych) 
library(caret)
library(plyr)
library(nnet)




#setwd("/Users/sayalikardile/Downloads/Big Data Analytics")
setwd("C:/CIS8392/data")


train = read.csv("train.csv", stringsAsFactors = FALSE)
test = read.csv("test.csv", stringsAsFactors = FALSE)

#combine train and test data, clean them together
index_train <- train$Id
train$Id <- NULL
test$Id <- NULL

#create the SalePrice variable in test dataset
test$SalePrice <- NA
total <- rbind(train, test)

#see how many missing data do we have in each column
withNA <- which(colSums(is.na(total)) > 0)
sort(colSums(sapply(total[withNA], is.na)), decreasing = TRUE)


#some NA means the house don't have that feature, so remove NAs in these selected column, replace them with "none"

change <- c("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageFinish", "GarageQual", 
            "GarageCond", "GarageType", "BsmtCond", "BsmtExposure", "BsmtQual","BsmtFinType1",
            "BsmtFinType2", "BsmtFinType")

min(train$SalePrice)

i = 1
for(i in 1:dim(total)[2]){
  if(colnames(total[i]) %in% change == TRUE){
    total[,colnames(total[i])][is.na(total[,colnames(total[i])])] <- "None"
    i = i + 1
  }else{
    i = i + 1
  }
}

rm(change, i)

#check NAs again
withNA <- which(colSums(is.na(total)) > 0)
sort(colSums(sapply(total[withNA], is.na)), decreasing = TRUE)



#check NA in GarageType
total_clean <- total
total_clean[total_clean$GarageFinish == "None" & total_clean$GarageType != "None" ,
            c ("GarageFinish","GarageQual","GarageCond","GarageType")]

#replace with "None"
total_clean[total_clean$GarageFinish == "None" & total_clean$GarageType != "None" ,
            "GarageType"] <- "None"

#check NA in column GarageCars, GarageArea and GarageYrBlt
total_clean %>%
  filter(GarageType == "None") %>%
  filter(GarageCars!=0, GarageArea!=0) %>%
  filter(!is.na(GarageCars) |!is.na(GarageArea) | !is.na(GarageYrBlt)) %>%
  dplyr::select(GarageCars,GarageArea,GarageYrBlt, GarageType,YearBuilt)


# Replace NA GaraType with average type shows in data

#find the average type
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}


Mode(total_clean$GarageType)

#replace
total_clean[total_clean$GarageType == "None" & total_clean$GarageCars == 1 & 
              total_clean$GarageArea == 360 & total_clean$YearBuilt == 1910, 
            "GarageType"] <- Mode(total_clean$GarageType) 
#check NAs
total_clean %>%
  filter(is.na(GarageYrBlt) == TRUE ) %>%
  filter(GarageArea != "None", GarageCars != 0, GarageFinish != "None", 
         GarageQual != "None", GarageArea != 0)


#modyfy Garage Yearl Built column "GarageYrBlt", use the year the house was build. 
# If there was a remodeled record, use the remodeled year instead.

for (i in 1:nrow(total_clean)){
  if(is.na(total_clean$GarageYrBlt[i])){
    total_clean$GarageYrBlt[i] <- ifelse(!is.na(total_clean$YearRemodAdd[i]),total_clean$YearRemodAdd[i],total_clean$YearBuilt[i])    
    i = i + 1
  }else{
    i = i + 1
  }
}

#check NAs
total_clean %>%
  filter(is.na(GarageCars) | is.na(GarageArea)) %>%
  dplyr::select("GarageCars","GarageArea","GarageFinish","GarageQual","GarageCond","GarageType")

#Impute 0 for house dont have garage
total_clean$GarageCars[is.na(total_clean$GarageCars)] <- 0
total_clean$GarageArea[is.na(total_clean$GarageArea)] <- 0

#Doing data cleaning for basement related columns

#find NAs, replace with average type
total_clean %>%
  dplyr::select(BsmtCond,BsmtExposure,BsmtQual,BsmtFinType2,BsmtFinType1) %>%
  filter(BsmtFinType1 != "None") %>%
  filter(BsmtCond == "None" | BsmtExposure == "None" | BsmtQual == "None" | BsmtFinType2 == "None" )


total_clean[total_clean$BsmtFinType1 != "None" & total_clean$BsmtCond == "None", "BsmtCond"] <- Mode(total_clean$BsmtCond)

total_clean[total_clean$BsmtFinType1 != "None" & total_clean$BsmtExposure == "None", "BsmtExposure"] <- Mode(total_clean$BsmtExposure)

total_clean[total_clean$BsmtFinType1 != "None" & total_clean$BsmtQual == "None", "BsmtQual"] <- Mode(total_clean$BsmtQual)

total_clean[total_clean$BsmtFinType1 != "None" & total_clean$BsmtFinType2 == "None", "BsmtFinType2"] <- Mode(total_clean$BsmtFinType2)

total_clean[(is.na(total_clean$BsmtFullBath)|is.na(total_clean$BsmtHalfBath)|
               is.na(total_clean$BsmtFinSF1)|is.na(total_clean$BsmtFinSF2)|
               is.na(total_clean$BsmtUnfSF)|is.na(total_clean$TotalBsmtSF)),
            c("BsmtQual", "BsmtFullBath", "BsmtHalfBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", 
              "TotalBsmtSF")]

#modify row 2121 and 2189
total_clean[2121, c("BsmtFullBath", "BsmtHalfBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                    "TotalBsmtSF")] <- 0
total_clean[2189, c("BsmtFullBath", "BsmtHalfBath")] <- 0

#Do the same for Masonry columns
total_clean[is.na(total_clean$MasVnrType) & !is.na(total_clean$MasVnrArea), 
            c("MasVnrType","MasVnrArea")]

total_clean[is.na(total_clean$MasVnrType) & !is.na(total_clean$MasVnrArea), 
            "MasVnrType"] <- Mode(total_clean$MasVnrType[total_clean$MasVnrType!="None"])

total_clean$MasVnrType[is.na(total_clean$MasVnrType)] <- "None"
total_clean$MasVnrArea[is.na(total_clean$MasVnrArea)] <- 0

#same for Pool
total_clean[total_clean$PoolQC == "None" & total_clean$PoolArea>0, 
            c("PoolQC","PoolArea","OverallQual")]

table(total_clean$PoolQC,total_clean$OverallQual)
chisq.test(total_clean$PoolQC , total_clean$OverallQual)

total_clean$PoolQC[2421] <- "Fa"
total_clean$PoolQC[2504] <- "TA"
total_clean$PoolQC[2600] <- "Fa"

#and Lot
summary(aov(LotFrontage ~ Neighborhood, total_clean))

ggplot(total_clean, aes(Neighborhood,  LotFrontage)) + geom_bar(stat = "summary", 
                                                                fun.y = "median") + 
  coord_flip() + labs(title="LotFrontage median by neighborhood", x="")

for(i in 1:length(total_clean$MSSubClass)){
  if(is.na(total_clean$LotFrontage[i])){
    total_clean$LotFrontage[i] <- median(total_clean$LotFrontage[total_clean$Neighborhood == total_clean$Neighborhood[i]], na.rm=TRUE)
    i = i + 1
  }else{
    i = i + 1
  }
}

#and zoning

table(total_clean$Neighborhood,total_clean$MSZoning)
total_clean[is.na(total_clean$MSZoning), c("MSZoning", "Neighborhood")]

total_clean$MSZoning[c(1916,2217,2251)] <- "RM"
total_clean$MSZoning[2905] <- "RL"

#Exterior
total_clean[is.na(total_clean$Exterior1st)|is.na(total_clean$Exterior2nd), 
            c("Exterior1st","Exterior2nd","ExterCond","ExterQual","Neighborhood")]
table(total_clean$Exterior1st, total_clean$Exterior2nd)[1:10,1:10]

table(total_clean$Exterior1st[total_clean$Neighborhood == "Edwards"], 
      total_clean$ExterQual[total_clean$Neighborhood == "Edwards"])

total_clean[is.na(total_clean$Exterior1st)|is.na(total_clean$Exterior2nd), 
            c("Exterior1st","Exterior2nd")] <- "Wd Sdng"

#check remaining variables, seems all variables are same, drop column
summary(as.factor(total_clean$Utilities))


total_clean$Utilities <- NULL

#Assume typical unless deductions are warranted, so replace NAs with "typ"
total_clean$Functional[is.na(total_clean$Functional)] <- "Typ"
#Electrical, major of them are "SBrkr", use that value for all rows
summary(as.factor(total_clean$Electrical))
total_clean$Electrical[is.na(total_clean$Electrical)] <- "SBrkr"
#Use overall quality to replace NAs in Kitchen Quality
total_clean[is.na(total_clean$KitchenQual), c("OverallQual")]
total_clean$KitchenQual[is.na(total_clean$KitchenQual)] <- "TA"
#Use average type to ereplace NAs in Sale Type
total_clean$SaleType[is.na(total_clean$SaleType)] <- Mode(total_clean$SaleType)

withNA3 <- which(colSums(is.na(total_clean)) > 0)
sort(colSums(sapply(total_clean[withNA3], is.na)), decreasing = TRUE)

#Data Encoding
#standarized ordinal values  -> not done as error








#nominal variable
#transfer character type to factor, prepare for ML
character <- which(sapply(total_clean, is.character))

for(i in 1:ncol(total_clean)){
  x = colnames(total_clean[i])
  if(i %in% character == TRUE){
    total_clean[i] <- as.factor(total_clean[,x])
    i = i + 1
  }else{
    i = i + 1
  }
}

total_clean$SalePrice <- as.numeric(total_clean$SalePrice)

which(sapply(total_clean, is.character))

#data structure
str(total_clean)

#month and year should in factors
total_clean$MoSold <- as.factor(total_clean$MoSold)

#Age of the house
#We convert the variable into numeric
total_clean$Age <- total_clean$YrSold - total_clean$YearRemodAdd

#We convert the variable back to factor
total_clean$YrSold <- as.factor(total_clean$YrSold)


#plot price VS house age
ggplot(total_clean, aes(Age, SalePrice)) +
  geom_point(alpha=0.4) + geom_smooth(method = "lm", se=FALSE) +
  labs(title="House price by year", x ="House Age")


#add Total Square feet column
total_clean$TotalSF <- total_clean$GrLivArea + total_clean$TotalBsmtSF +
  total_clean$OpenPorchSF + total_clean$WoodDeckSF + total_clean$EnclosedPorch +
  total_clean$X3SsnPorch + total_clean$ScreenPorch + total_clean$PoolArea

ggplot(total_clean, aes(TotalSF, SalePrice)) + geom_point(alpha=0.4) +
  geom_smooth(method = "loess", se=FALSE) + labs(title="Houseprice by total square feet")

cor(total_clean$TotalSF,total_clean$SalePrice, use = "complete.obs" )

total_clean[total_clean$TotalSF>7500, 
            c("GrLivArea","TotalBsmtSF","OpenPorchSF","WoodDeckSF","PoolArea","SalePrice") ]

t(total_clean[c(524,1299,2550),])

total_clean <- total_clean[-c(524,1299),]

ggplot(total_clean, aes(TotalSF, SalePrice)) +
  geom_point(alpha=0.4) + geom_smooth(method = "loess", se=FALSE) +
  labs(title="Houseprice by total square feet (adjusted)")

cor(total_clean$TotalSF,total_clean$SalePrice, use = "complete.obs" )

#Number of Batrhooms
total_clean$Bathrooms <- (total_clean$BsmtHalfBath + total_clean$HalfBath)*0.5 +
  total_clean$FullBath + total_clean$BsmtFullBath   

ggplot(total_clean, aes(Bathrooms, SalePrice)) + geom_jitter(alpha=0.4) +
  geom_smooth(method = "loess", se=FALSE) + scale_x_continuous(breaks=seq(1,6,0.5)) +
  labs(title = "Houseprice by Amount of bathrooms")

cor(total_clean$Bathrooms, total_clean$SalePrice, use = "complete.obs")



# We create a new df for numeric variables
total_clean_numeric_index <- names(which(sapply(total_clean, is.numeric)))
total_clean_numeric <- total_clean[, which(names(total_clean) %in% total_clean_numeric_index)]

#We calculate correlations and filter for just  those higher than |0.5| and pick the names
correlations <- as.matrix(x = sort(cor(total_clean_numeric, 
                                       use="pairwise.complete.obs")[,"SalePrice"], 
                                   decreasing = TRUE))
names <- names(which(apply(correlations,1, function(x) abs(x)>0.5))) 

#We sort the dataset to just show those variables
total_clean_numeric <- total_clean_numeric[, names]

#We create and represent the correlations matrix
correlations <- cor(total_clean_numeric, use="pairwise.complete.obs")
cor.plot(correlations, numbers=TRUE, xlas = 2, upper= FALSE,
         main="Correlations among important variables", zlim=c(abs(0.65),abs(1)), colors=FALSE)
#DF with imp variables
important <- c("SalePrice","TotalSF","OverallQual","Bathrooms","GarageCars","YearBuilt","BsmtQual","GarageFinish","GarageYrBlt","FireplaceQu","YearRemodAdd", "Neighborhood")

total_models_clean <- total_clean[, which(names(total_clean) %in% important)]

#Data Preparation

#Treating Skewness
numeric <- c("SalePrice","TotalSF","YearBuilt","GarageYrBlt","YearRemodAdd")
total_model_clean_numeric <- total_models_clean[,names(total_models_clean)%in% numeric]

for(i in 1:ncol(total_model_clean_numeric)){
  if (abs(skew(total_model_clean_numeric[,i]))>0.75){
    total_model_clean_numeric[,i] <- log(total_model_clean_numeric[,i] +1)
  }
}


head(total_model_clean_numeric,10)

#skewness of data with out transformation
qqnorm(total_clean$SalePrice, main = "Skewness of data without transformation (SalePrice)")
qqline(total_clean$SalePrice)

#skewness of transformed data
qqnorm(total_model_clean_numeric$SalePrice, main = "Skewness of data transformed (SalePrice)")
qqline(total_model_clean_numeric$SalePrice)


##Normalizations and standarization
SalePrice <- total_model_clean_numeric$SalePrice
total_model_clean_numeric <- sapply(total_model_clean_numeric[, -which(names(total_model_clean_numeric) %in% "SalePrice")], scale)

total_model_clean_numeric <- cbind(SalePrice,total_model_clean_numeric)
total_model_clean_numeric <- as.data.frame(total_model_clean_numeric)

##Variable Dummification

#coerce categorical variables
total_models_clean$OverallQual <- as.factor(total_models_clean$OverallQual)
total_models_clean$BsmtQual <- as.factor(total_models_clean$BsmtQual)
total_models_clean$FireplaceQu <- as.factor(total_models_clean$FireplaceQu)
total_models_clean$GarageFinish <- as.factor(total_models_clean$GarageFinish)
total_models_clean$GarageCars <- as.factor(total_models_clean$GarageCars)
total_models_clean$Bathrooms <- as.factor(total_models_clean$Bathrooms)

#Categorical variables to dummy variables
categorical <- total_models_clean[!colnames(total_models_clean) %in% numeric]
dummy <- as.data.frame(model.matrix(~.-1, categorical))
total_final_2 <- cbind(SalePrice = total_model_clean_numeric$SalePrice, dummy)

#get rid off the variables that appear less than 5 times in the train dataset along with others that do not appear in the test dataset.
names1 <- names(colSums(total_final_2[!is.na(total_final_2$SalePrice),])[colSums(total_final_2[!is.na(total_final_2$SalePrice),])<5])

names2 <- names(colSums(total_final_2[is.na(total_final_2$SalePrice), -which(names(total_final_2) == "SalePrice") ])[colSums(total_final_2[is.na(total_final_2$SalePrice),-which(names(total_final_2) == "SalePrice")])==0])

delete <- c(names1,names2)

#get rid off these columns and bind them with the normalized variables.
total_final_2 <- total_final_2[, -which(names(total_final_2)%in% c(delete, "SalePrice"))]
total_final_2 <- cbind(total_model_clean_numeric, total_final_2)


#split data into train and test, and the response and explanatory variables.
train_2 <- total_final_2[!is.na(total_final_2$SalePrice),]
train_2_explanatory <- train_2[, -which(names(train_2) %in% "SalePrice")]
train_2_response <- train_2[, which(names(train_2) %in% "SalePrice")]



length(train_2[[1]])

#split dataset
which_train <- sample(x = c(TRUE, FALSE), size = nrow(train_2),
                      replace = TRUE, prob = c(0.8, 0.2))

recc_data_train <- train_2[which_train, ]
recc_data_valid <- train_2[!which_train, ]

length(recc_data_train[[1]])
length(recc_data_valid[[1]])



test_2 <- total_final_2[is.na(total_final_2$SalePrice),]
test_2_explanatory <- test_2[, -which(names(test_2) %in% "SalePrice")]



###
predictors<-c("YearBuilt","YearRemodAdd","GarageYrBlt","TotalSF",
              "NeighborhoodBlmngtn","NeighborhoodBrDale","NeighborhoodBrkSide",
              "NeighborhoodClearCr","NeighborhoodCollgCr","NeighborhoodCrawfor",
              "NeighborhoodEdwards","NeighborhoodGilbert","NeighborhoodIDOTRR",
              "NeighborhoodMeadowV","NeighborhoodMitchel","NeighborhoodNAmes",
              "NeighborhoodNoRidge","NeighborhoodNPkVill","NeighborhoodNridgHt",
              "NeighborhoodNWAmes","NeighborhoodOldTown","NeighborhoodSawyer",
              "NeighborhoodSawyerW","NeighborhoodSomerst","NeighborhoodStoneBr",
              "NeighborhoodSWISU","NeighborhoodTimber","NeighborhoodVeenker",
              "OverallQual3","OverallQual4","OverallQual5","OverallQual6",
              "OverallQual7","OverallQual8","OverallQual9","OverallQual10","BsmtQualFa",
              "BsmtQualGd","BsmtQualNone","BsmtQualTA","FireplaceQuFa","FireplaceQuGd","FireplaceQuNone","FireplaceQuPo",
              "FireplaceQuTA","GarageFinishNone","GarageFinishRFn","GarageFinishUnf","GarageCars1","GarageCars2","GarageCars3","GarageCars4","Bathrooms1.5",
              "Bathrooms2","Bathrooms2.5","Bathrooms3","Bathrooms3.5","Bathrooms4","Bathrooms4.5")

outcomeName<-c("SalePrice")
#######################################################################


write.csv(recc_data_train, file = "recc_data_train.csv", row.names = FALSE)
write.csv(recc_data_valid, file = "recc_data_valid.csv", row.names = FALSE)

recc_data_train= read.csv("recc_data_train.csv", stringsAsFactors = FALSE)
recc_data_valid= read.csv("recc_data_valid.csv", stringsAsFactors = FALSE)





# XGBoost

tune_grid <- expand.grid(nrounds = 10,
                         eta = c(0.1, 0.05, 0.01),
                         max_depth = c(2, 3, 4, 5, 6),
                         gamma = 0,
                         colsample_bytree=1,
                         min_child_weight=c(1, 2, 3, 4 ,5),
                         subsample=1)

#xgb_tune_2$bestTune
xgb_tune_2 <- caret::train(x = recc_data_train[,-which(names(recc_data_train)=="SalePrice")],
                           y = recc_data_train$SalePrice,
                           tuneGrid = tune_grid,
                           method = "xgbTree",
                           verbose = TRUE)


prediction_2_xgb <-predict(xgb_tune_2,recc_data_valid)
prediction_2_xgb <- exp(prediction_2_xgb)
#prediction_2_xgb <- data.frame(Id=(1461:2919), SalePrice=prediction_2_xgb)
write.csv(prediction_2_xgb,"model_xgb_2.csv", quote=FALSE,row.names = FALSE)


accuracy(as.vector(prediction_2_xgb), exp(recc_data_valid$SalePrice))





# Lasso

my_control <-trainControl(method="cv", number=5)
lassoGrid <- expand.grid(alpha = 1, lambda = seq(0,1,by = 0.1))




model_2_glml <- train( 
  x = recc_data_train[,-which(names(recc_data_train)=="SalePrice")],
  y = recc_data_train$SalePrice, method='glmnet',
  trControl= my_control, tuneGrid=lassoGrid) 

model_2_glml$bestTune


prediction_2_glml <- predict(model_2_glml,recc_data_valid)
#prediction_2_glml <- data.frame(Id = (1461:2919), SalePrice = exp(prediction_2_glml))
write.csv(prediction_2_glml,"model_2_glml.csv", quote=FALSE,row.names = FALSE)


accuracy(as.vector(exp(prediction_2_glml)), exp(recc_data_valid$SalePrice))




# SVM

model_svm_2 <- train(SalePrice~., data=recc_data_train, 
                     method = "svmLinear",trControl=my_control,
                     tuneLength = 10)
prediction_2_svm <-predict(model_svm_2,recc_data_valid)

#prediction_2_svm <- data.frame(Id=(1461:2919), SalePrice=exp(prediction_2_svm))

write.csv(prediction_2_svm,"model_svm_2.csv", quote=FALSE,row.names = FALSE)

accuracy(as.vector(exp(prediction_2_svm)), exp(recc_data_valid$SalePrice))



#KNN


knn_ctrl <- trainControl(method="repeatedcv",repeats = 3) 
prePro_2 <- preProcess(recc_data_train, method = c("center", "scale"))
norm_2 <- predict(prePro_2, recc_data_train)
model_knn_2 <- train(SalePrice ~ ., data = norm_2, method = "knn", trControl = knn_ctrl, tuneLength = 20)


prediction_2_knn <-predict(model_knn_2,recc_data_valid)

write.csv(prediction_2_knn,"model_knn_2.csv", quote=FALSE,row.names = FALSE)





#neural network
library(neuralnet)







#write.csv(recc_data_train, file = "recc_data_train.csv", row.names = FALSE)
#write.csv(recc_data_valid, file = "recc_data_valid.csv", row.names = FALSE)

#recc_data_train= read.csv("recc_data_train.csv", stringsAsFactors = FALSE)
#recc_data_valid= read.csv("recc_data_valid.csv", stringsAsFactors = FALSE)


trainnn <- recc_data_train
testnn <- recc_data_valid



length(trainnn[[1]])
length(testnn[[1]])

train_o <- trainnn


allVars <- colnames(trainnn)
predictorVars <- allVars[!allVars%in%"SalePrice"]
predictorVars <- paste(predictorVars, collapse = "+")
form = as.formula(paste("SalePrice~", predictorVars, collapse = "+"))

# Prediction Model
nn_model <- neuralnet(formula = form, trainnn, hidden = c(4,2), linear.output = TRUE)

# the fitted values i.e. weights
nn_model$net.result
plot(nn_model)

#PREDICTION
prediction1 <- compute(nn_model, testnn)
str(prediction1)


ActualPrediction <-  prediction1$net.result 
table(ActualPrediction)

write.csv(ActualPrediction, file = "model_neural_2.csv", row.names = FALSE)





# Ensemble model






svm = exp(read.csv("model_svm_2.csv", stringsAsFactors = FALSE))
lasso = exp(read.csv("model_2_glml.csv", stringsAsFactors = FALSE))
xgb = read.csv("model_xgb_2.csv", stringsAsFactors = FALSE)
nn= exp(read.csv("model_neural_2.csv", stringsAsFactors = FALSE))




accuracy(as.vector(exp(prediction_2_svm)), exp(recc_data_valid$SalePrice))
accuracy(as.vector(exp(prediction_2_glml)), exp(recc_data_valid$SalePrice))
accuracy(as.vector(prediction_2_xgb), exp(recc_data_valid$SalePrice))
accuracy(as.vector(exp(ActualPrediction)), exp(recc_data_valid$SalePrice))


mean(exp(recc_data_valid$SalePrice))


recc_data_valid$pred_avg<-(svm$x+lasso$x+xgb$x+nn$V1)/4

accuracy(as.vector(recc_data_valid$pred_avg), exp(recc_data_valid$SalePrice))


#apply weight

recc_data_valid$pred_weight<-(svm$x*0.45+lasso$x*0.45+xgb$x*0.05+nn$V1*0.05)

accuracy(as.vector(recc_data_valid$pred_weight), exp(recc_data_valid$SalePrice))























############################################################################
#logistic regression part, to see the estimate


bank.df <- read.csv("recc_data_train.csv")
bank.df$SalePrice<-exp(bank.df$SalePrice)


# Select variables
selected.var <- c(1:60)

# partition data
set.seed(2)
train.index <- sample(c(1:dim(bank.df)[1]), dim(bank.df)[1]*0.6)
train.df <- bank.df[train.index, selected.var]
valid.df <- bank.df[-train.index, selected.var]




# use lm() to run a linear regression of Price on all 11 predictors in the training set. 
car.lm <- lm(SalePrice ~ ., data = train.df)
#  use options() to ensure numbers are not displayed in scientific notation.
options(scipen = 999)
table1<-summary(car.lm)   # Get model summary



