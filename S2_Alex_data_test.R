#----- PART 1: PRE-VIF RF ANALYSIS ----

# install required libraries
library(raster)
library(sf)
library(sp)
library(plotly)
library(caret)
library(randomForest)
library(data.table)
library(dplyr)
library(stringr)
library(doParallel)


# set the temporary folder for raster package operations (optional)
rasterOptions(tmpdir = "./cache/temp")

#upload of raster image and plotting the layers
raster_indices_masked  <- stack("raster_indices_masked.tif")

plot(raster_indices_masked) 

#importing the point dataset for RF
S2_RF_dataset <- st_read("samples_S2_14clas.shp")

#I create another column with a numerical ID for each class, which I use to later assign class codes to each 
S2_RF_dataset$id <- as.integer(factor(S2_RF_dataset$EUNIS_s))

#I extract the values of the 40 VIs for each sample point
dt_RF_dataset <- raster_indices_masked %>% 
  extract(y = S2_RF_dataset) %>% 
  as.data.table %>% 
  mutate(id = S2_RF_dataset$id)

#I assign a code to each ID
dt_RF_dataset$id[dt_RF_dataset$id == 1] <- "C2"
dt_RF_dataset$id[dt_RF_dataset$id == 2] <- "C3"
dt_RF_dataset$id[dt_RF_dataset$id == 3] <- "C4"
dt_RF_dataset$id[dt_RF_dataset$id == 4] <- "N1G"
dt_RF_dataset$id[dt_RF_dataset$id == 5] <- "C5"
dt_RF_dataset$id[dt_RF_dataset$id == 6] <- "C6"
dt_RF_dataset$id[dt_RF_dataset$id == 7] <- "C7"
dt_RF_dataset$id[dt_RF_dataset$id == 8] <- "S51"
dt_RF_dataset$id[dt_RF_dataset$id == 9] <- "C8"
dt_RF_dataset$id[dt_RF_dataset$id == 10] <- "T195"
dt_RF_dataset$id[dt_RF_dataset$id == 11] <- "T19B6"
dt_RF_dataset$id[dt_RF_dataset$id == 12] <- "T1E1"
dt_RF_dataset$id[dt_RF_dataset$id == 13] <- "T211"
dt_RF_dataset$id[dt_RF_dataset$id == 14] <- "T212"

#I save the codes as factor for the RF classification and remove the id column
dt_RF_dataset$EUNIS_2021 <- as.factor(dt_RF_dataset$id)

dt_RF_dataset <- dt_RF_dataset[,-c(41)]

#I divide the dataset between 70% training and 30% test
set.seed(321)

dataset_split <- createDataPartition(dt_RF_dataset$EUNIS_2021,p = 0.7,list = FALSE)

dt_train <- dt_RF_dataset[dataset_split]
dt_test <- dt_RF_dataset[-dataset_split]

#I use na.omit to avoid possible problems with na values later on
tr_fix <- na.omit(dt_train)
te_fix <- na.omit(dt_test)

# create cross-validation folds (splits the data into n random groups)
n_folds <- 10
set.seed(321)
folds <- createFolds(1:nrow(tr_fix), k = n_folds)
# Set the seed at each resampling iteration. Useful when running CV in parallel.
seeds <- vector(mode = "list", length = n_folds + 1) # +1 for the final model
for(i in 1:n_folds) seeds[[i]] <- sample.int(1000, n_folds)
seeds[n_folds + 1] <- sample.int(1000, 1) # seed for the final model

control <- trainControl(summaryFunction = multiClassSummary,
                        method = "cv",
                        number = n_folds,
                        search = "grid",
                        classProbs = TRUE, 
                        savePredictions = TRUE,
                        index = folds,
                        seeds = seeds)

#RANDOM FOREST
cluster <- makeCluster(3/4 * detectCores())
registerDoParallel(cluster)
model_rf <- caret::train(EUNIS_2021 ~ . , method = "rf", data = tr_fix,
                         importance = TRUE, # passed to randomForest()
                         # run CV process in parallel;
                         # see https://stackoverflow.com/a/44774591/5193830
                         allowParallel = TRUE,
                         tuneGrid = data.frame(mtry = c(5,6,13)), #I choose these values of mtry because,
                         #respectively, 5 is log base2 of 40, 6 is sqrt of 40 and 13 is one third;
                         #these three methods are rules of thumb to define the mtry value
                         trControl = control)
stopCluster(cluster); remove(cluster) #Unregister the doParallel cluster so that we can use sequential operations
#if needed; details at https://stackoverflow.com/a/25110203/5193830
registerDoSEQ()
saveRDS(model_rf, file = "model_S2_ALex_data_test.rds") #if you want to save the model output; optional

model_rf$times$everything #total computation time

plot(model_rf) #to see the difference in accuracy with the three mtry parameters; in this case, I obtain 
#the best results with mtry=13

#I get the top 10 most important variables in the RF prediction
importance_values <- randomForest::importance(model_rf$finalModel)

top_10_indices <- order(importance_values[, 1], decreasing = TRUE)[1:10]

top_10_names <- rownames(importance_values)[top_10_indices]

print(top_10_names)

# [1] "PSRI"  "ARI"   "CIRE"  "MNDVI" "EVI"   "MGRVI" "SIPI"  "CIG"   "SeLI"  "GNDVI"

#I keep only the 10 most signficant indexes
VarImport_indices <- c("PSRI" , "ARI" ,  "CIRE" , "MNDVI" ,"EVI" ,  "MGRVI", "SIPI",  "CIG" ,  "SeLI" , "GNDVI")
raster_indices_top10 <- raster_indices_masked[[VarImport_indices]]
plot(raster_indices_top10)

#converting raster into dataframe for VIF
raster_indices_top10_dt <- as.data.frame(raster_indices_top10, xy = TRUE)

raster_indices_top10_dt <- raster_indices_top10_dt[,-c(1,2)]

#na.omit to fix na values which may create problems later on
raster_indices_top10_dt_nafix <- na.omit(raster_indices_top10_dt)

#----- VIF ANALYSIS ----

#load the package necessary for VIF
library(usdm)

#trying VIF with th=5
VIF <- vifstep(x = raster_indices_top10_dt_nafix, th = 5, keep = NULL, method = 'pearson')
print(VIF)

#7 variables from the 10 input variables have collinearity problem: 
  
#  CIG GNDVI SeLI EVI MGRVI MNDVI PSRI 

#After excluding the collinear variables, the linear correlation coefficients ranges between: 
#  min correlation ( SIPI ~ CIRE ):  -0.3136835 
#max correlation ( CIRE ~ ARI ):  0.6847578 

#---------- VIFs of the remained variables -------- 
#  Variables      VIF
#1       ARI 1.993869
#2      CIRE 1.895362
#3      SIPI 1.174527

#trying again with th=10 (needs a check with litearute to justify it)
VIF_th10 <- vifstep(x = raster_indices_top10_dt_nafix, th = 10, keep = NULL, method = 'pearson')
print(VIF_th10)

#5 variables from the 10 input variables have collinearity problem: 
  
#  CIG GNDVI SeLI EVI MGRVI 

#After excluding the collinear variables, the linear correlation coefficients ranges between: 
#  min correlation ( SIPI ~ CIRE ):  -0.5106298 
#max correlation ( MNDVI ~ CIRE ):  0.9226633 

#---------- VIFs of the remained variables -------- 
#  Variables      VIF
#1      PSRI 7.206892
#2       ARI 3.487081
#3      CIRE 7.082065
#4     MNDVI 9.981227
#5      SIPI 1.873521


