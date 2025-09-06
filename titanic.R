


###################################################
# Logistic Regression - Titanic Dataset
###################################################

# Load Libraries
library(readr)
library(naniar)
library(mice)
library(car)
library(tidymodels)
library(pROC)
library(ResourceSelection)
library(pscl)
library(nortest)
library(corrplot)

###################################################
# Load & Explore Data
###################################################
Titanic <- read_csv("C:/Users/beven/Downloads/teams/Titanic.csv")

# Initial Exploration
head(Titanic)
str(Titanic)
summary(Titanic)
Titanic$Sex    <- as.factor(Titanic$Sex)
Titanic$Pclass <- as.factor(Titanic$Pclass)
Titanic$Embarked <- as.factor(Titanic$Embarked)


# Check Age <= 1
Titanic$Age[Titanic$Age <= 1]

# Check duplicates and missing values
sum(duplicated(Titanic))
colSums(is.na(Titanic))
nrow(Titanic)
ncol(Titanic)

# Remove Cabin column (10th col)
n_titanic <- Titanic[ , -10]
head(n_titanic)
str(n_titanic)
colSums(is.na(n_titanic))

# Percentage missing (Age)
177/891 * 100


###################################################
# Missing Data Analysis
###################################################
# MCAR Test
mcar_test(Titanic)

# Missing indicators
data <- n_titanic
data$Age_missing      <- ifelse(is.na(data$Age), 1, 0)
data$Embarked_missing <- ifelse(is.na(data$Embarked), 1, 0)

# Chi-square association with missingness
chisq.test(table(data$Age_missing, data$Pclass))
chisq.test(table(data$Age_missing, data$Sex))
chisq.test(table(data$Age_missing, data$SibSp))
chisq.test(table(data$Age_missing, data$Parch))
chisq.test(table(data$Age_missing, data$Fare))
chisq.test(table(data$Age_missing, data$Embarked))


###################################################
# Multiple Imputation (MICE)
###################################################
# Convert categorical variables to factors
n_titanic$Sex      <- as.factor(n_titanic$Sex)
n_titanic$Embarked <- as.factor(n_titanic$Embarked)

# Define imputation methods
meth <- make.method(n_titanic)
meth["Age"]      <- "pmm"      # numeric
meth["Embarked"] <- "polyreg"  # categorical
meth["Sex"]      <- "logreg"   # binary categorical

# Run MICE
imp <- mice(n_titanic, method = meth, m = 5, seed = 123)

# Get completed data
completed_data <- complete(imp)
colSums(is.na(completed_data))

completed_data_n <- completed_data[ , -10]
str(completed_data_n)


###################################################
# Median Imputation (for skewed Age)
###################################################
median(n_titanic$Age, na.rm = TRUE)

n_titanic$Embarked=as.character(n_titanic$Embarked)
# Imputation: Age (median), Embarked (mode)
imputed_data <- n_titanic %>%
  replace_na(list(
    Age      = median(n_titanic$Age, na.rm = TRUE),
    Embarked = mode(n_titanic$Embarked)
  ))

head(imputed_data)
colSums(is.na(imputed_data))


###################################################
# Logistic Regression Model
###################################################
# Fit model
model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data = imputed_data,
             family = binomial)
summary(model)

# Backward Selection
backward_model <- stats::step(model, direction = "backward")
summary(backward_model)

# Odds Ratios with 95% CI
exp(cbind(OR = coef(backward_model), confint(backward_model)))


###################################################
# Train-Test Split
###################################################
set.seed(123)
data_split <- initial_split(imputed_data, prop = 0.8, strata = Survived)

train_data <- training(data_split)
test_data  <- testing(data_split)

# Train model
model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data = train_data,
             family = binomial)
summary(model)

# Backward Selection
backward_model <- stats::step(model, direction = "backward")
summary(backward_model)

# Predictions
pred_probs  <- predict(backward_model, newdata = test_data, type = "response")
pred_class  <- ifelse(pred_probs > 0.5, 1, 0)

# Confusion Matrix
table(Predicted = pred_class, Actual = test_data$Survived)


###################################################
# Diagnostics
###################################################
# Frequency table
table(imputed_data$Survived)

# Multicollinearity (VIF)
vif(backward_model)



# Influence diagnostics
plot(backward_model, which = 4)  # Cook's distance
plot(backward_model, which = 1)  # Residuals vs fitted

nrow(imputed_data$Survived)
###################################################
# ROC Curve & AUC
###################################################
library(pROC)

# Get predicted probabilities for class = 1
pred_probs <- predict(backward_model, newdata = imputed_data, type = "response")

# Now both are length 891
roc_curve <- roc(imputed_data$Survived, pred_probs)

plot(roc_curve, col = "blue")
auc(roc_curve)


###################################################
# Hosmer-Lemeshow Test
###################################################

hoslem.test(train_data$Survived, fitted(backward_model))

# Likelihood ratio test
anova(backward_model, test = "Chisq")

# Pseudo R-squared
pR2(backward_model)


###################################################
# Influence Diagnostics - Cook's Distance
###################################################
cooks_d <- cooks.distance(backward_model)
cutoff  <- 4 / nrow(imputed_data)
influential_points <- which(cooks_d > cutoff)

cat("Number of influential points removed:", length(influential_points), "\n")

# Remove influential points
clean_data <- imputed_data[-influential_points, ]


###################################################
# Refit Model on Clean Data
###################################################
model_clean <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                   data = clean_data,
                   family = binomial)

# Stepwise Backward Selection
backward_model <- stats::step(model_clean, direction = "backward")
summary(backward_model)

# Train-test split (clean data)
set.seed(123)
data_split <- initial_split(clean_data, prop = 0.8, strata = Survived)

train_data <- training(data_split)
test_data  <- testing(data_split)

model <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
             data = train_data,
             family = binomial)
summary(model)

backward_model <- stats::step(model, direction = "backward")
summary(backward_model)

# Predictions
pred_probs <- predict(backward_model, newdata = test_data, type = "response")
pred_class <- ifelse(pred_probs > 0.5, 1, 0)

# Confusion Matrix
y <- table(Predicted = pred_class, Actual = test_data$Survived)
y


###################################################
# Odds Ratios (Clean Data)
###################################################
exp(cbind(OR = coef(backward_model), confint(backward_model)))


###################################################
# ROC Curve & AUC (Clean Data)
###################################################

library(pROC)

# Get predicted probabilities for class = 1
pred_probs <- predict(backward_model, newdata = clean_data, type = "response")

# Now both are length 891
roc_curve <- roc(clean_data$Survived, pred_probs)

plot(roc_curve, col = "blue")
auc(roc_curve)




###################################################
# Hosmer-Lemeshow Test (Clean Data)
###################################################
hoslem.test(train_data$Survived, fitted(backward_model))

# Pseudo R-squared
pR2(backward_model)


###################################################
# Normality Checks - Kolmogorov-Smirnov Test
###################################################
# Age
ks.test(imputed_data$Age, "pnorm", 
        mean=mean(imputed_data$Age, na.rm=TRUE), 
        sd=sd(imputed_data$Age, na.rm=TRUE))

# Fare
ks.test(imputed_data$Fare, "pnorm", 
        mean=mean(imputed_data$Fare, na.rm=TRUE), 
        sd=sd(imputed_data$Fare, na.rm=TRUE))

# Loop for continuous vars
cont_vars <- c("Age", "Fare")
for (v in cont_vars) {
  cat("\nK-S test for", v, ":\n")
  print(
    ks.test(imputed_data[[v]], "pnorm", 
            mean=mean(imputed_data[[v]], na.rm=TRUE), 
            sd=sd(imputed_data[[v]], na.rm=TRUE))
  )
}


###################################################
# Visualizations
###################################################
# Histogram - Age
hist(Titanic$Age, main="Age Distribution", xlab="Age", 
     col="skyblue", border="white")

# Histogram - Fare
hist(Titanic$Fare, main="Fare Distribution", xlab="Fare", 
     col="lightgreen", border="white", breaks=50)

# Boxplot - Age by Survival
boxplot(Age ~ Survived, data=Titanic,
        main="Age by Survival", xlab="Survived (0=No, 1=Yes)", 
        ylab="Age", col=c("tomato", "lightblue"))

# Boxplot - Fare by Class
boxplot(Fare ~ Pclass, data=Titanic,
        main="Fare by Passenger Class", xlab="Class", ylab="Fare",
        col=c("orange","lightgreen","lightblue"), log="y")

# Correlation Plot
# Select only numeric columns
numeric_vars <- Titanic[, sapply(Titanic, is.numeric)]

# Compute correlation matrix
corr_matrix <- cor(numeric_vars, use = "complete.obs")

# Load library
library(corrplot)

# Plot correlation heatmap with numbers
corrplot(corr_matrix,
         method = "color",        # colored squares
         type = "full",           # full matrix
         addCoef.col = "black",   # add correlation numbers in black
         tl.col = "black",        # text label color
         tl.srt = 45,             # text label rotation
         number.cex = 0.8)        # size of correlation numbers

# Pie chart - Survival
survived_counts <- table(Titanic$Survived)
pie(survived_counts,
    labels=c("Did Not Survive","Survived"),
    main="Survival Distribution",
    col=c("red","green"))

# Pie chart - Sex
sex_counts <- table(Titanic$Sex)
pie(sex_counts,
    labels=paste(names(sex_counts)," (",sex_counts,")",sep=""),
    main="Gender Distribution",
    col=c("lightblue","pink"))

library(ggplot2)

# Clustered bar plot
ggplot(Titanic, aes(x = Sex, fill = factor(Survived))) +
  geom_bar(position = "dodge") +
  labs(title = "Survival by Sex",
       x = "Sex",
       y = "Count",
       fill = "Survived") +
  scale_fill_manual(values = c("0" = "red", "1" = "green"),
                    labels = c("0" = "Did Not Survive", "1" = "Survived")) +
  theme_minimal()



###################################################
# Residual Analysis
###################################################
# Deviance residuals
dev_resid <- residuals(model, type="deviance")
plot(dev_resid, main="Deviance Residuals", ylab="Residuals")
abline(h=0, col="red")

# Pearson residuals
pearson_resid <- residuals(model, type="pearson")
plot(pearson_resid, main="Pearson Residuals", ylab="Residuals")
abline(h=0, col="red")
