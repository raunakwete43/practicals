df1 <- read.csv('Experience-Salary.csv')

print(summary(df1))

plot(
  df1$exp.in.months.,
  df1$salary.in.thousands.,
  main = "Experience vs Salary",
  xlab = "Experience (Years)",
  ylab = "Salary (USD)",
  col = "blue",
  pch = 19
)

# Fit a linear regression model
model <- lm(salary.in.thousands. ~ exp.in.months., data = df1)
summary(model)

# Add reg line to plot
abline(model, col = "red", lwd = 2)


## Multiple Linear Regression
df2 <- read.csv('Student_Performance.csv')
print(summary(df2))


print(sum(duplicated(df2)))
df2 <- na.omit(df2)
df2 <- df2[!duplicated(df2), ]

# Fit model
model2 <- lm(
  Performance.Index ~ Hours.Studied + Previous.Scores + Sleep.Hours + Sample.Question.Papers.Practiced,
  data = df2
)
summary(model2)

# Plot Multiple Linear Regression plot
df2$predicted <- predict(model2)

# Plot predicted vs actual
library(ggplot2)
ggplot(df2, aes(x = predicted, y = Performance.Index)) +
  geom_point() +
  geom_abline(col = "red") +
  labs(title = "Predicted vs Actual Performance Index", x = "Predicted", y = "Actual")
