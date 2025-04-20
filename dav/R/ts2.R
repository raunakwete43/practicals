df <- read.csv('Month_Value_1.csv')
print(head(df))

df <- na.omit(df)
# Check for duplicates
print(sum(duplicated(df)))

ts_df <- df$Sales_quantity

ts_data <- ts(ts_df, frequency = 12, start = c(2015, 1))

plot(ts_data, main = "Time Series Data", ylab = "Sales Quantity", xlab = "Time", col = "blue")

adf_test <- adf.test(ts_data)
print(adf_test)

acf(ts_data, main = "ACF - Original Series")
pacf(ts_data, main = "PACF - Original Series")