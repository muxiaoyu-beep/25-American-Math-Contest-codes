# Load data
# Set file paths (modify paths according to your OS)
# Windows path
file_path1 <- "C:/Users/joker/Desktop/data_dictionary.csv"
data_dictionary <- read.csv(file_path1, header = TRUE, stringsAsFactors = FALSE)  # Assuming file has header

file_path2 <- "C:/Users/joker/Desktop/summerOly_medal_counts.csv"
medal_data <- read.csv(file_path2, header = TRUE, stringsAsFactors = FALSE)  # Assuming file has header

file_path3 <- "C:/Users/joker/Desktop/summerOly_hosts.csv"
host_data <- read.csv(file_path3, header = TRUE, stringsAsFactors = FALSE)  # Assuming file has header

file_path4 <- "C:/Users/joker/Desktop/summerOly_programs.csv"
summerOly_programs <- read.csv(file_path4, header = TRUE, stringsAsFactors = FALSE)  # Assuming file has header

file_path5 <- "C:/Users/joker/Desktop/summerOly_athletes.csv"
data <- read.csv(file_path5, header = TRUE, stringsAsFactors = FALSE)  # Assuming file has header

# Trend plot
# Load required libraries
library(ggplot2)
library(dplyr)
library(readr)

# Data preprocessing
# Group data by country and year, calculate annual total medals
medal_trends <- data %>%
  mutate(Year = as.integer(Year)) %>%
  group_by(NOC, Year) %>%
  summarise(Total_Medals = sum(Total, na.rm = TRUE), .groups = 'drop')

# Check processed data
head(medal_trends)

# Select representative countries for prediction
selected_countries <- c("United States", "China", "Russia", "Great Britain", "Germany", "Japan")

# Filter selected countries
filtered_data <- medal_trends %>%
  filter(NOC %in% selected_countries)

# Create trend plot
ggplot(filtered_data, aes(x = Year, y = Total_Medals, color = NOC)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  scale_x_continuous(breaks = seq(1896, 2024, by = 4), labels = seq(1896, 2024, by = 4)) +
  labs(title = "Trend of Medal Counts Over Time (1896-2024)",
       x = "Year",
       y = "Total Medals",
       color = "Country") +
  theme_minimal() +
  theme(legend.position = "right",
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))  # Rotate x-axis labels vertically

################################################ Question 2
# Load required libraries
library(dplyr)
library(ggplot2)

# Assume data is loaded into a data.frame named medal_data
# To load data, use:
# medal_data <- read.csv("summerOly_medal_counts.csv", stringsAsFactors = FALSE)

# Check column names
print(colnames(medal_data))

# Filter data for Romania and USA
filtered_data <- medal_data %>%
  dplyr::filter(NOC %in% c("Romania", "United States"))

# View filtered data
print(filtered_data)

# Create line plot
ggplot(filtered_data, aes(x = Year, y = Total, color = NOC)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  labs(title = "Total Medals for Romania and USA over Time",
       x = "Year",
       y = "Total Medals",
       color = "Country") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_x_continuous(breaks = seq(min(filtered_data$Year), max(filtered_data$Year), by = 4))

# Load required libraries
library(ggplot2)
library(dplyr)

# Filter data for Romania and USA from 1924 onward
filtered_data <- medal_data %>%
  filter(NOC %in% c("Romania", "United States") & Year >= 1924)

# Calculate scores
filtered_data <- filtered_data %>%
  mutate(Score = Gold * 1 + Silver * 0.6 + Bronze * 0.4)

# Calculate change rates
filtered_data <- filtered_data %>%
  arrange(NOC, Year) %>%
  group_by(NOC) %>%
  mutate(
    Past_Two_Years_Score_Avg = (lag(Score, default = NA) + lag(Score, n = 2, default = NA)) / 2,
    Change_Rate = ifelse((Score - coalesce(Past_Two_Years_Score_Avg, 0)) / coalesce(Past_Two_Years_Score_Avg, 1), NA_real_)
  ) %>%
  ungroup()

# Remove NA values
filtered_data <- filtered_data %>% filter(!is.na(Change_Rate))

# Create volcano plot
ggplot(filtered_data, aes(x = Year, y = NOC, size = Score, color = Change_Rate)) +
  geom_point(alpha = 0.7) +
  scale_size(range = c(3, 30)) +  # Adjust point size range
  scale_color_gradientn(colors = c("#FF9999", "#FF0000"), values = scales::rescale(c(-1, 1), c(-1, 1)), 
                        name = "Change Rate", labels = scales::label_percent()) +
  labs(title = "Medal Count Volcano Plot for Romania and USA (1924-2024)",
       x = "Year",
       y = "Country",
       size = "Total Medals",
       color = "Change Rate") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Filter data for USA and China from 1984 onward
filtered_data <- medal_data %>%
  filter(NOC %in% c("United States", "China") & Year >= 1984)

# Calculate scores
filtered_data <- filtered_data %>%
  mutate(Score = Gold * 1 + Silver * 0.6 + Bronze * 0.4)

# Calculate change rates
filtered_data <- filtered_data %>%
  arrange(NOC, Year) %>%
  group_by(NOC) %>%
  mutate(
    Past_Two_Years_Score_Avg = (lag(Score, 1, default = NA) + lag(Score, 2, default = NA)) / 2,
    Change_Rate = (Score - coalesce(Past_Two_Years_Score_Avg, 0)) / coalesce(Past_Two_Years_Score_Avg, 1)
  ) %>%
  ungroup()

# Remove NA values
filtered_data <- filtered_data %>% filter(!is.na(Change_Rate))

# Create line plot
ggplot(filtered_data, aes(x = Year, y = Score, group = NOC, color = NOC)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  scale_color_manual(values = c("United States" = "blue", "China" = "red")) +
  labs(title = "Medal Count Change Plot for USA and China (1924-2024)",
       x = "Year",
       y = "Total Medals",
       color = "Country") +
  theme_minimal() +
  theme(legend.position = "bottom")

# Filter data for USA and China from 1924 onward
filtered_data <- medal_data %>%
  filter(NOC %in% c("United States", "China") & Year >= 1924)

# Calculate scores
filtered_data <- filtered_data %>%
  mutate(Score = Gold * 1 + Silver * 0.6 + Bronze * 0.4)

# Calculate change rates
filtered_data <- filtered_data %>%
  arrange(NOC, Year) %>%
  group_by(NOC) %>%
  mutate(
    Past_Two_Years_Score_Avg = (lag(Score, 1, default = NA) + lag(Score, 2, default = NA)) / 2,
    Change_Rate = (Score - coalesce(Past_Two_Years_Score_Avg, 0)) / coalesce(Past_Two_Years_Score_Avg, 1)
  ) %>%
  ungroup()

# Remove NA values
filtered_data <- filtered_data %>% filter(!is.na(Change_Rate))

# Create volcano plot
ggplot(filtered_data, aes(x = Year, y = NOC, size = abs(Change_Rate), color = Change_Rate)) +
  geom_point(alpha = 0.7) +
  scale_size(range = c(1, 30)) +
  scale_color_gradient2(low = "white", mid = "red", midpoint = 0, limits = c(-1, 1), 
                        name = "Change Rate", labels = scales::label_percent(accuracy = 0.01)) +
  labs(title = "Medal Change Rate Volcano Plot for USA and China (1924-2024)",
       x = "Year",
       y = "Country",
       size = "Change Rate Magnitude",
       color = "Change Rate") +
  theme_minimal() +
  theme(legend.position = "bottom",
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(size = 12),
        legend.text = element_text(size = 12))

# Load packages and declare function priorities
library(tidyverse)
library(changepoint)
library(ggplot2)
conflicts_prefer(dplyr::filter, dplyr::select, dplyr::group_by, dplyr::summarise)

# Read data
file_path2 <- "C:/Users/joker/Desktop/summerOly_medal_counts.csv"
medal_data <- read.csv(file_path2, header = TRUE, stringsAsFactors = FALSE)

# Data preprocessing
data_clean <- medal_data %>%
  dplyr::select(NOC, Year, Total) %>%
  dplyr::group_by(NOC, Year) %>%
  dplyr::summarise(Total = sum(Total), .groups = "drop") %>%
  dplyr::arrange(NOC, Year)

# Filter countries with ≥5 participations
valid_countries <- data_clean %>%
  dplyr::group_by(NOC) %>%
  dplyr::filter(n() >= 5) %>%
  dplyr::pull(NOC) %>%
  unique()

# Change point detection and improvement calculation
results <- data.frame(
  Country = character(),
  ChangeYear = integer(),
  PreMean = numeric(),
  PostMean = numeric(),
  Improvement = numeric(),
  PValue = numeric()
)

for (country in valid_countries) {
  country_data <- data_clean %>% dplyr::filter(NOC == country)
  ts_data <- country_data$Total
  
  cpt <- cpt.mean(ts_data, method = "PELT")
  if (length(cpts(cpt)) == 0) next
  
  change_year <- country_data$Year[cpts(cpt)[1]]
  
  pre_period <- country_data %>%
    dplyr::filter(Year < change_year) %>%
    tail(3) %>% 
    dplyr::pull(Total)
  post_period <- country_data %>%
    dplyr::filter(Year >= change_year) %>%
    head(3) %>% 
    dplyr::pull(Total)
  
  # Check data sufficiency for t-test
  if (length(pre_period) < 2 | length(post_period) < 2) next
  
  pre_mean <- mean(pre_period)
  post_mean <- mean(post_period)
  improvement <- (post_mean / pre_mean - 1) * 100
  t_test <- t.test(post_period, pre_period, alternative = "greater")
  
  results <- rbind(results, data.frame(
    Country = country,
    ChangeYear = change_year,
    PreMean = pre_mean,
    PostMean = post_mean,
    Improvement = improvement,
    PValue = t_test$p.value
  ))
}

# Filter significant results (improvement>50% & p<0.05)
significant_results <- results %>%
  dplyr::filter(PValue < 0.05 & Improvement > 50) %>%
  dplyr::arrange(desc(Improvement))

# --------------------------
# Visualization code (unchanged)
# --------------------------
# Example: China's medal trend
china_data <- data_clean %>% dplyr::filter(NOC == "China")
change_year_china <- significant_results %>% 
  dplyr::filter(Country == "China") %>% 
  dplyr::pull(ChangeYear)

ggplot(china_data, aes(x = Year, y = Total)) +
  geom_line(color = "#2c7bb6", linewidth = 1) +
  geom_point(color = "#d7191c", size = 2) +
  geom_vline(xintercept = change_year_china, linetype = "dashed", color = "#fdae61") +
  annotate("text", x = change_year_china, y = max(china_data$Total), 
           label = paste("Change Point:", change_year_china), hjust = -0.1, color = "#fdae61") +
  labs(title = "The trend of China's Olympic medal count in the Summer Olympics ", 
       x = "Year", y = "Total Number of Medals") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# 1. Change point visualization (generic function)
plot_cpt <- function(country_name) {
  country_data <- data_clean %>% filter(NOC == country_name)
  change_year <- significant_results %>% 
    filter(Country == country_name) %>% 
    pull(ChangeYear)
  
  ggplot(country_data, aes(x = Year, y = Total)) +
    geom_line(color = "#2c7bb6", linewidth = 1) +
    geom_point(color = "#d7191c", size = 2) +
    geom_vline(xintercept = change_year, linetype = "dashed", color = "#fdae61") +
    geom_smooth(method = "loess", se = FALSE, color = "#abd9e9") +
    labs(title = paste("Medal Trend with Change Point:", country_name),
         x = "Year", y = "Total Medals") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
}

# Example usage
plot_cpt("United States")
plot_cpt("Germany")

# 2. Pre-post comparison plot
mean_comparison <- significant_results %>%
  pivot_longer(cols = c(PreMean, PostMean), 
               names_to = "Period", 
               values_to = "MeanMedals") %>%
  mutate(Period = factor(Period, levels = c("PreMean", "PostMean"),
                         labels = c("Pre-Change", "Post-Change")))

ggplot(mean_comparison, aes(x = Period, y = MeanMedals, fill = Period)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = round(MeanMedals, 1)), vjust = -0.3) +
  facet_wrap(~ Country, scales = "free_y") +
  scale_fill_manual(values = c("#fdae61", "#2c7bb6")) +
  labs(title = "Comparison of Medal Averages Before/After Change Points",
       y = "Average Medals (3 Games Period)") +
  theme_minimal() +
  theme(legend.position = "top",
        axis.title.x = element_blank())

# 3. Improvement ranking plot
top_countries <- significant_results %>%
  arrange(desc(Improvement)) %>%
  head(10) %>%
  mutate(Country = fct_reorder(Country, Improvement))

ggplot(top_countries, aes(x = Improvement, y = Country)) +
  geom_col(fill = "#2c7bb6", width = 0.7) +
  geom_text(aes(label = paste0(round(Improvement, 1), "%")), 
            hjust = -0.1, color = "grey30") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(title = "Top 10 Countries by Medal Improvement",
       x = "Percentage Improvement (%)",
       y = NULL) +
  theme_minimal() +
  theme(panel.grid.major.y = element_blank())

# 4. Multi-country comparison plot
selected_countries <- significant_results %>%
  arrange(desc(Improvement)) %>%
  head(5) %>% 
  pull(Country)

multi_data <- data_clean %>% 
  filter(NOC %in% selected_countries) %>%
  left_join(select(significant_results, Country, ChangeYear), 
            by = c("NOC" = "Country"))

ggplot(multi_data, aes(x = Year, y = Total, color = NOC)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  geom_vline(aes(xintercept = ChangeYear, color = NOC), 
             linetype = "dashed", show.legend = FALSE) +
  scale_color_brewer(palette = "Set2") +
  labs(title = "Medal Trends Comparison with Change Points",
       x = "Year", y = "Total Medals") +
  theme_minimal() +
  theme(legend.position = "bottom",
        panel.grid.minor = element_blank()) +
  facet_wrap(~ NOC, ncol = 2, scales = "free_y")