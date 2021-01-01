#!/usr/bin/env python
# coding: utf-8

# # EDA of Bundesliga Football Data

# In this project, I've tried to make an exploratory data analysis and visualized some key points regarding Bundesliga metrics from the dataset in this link: https://www.kaggle.com/slehkyi/extended-football-stats-for-european-leagues-xg
# 
# Note: This dataset only includes domestic league stats, therefore Champions League, Europa League or DFB Pokal performances weren't put into consideration in this project.

# In[1]:


#Necessary library importations and creating the dataframe
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

df = pd.read_csv(r"I:\archive\understat_per_game.csv")


# In[2]:


#Getting a general view of the data
print(df.head())
df.date = pd.to_datetime(df.date)
df.describe()
df.info()


# In[3]:


#Definitely not necessary, although I just couldn't bear to see it wrote like "Fortuna Duesseldorf" so 
#fixed that
df["team"] = df.team.replace("Fortuna Duesseldorf" ,"Fortuna Düsseldorf")

#Simple function to extract a specific year from the year column
def season_date(frame, year):
    new = frame[frame["year"] == year]
    return new
df_2019 = season_date(df, 2019)
bundesliga = df_2019[df_2019["league"] == "Bundesliga"]
df_2019.head()


# #  Definition of some metrics according to the original author of the dataset
# 
# xG - expected goals metric, it is a statistical measure of the quality of chances created and conceded(from understat.com)
# 
# xG_diff - difference between actual goals scored and expected goals.
# 
# npxG - expected goals without penalties and own goals.
# 
# xGA - expected goals against.
# 
# xGA_diff - difference between actual goals missed and expected goals against.
# 
# npxGA - expected goals against without penalties and own goals.
# 
# npxGD - difference between "for" and "against" expected goals without penalties and own goals.
# 
# ppda_coef - passes allowed per defensive action in the opposition half (power of pressure)
# 
# oppda_coef - opponent passes allowed per defensive action in the opposition half (power of opponent's pressure)
# 
# deep - passes completed within an estimated 20 yards of goal (crosses excluded)
# 
# deep_allowed - opponent passes completed within an estimated 20 yards of goal (crosses excluded)
# 
# xpts - expected points
# 
# xpts_diff - difference between actual and expected points

# In[4]:


#Grouped by the team column, then took the means of xG and scored columns 
#in order to compare average xG and goal amounts by team
bundesliga_expected_actual = df_2019[df_2019["league"] == "Bundesliga"].groupby("team")[["xG", "scored"]].mean()
#Reset the index for easier column operations
bundesliga_expected_actual = bundesliga_expected_actual.reset_index()
#Sort values by goal expectation
bundesliga_expected_actual = bundesliga_expected_actual.sort_values("xG", ascending = False)
display(bundesliga_expected_actual)


# In[5]:


#Again group by the team columns, although this time compare with
#average successful deep passes per game
bundesliga_deep = bundesliga.groupby("team").deep.mean()
bundesliga_deep = bundesliga_deep.reset_index()
bundesliga_deep = bundesliga_deep.sort_values("deep", ascending = False)
display(bundesliga_deep)


# In[6]:


#Same as the above two cells
bundesliga_points = df_2019[df_2019["league"] == "Bundesliga"].groupby("team")[["pts", "xpts"]].sum()
bundesliga_points = bundesliga_points.reset_index()
bundesliga_points = bundesliga_points.sort_values("pts", ascending = False)
bundesliga_xpoints = bundesliga_points.sort_values("xpts", ascending = False)
display(bundesliga_points)


# In[7]:


sns.set_context("talk")
sns.set_style("darkgrid")

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(wspace = 0.5)
sns.barplot(x="xG", y="team", palette="rocket_r", data=bundesliga_expected_actual, ax=ax1)
ax1.set(title=("Average Goal Expectations per Team"), xlabel=("Goal Expectation"), ylabel=("Teams"))

sns.barplot(x="deep", y="team", palette="rocket_r", data=bundesliga_deep, ax=ax2)
plt.title("Average Successful Deep Passes per Game")
plt.xlabel("Successful deep (within 20 yards of goal) passes per game")
plt.ylabel("Teams")
plt.show()


# # Comparison of deep passing and goal expectations (xG) by team 
# We can see on the above left graph that Bayern München had the highest average xG per game, while Fortuna Düsseldorf was last place. 
# 
# The metric "deep", which I also used for the right plot was defined by the original author of this dataset as:
# 
# "deep - passes completed within an estimated 20 yards of goal (crosses excluded)"
# 
# According to the graph, again, Bayern performed best regarding the metric "deep", which pretty much stands for passes around and inside the opponent's penalty box. This might offer a general idea on how teams operate in terms of offensive play, as an example Bayern completed a little bit more than three times the amount of passes Paderborn could (which are last place), and we could extract insights such as: "Bayern have internalized a mentality that mostly employs playing into box with organized passing", 
# 
# or: "Paderborn couldn't complete many passes around the box, therefore it indicates that they emphasize a more direct approach when attacking or cross more often" 
# 
# using this metric alone.

# In[25]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(wspace = 0.7)
sns.barplot(x="pts", y="team", palette="rocket_r", data=bundesliga_points, ax=ax1)
ax1.set(title = ("Total Points Collected by Teams"), xlabel=("Total Points"), ylabel=("Teams"), xlim=[0, 85])

sns.barplot(x="xpts", y="team", palette="rocket_r", data=bundesliga_xpoints, ax=ax2)
plt.xlabel("Expected Points Total")
plt.ylabel("Teams")
plt.title("Total Points Expected to Have Been Collected by Teams")
plt.xlim([0, 85])
plt.show()


# # Comparison of actual and expected points collected by team in Bundesliga 19/20 season
# These couple of plots are more interesting to look at in my opinion mostly since they're metrics in direct correlation with each other. 
# 
# As could be seen above, in terms of actual points collected Bayern sit on the top, then come Dortmund and Leipzig. Although on the right-side bar plot, that order changes by Bayern, Leipzig, and Gladbach which have the same expected points as Dortmund. This means that Dortmund overperformed in terms of actual count of total points collected, while Leipzig and Gladbach underperformed. In fact, Bayern also underperformed since they went above 80 points in reality while the expected points metric for them is below that number.
# 
# Let's create a ratio of actual points divided by expected points and visualize that for each team, so that we could see much easier which teams underperformed/overperformed and to what extent.

# In[9]:


#Create variables that represent overperformance and underperformance
#in terms of collecting points
high_xpts = bundesliga_points[bundesliga_points["xpts"] >= bundesliga_points["pts"]]
high_pts = bundesliga_points[bundesliga_points["pts"] > bundesliga_points["xpts"]]
#Create a new column on the dataframe that was grouped by teams and expected points
#that show the ratio of actual points to expected points
bundesliga_points["pts/xpts"] = bundesliga_points["pts"] / bundesliga_points["xpts"]


# In[10]:


ax = plt.subplots(figsize=(15, 10))
sns.barplot("pts/xpts", "team", data=bundesliga_points.sort_values(by="pts/xpts", ascending=False), palette="GnBu")
plt.title("Performance in Point Collection")
plt.xlabel("Points to Expected Points Ratio")
plt.ylabel("Team")
plt.show()


# # Evaluation of whether if the teams could live up to expectations in terms of points collected
# So, let's try to get a more intuitive view of this bar plot. Numbers on the x-axis represent the ratio that we talked about on the above cell: actual points/expected points. Setting this as our starting point, a ratio on 1.0 would mean that the actual amount of points gathered by a team were 100% consistent with the expectation metric. So, simply put, it could be said that teams that are above the ratio 1.0 *overperformed*, while those under 1.0 *underperformed*.
# 
# Let's try the same thing with the actual amount of goals scored and goal expectations!

# In[26]:


#Create variables that represent overperformance and underperformance
#in terms of goal scoring
high_xg = bundesliga_expected_actual[bundesliga_expected_actual["xG"] >= bundesliga_expected_actual["scored"]]
high_scoring = bundesliga_expected_actual[bundesliga_expected_actual["scored"] > bundesliga_expected_actual["xG"]]
#Create a new column on the dataframe that was grouped by teams and goal expectations
#that show the ratio of goals to xG
bundesliga_expected_actual["goal/xG"] = bundesliga_expected_actual["scored"] / bundesliga_expected_actual["xG"]


# In[12]:


#Check if all's well
bundesliga_expected_actual.head()


# In[13]:


ax = plt.subplots(figsize=(15, 10))
sns.barplot("goal/xG", "team", data=bundesliga_expected_actual.sort_values(by="goal/xG", ascending=False), palette="RdYlBu")
plt.ylabel("Team")
plt.xlabel("Goal to Goal Expectation Ratio")
plt.title("Goal/xG Ratios of Bundesliga Clubs in 19/20 Season")


# # Evaluation of whether if the teams could live up to expectations in terms of goals scored
# This bar plot essentially gives us a visual represantation of the teams' "goal efficiency" rates, where 1.0 means similarly to the bar plot above that the team in question scored as many goals as they were expected to. 
# 
# Therefore the teams with a higher ratio, such as Borussia Dortmund or Werder Bremen, could be said to have overperformed in terms of converting chances to goals compared to those that are located at the lower sections of the graph, like Wolfsburg or Borussia Mönchengladbach.
# 
# Instead of the league in general, let's focus on a single team now. I chose RB Leipzig, since I've been enjoying their playing style and Julian Nagelsmann's display of his managerial skill set.

# In[14]:


#Filter the Bundesliga dataset to get rows that only contain data for RB Leipzig
leipzig = bundesliga[bundesliga["team"] == "RasenBallsport Leipzig"]
leipzig.head()


# In[15]:


#Create a heatmap to visulize the Pearson correlation coefficient for RB Leipzig metrics
sns.set_context("talk")
ax=plt.subplots(figsize=(10,10))
sns.heatmap(leipzig.corr())
plt.title("RB Leipzig's 19/20 season correlation matrix")
plt.savefig("leipzig_matrix")


# # Correlation between RB Leipzig data features 
# The correlation matrix is a very intuitive visualization method to discover *linear* relationships between features. Although it must be noted that a low Pearson coefficient doesn't necessarily indicate that the features in question are not correlated at all, this value only informs us of whether if the selected features are *linearly* correlated. Therefore, a very strong polynomial correlation (such as a parabola) between two variables will still return a low Pearson coefficient.

# In[16]:


#Create variables by filtering which venue the game was played at
leipzig_home = leipzig[leipzig["h_a"] == "h"]
leipzig_away = leipzig[leipzig["h_a"] == "a"]


# In[17]:


#Calculate the average goal expectations by venue 
print(leipzig_home.xG.mean(), leipzig_away.xG.mean())


# In[18]:


#See how many games RB Leipzig won, lost or drew
leipzig["result"].value_counts()


# In[23]:


leipzig_r_xg = leipzig.groupby("result").xG_diff.mean().head()
leipzig_r_xg = leipzig_r_xg.reset_index()

sns.set_style("darkgrid")
sns.set_context("poster")
ax = plt.subplots(figsize=(8, 8))
sns.barplot("result", "xG_diff", data=leipzig_r_xg, palette="cividis")
plt.title("Difference of Actual Goals - Expected Goals")
plt.ylabel("Difference")
plt.xlabel("Result")
plt.show()


# # Goals scored - goal expectation difference per result category
# Here's another plot that I find enjoyable to look at. Each bar stands for a result (win/draw/lose) and the y-axis represents the mean difference between the actual count of goals scored and the goal expectation metric. Or to mathematically notate:
# 
# (The average of actual goal counts for each result category) - (The average of xG for each result category)
# 
# According to the above graph, Leipzig have exceeded expectations in terms of goal scoring for the games they won, which means that they scored fewer goals than they were supposed to. And interestingly enough, Leipzig apparently have overperformed in the games they couldn't win and scored more goals than they were expected to. This might indicate that Leipzig won the games they bagged 3 points by dominating their opponents, whereas for the ones they couldn't win the team were outperformed on the pitch and couldn't get many scoring opportunities, yet somehow managed to make the ball cross the goal line.

# In[20]:


plt.clf()


# In[21]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

sns.set_context("poster")

sns.barplot("h_a", "scored", data=leipzig, ax=ax1, ci=False, palette = "cividis")
ax1.set(xlabel=("Home/Away"), ylabel=("Goals scored"), ylim=[0, 2.8])

sns.barplot("h_a", "xG", data=leipzig, ax=ax2 , ci=False, palette = "cividis")
plt.ylim([0, 2.8])
plt.xlabel("Home/Away")
plt.ylabel("Goal Expectation")
plt.show()


# # Comparison of goals scored and xG by venue played (home/away)
# The plots above allow us to compare the average counts of goals scored and expected per game by whether the game was played at Red Bull Arena or not. 
# 
# It appears that Leipzig have performed better at away games in terms of scoring goals and there's just a slight difference between the actual and expected amount of goals for away games. 
# 
# For home games however, the difference between actual goals and xG seem to differ roughly by 0.5 goals in average. Basically, Leipzig have (again, roughly) scored a goal more than they should have for each two game played at home. This might've been caused by a variety of factors, such as fan support or opponents shifting to a more reactive style of play since it's an away game. Therefore it could lead to incorrect assumptions if we were to draw conclusions from these graphs alone.

# In[22]:


results_h_a = leipzig.groupby("h_a").result.value_counts()
ax = plt.subplots(figsize = (8, 8))

sns.countplot("result",data=leipzig,hue="h_a", palette = "cividis")
plt.yticks([0, 2, 5, 8, 10])
plt.title("Leipzig 19/20 game result counts by venue played")
plt.xlabel("Result")
plt.ylabel(" ")
plt.legend(["Away", "Home"])
plt.show()
print(leipzig.groupby("h_a").result.value_counts())


# # Result counts by venue the game was played
# Just like the bar plots under the above cells which depicted that Leipzig had scored more goals away, the team appears to have also won almost 50% more games away compared to home matches. The significance of their away performance manifests itself further considered that Leipzig has more draws at home than wins. 
# 
# It wouldn't be correct to draw any conclusions without further analysis although, on whether if they just overperformed at away games or their home record's an outcome of underperformance. These statements might sound like they are essentially addressing the same thing at first, however it could be seen that is not the case after going over them once more carefully.

# # What does it take to win the title?
# 
# We looked into RB Leipzig's performance data, and we saw that RB Leipzig should've been second place if Borussia Dortmund didn't overperform in terms of point collected throughout the season, which'd put them right under the reigning champions, Bayern München.
# 
# So what else do Leipzig need to show in order to reach the finish line first? Let's compare them against Bayern and see what we can find!

# In[35]:


#Select rows with only Bayern and Leipzig as teams
leipzig_bayern = bundesliga[(bundesliga["team"] == "Bayern Munich") | (bundesliga["team"] == "RasenBallsport Leipzig")]
leipzig_bayern.head()


# In[47]:


#Compare the results of Bayern and Leipzig
ax = plt.subplots(figsize = (8, 8))
sns.set_context("notebook")
sns.countplot("result", hue="team", data=leipzig_bayern, palette="RdBu")
plt.title("Result Counts of Bayern Munich vs RB Leipzig")
plt.xlabel("Result")
plt.ylabel(" ")


# In[53]:


ax = plt.subplots(figsize = (8, 8))
sns.set_context("talk")
sns.scatterplot("oppda_coef", "deep", hue="team", data=leipzig_bayern, palette="RdBu")
print(leipzig_bayern.groupby("team").deep.mean())
print(leipzig_bayern.groupby("team").oppda_coef.mean())


# What this graph and the mean values of oppda_coef and deep metrics tell us is that despite facing tougher average pressure (50% tougher!) from their opponents, Bayern managed to achieve a higher number of completed passes around and inside their opponents' penalty box.
