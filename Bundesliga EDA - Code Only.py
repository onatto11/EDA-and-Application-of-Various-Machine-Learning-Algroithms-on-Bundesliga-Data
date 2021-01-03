#Necessary library importations and creating the dataframe
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

df = pd.read_csv(r"I:\archive\understat_per_game.csv")

#Getting a general view of the data
print(df.head())
df.date = pd.to_datetime(df.date)
df.describe()
df.info()

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

#Plot the cumulative change in total points and therefore the rankings in time

ax = plt.subplots(figsize=(15, 10))
sns.set_context("notebook")
sns.set_style("dark")
sns.lineplot(bundesliga.date, bundesliga.groupby("team").pts.cumsum(), hue="team", data=bundesliga, palette = "Dark2")
sns.despine(top=True, bottom=True, right=True)
plt.title("Change in Bundesliga rankings by time")
plt.xlabel("Date")
plt.ylabel("Points total")

#Grouped by the team column, then took the means of xG and scored columns 
#in order to compare average xG and goal amounts by team
bundesliga_expected_actual = df_2019[df_2019["league"] == "Bundesliga"].groupby("team")[["xG", "scored"]].mean()
#Reset the index for easier column operations
bundesliga_expected_actual = bundesliga_expected_actual.reset_index()
#Sort values by goal expectation
bundesliga_expected_actual = bundesliga_expected_actual.sort_values("xG", ascending = False)
display(bundesliga_expected_actual)

#Again group by the team columns, although this time compare with
#average successful deep passes per game
bundesliga_deep = bundesliga.groupby("team").deep.mean()
bundesliga_deep = bundesliga_deep.reset_index()
bundesliga_deep = bundesliga_deep.sort_values("deep", ascending = False)
display(bundesliga_deep)

#Same as the above two cells
bundesliga_points = df_2019[df_2019["league"] == "Bundesliga"].groupby("team")[["pts", "xpts"]].sum()
bundesliga_points = bundesliga_points.reset_index()
bundesliga_points = bundesliga_points.sort_values("pts", ascending = False)
bundesliga_xpoints = bundesliga_points.sort_values("xpts", ascending = False)
display(bundesliga_points)

#Create a column named press_eff showing the ratio of passes that teams allowed 
#their opponents to make inside their box before recieving the ball

bundesliga["press_eff"] = bundesliga["deep_allowed"]/bundesliga["ppda_coef"]
bundesliga_press = bundesliga.groupby("team").press_eff.mean()
bundesliga_press = bundesliga_press.reset_index()
bundesliga_press = bundesliga_press.sort_values("press_eff", ascending = False)
display(bundesliga_press)

#Create a column named peneteration showing the ratio of passes that teams made 
#inside opposition box before their opponents recovered the ball

bundesliga["peneteration"] = bundesliga["deep"]/bundesliga["oppda_coef"]
bundesliga_penet = bundesliga.groupby("team").peneteration.mean()
bundesliga_penet = bundesliga_penet.reset_index()
bundesliga_penet = bundesliga_penet.sort_values(by = "peneteration", ascending = False)
display(bundesliga_penet)

#Plot bar graphs of average xpts and deep for each team side-by-side

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

#Plot bar graphs of total pts and xpts for each team side-by-side

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

#Create variables that represent overperformance and underperformance
#in terms of collecting points
high_xpts = bundesliga_points[bundesliga_points["xpts"] >= bundesliga_points["pts"]]
high_pts = bundesliga_points[bundesliga_points["pts"] > bundesliga_points["xpts"]]
#Create a new column on the dataframe that was grouped by teams and expected points
#that show the ratio of actual points to expected points
bundesliga_points["pts/xpts"] = bundesliga_points["pts"] / bundesliga_points["xpts"]

#Plot bar graphs of average pts divided by xpts for each team to understand
#how much each team lived up to expectations in terms of point collection

ax = plt.subplots(figsize=(15, 10))
sns.barplot("pts/xpts", "team", data=bundesliga_points.sort_values(by="pts/xpts", ascending=False), palette="GnBu")
plt.title("Performance in point collection")
plt.xlabel("Points to expected points ratio")
plt.ylabel("Team")
plt.show()

#Create variables that represent overperformance and underperformance
#in terms of goal scoring
high_xg = bundesliga_expected_actual[bundesliga_expected_actual["xG"] >= bundesliga_expected_actual["scored"]]
high_scoring = bundesliga_expected_actual[bundesliga_expected_actual["scored"] > bundesliga_expected_actual["xG"]]
#Create a new column on the dataframe that was grouped by teams and goal expectations
#that show the ratio of goals to xG
bundesliga_expected_actual["goal/xG"] = bundesliga_expected_actual["scored"] / bundesliga_expected_actual["xG"]

#Check if all's well
bundesliga_expected_actual.head()

#Plot bar graphs of average goals and expected for each team to see
#how well they performed

ax = plt.subplots(figsize=(15, 10))
sns.barplot("goal/xG", "team", data=bundesliga_expected_actual.sort_values(by="goal/xG", ascending=False), palette="RdYlBu")
plt.ylabel("Team")
plt.xlabel("Goal to Goal Expectation Ratio")
plt.title("Goal/xG Ratios of Bundesliga Clubs in 19/20 Season")

#Plot a bar graph to show the percantage of how many of the passes allowed 
#before recovery were inside a team's penalty box

ax = plt.subplots(figsize=(15, 10))
sns.barplot("press_eff", "team", data=bundesliga_press, palette="RdYlBu")
plt.xlabel("Allowed opponent deep passing percantage")
plt.ylabel("Team")
plt.title("The percentage of passes made around teams' penalty box until recovery")

#Plot a bar graph to show the percantage of how many passes teams allowed 
#inside their penalty box before recovering the ball 

ax = plt.subplots(figsize=(15, 10))
sns.barplot("peneteration", "team", data=bundesliga_penet, palette="RdYlBu")
plt.xlabel("Percentage of passing inside oppponent's box before loss of possession")
plt.ylabel("Team")
plt.title("The percentage of passes made inside opponents' penalty box until loss")

#Create scatter plots side-by-side showing how goal expectation changes
#by metrics ppda_coef, oppda_coef, deep and deep_allowed

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(wspace = 0.3)
sns.scatterplot(x="ppda_coef", y="xG", data=bundesliga, ax=ax1, alpha=0.6)
sns.scatterplot(x="oppda_coef", y="xG", data=bundesliga, ax=ax1, alpha=0.6)
ax1.set(title = ("Change of xG by OPPDA/PPDA"), xlabel=("OPPDA/PPDA"), ylabel=("xG"))
ax1.legend(["PPDA","OPPDA"])

sns.scatterplot(x="deep", y="xG", data=bundesliga, ax=ax2, alpha=0.6)
sns.scatterplot(x="deep_allowed", y="xG", data=bundesliga, ax=ax2, alpha=0.6)
plt.xlabel("Deep passes made/allowed")
plt.ylabel("xG")
plt.title("Change of xG by deep passes made/allowed")
plt.legend(["Deep Made", "Deep Allowed"])
plt.show()

#Create scatter plots side-by-side showing how xpts changes
#by metrics ppda_coef, oppda_coef, deep and deep_allowed

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
plt.subplots_adjust(wspace = 0.3)
sns.scatterplot(x="ppda_coef", y="xpts", data=bundesliga, ax=ax1, alpha=0.6)
sns.scatterplot(x="oppda_coef", y="xpts", data=bundesliga, ax=ax1, alpha=0.6)
ax1.set(title = ("Change of expected points by OPPDA/PPDA"), xlabel=("OPPDA/PPDA"), ylabel=("Expected Points"))
ax1.legend(["PPDA","OPPDA"])

sns.scatterplot(x="deep", y="xpts", data=bundesliga, ax=ax2, alpha=0.6)
sns.scatterplot(x="deep_allowed", y="xpts", data=bundesliga, ax=ax2, alpha=0.6)
plt.xlabel("Deep passes made/allowed")
plt.ylabel("Expected Points")
plt.title("Change of expected points by deep passes made/allowed")
plt.legend(["Deep Made", "Deep Allowed"])
plt.show()

#Filter the Bundesliga dataset to get rows that only contain data for RB Leipzig
leipzig = bundesliga[bundesliga["team"] == "RasenBallsport Leipzig"]
leipzig.head()

#Create a heatmap to visulize the Pearson correlation coefficient for RB Leipzig metrics
sns.set_context("talk")
ax=plt.subplots(figsize=(10,10))
sns.heatmap(leipzig.corr())
plt.title("RB Leipzig's 19/20 season correlation matrix")
plt.savefig("leipzig_matrix")

#Create variables by filtering which venue the game was played at
leipzig_home = leipzig[leipzig["h_a"] == "h"]
leipzig_away = leipzig[leipzig["h_a"] == "a"]

#Calculate the average goal expectations by venue 
print(leipzig_home.xG.mean(), leipzig_away.xG.mean())

#See how many games RB Leipzig won, lost or drew
leipzig["result"].value_counts()

#Create a bar plot showing the difference between how much difference
#there was between goals that were actually scored and should've been
#scored in average by each result

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

plt.clf()

#Plot two bar graphs side-by-side, one showing how the average amounts of goals
#scored per game and the other xG per game depending on the venue played

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

sns.set_context("poster")

sns.barplot("h_a", "scored", data=leipzig, ax=ax1, ci=False, palette = "cividis")
ax1.set(xlabel=("Home/Away"), ylabel=("Goals scored"), ylim=[0, 2.8])

sns.barplot("h_a", "xG", data=leipzig, ax=ax2 , ci=False, palette = "cividis")
plt.ylim([0, 2.8])
plt.xlabel("Home/Away")
plt.ylabel("Goal Expectation")
plt.show()

#Create a count plot that visualizes the amounts of wins, draws and losses
#depending on the venue played

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

#Select rows with only Bayern and Leipzig as teams
leipzig_bayern = bundesliga[(bundesliga["team"] == "Bayern Munich") | (bundesliga["team"] == "RasenBallsport Leipzig")]
leipzig_bayern.head()

#Compare the results of Bayern and Leipzig
ax = plt.subplots(figsize = (8, 8))
sns.set_context("notebook")
sns.countplot("result", hue="team", data=leipzig_bayern, palette="RdBu")
plt.title("Result Counts of Bayern Munich vs RB Leipzig")
plt.xlabel("Result")
plt.ylabel(" ")

#Create a scatter plot visualizing how each team performed regarding deep passes
#under different levels of pressure received 

ax = plt.subplots(figsize = (10, 10))
sns.set_context("talk")
sns.scatterplot("oppda_coef", "deep", hue="team", data=leipzig_bayern, palette="RdBu")
plt.title("Change of passes around opposition box by opponent's pressing strength")
plt.ylabel("Successful passes")
plt.xlabel("Opposition pressing strength")
print(leipzig_bayern.groupby("team").deep.mean())
print(leipzig_bayern.groupby("team").oppda_coef.mean())

#Plot a bar graph showing what percentage of the passes made before losing the ball 
#was inside the opposition box

ax = plt.subplots(figsize=(8, 8))
sns.barplot("team", bundesliga_penet["peneteration"], data=bundesliga_penet[(bundesliga_penet["team"] == "RasenBallsport Leipzig") | 
                                                       (bundesliga_penet["team"] == "Bayern Munich")], palette="RdYlBu")
plt.xlabel("Percentage of passing inside oppponent's box before loss of possession")
plt.ylabel("Team")
plt.title("The percentage of passes made inside opponents' penalty box until loss")
plt.ylim([0, 0.85])

#Create a scatter plot that visualizes how many passes teams allowed inside
#their own box before recovering the ball

ax = plt.subplots(figsize = (10, 10))
sns.set_context("talk")
sns.scatterplot("ppda_coef", "deep_allowed", hue="team", data=leipzig_bayern, palette="RdBu")
plt.title("Passes inside box vs. pressing strength")
plt.ylabel("Opposition successful passes")
plt.xlabel("Pressing strength")
print(leipzig_bayern.groupby("team").deep_allowed.mean())
print(leipzig_bayern.groupby("team").ppda_coef.mean())

#Plot a bar graph showing what percentage of the passes were made inside
#the teams' box before recovery

ax = plt.subplots(figsize=(8, 8))
sns.barplot("team", "press_eff", data=bundesliga_press[(bundesliga_press["team"] == "RasenBallsport Leipzig") | 
                                                       (bundesliga_press["team"] == "Bayern Munich")], palette="RdYlBu_r")
plt.xlabel("Allowed opponent deep passing percantage")
plt.ylabel("Team")
plt.title("The percentage of passes made around teams' penalty box until recovery")

plt.ylim([0, 0.65])