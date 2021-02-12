# Part 3: Soccer Data

*Introductory - Intermediate level SQL*

---

## Setup

Download the [SQLite database](https://www.kaggle.com/hugomathien/soccer/download). *Note: You may be asked to log in, or "continue and download".* Unpack the ZIP file into your working directory (i.e., wherever you'd like to complete this challenge set). There should be a *database.sqlite* file.

As with Part II, you can check the schema:

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect('database.sqlite')

query = "SELECT * FROM sqlite_master"

df_schema = pd.read_sql_query(query, conn)

df_schema.tbl_name.unique()
```

---

Please complete this exercise using sqlite3 (the soccer data, above) and your Jupyter notebook.

#### 1. Which team scored the most points when playing at home?  
```
SELECT home_team_api_id,team_long_name, SUM(home_team_goal) FROM(
    SELECT * FROM Match
    JOIN Team
    ON Match.home_team_api_id = Team.team_api_id)
GROUP BY home_team_api_id
ORDER BY SUM(home_team_goal) DESC;
```

#### 2. Did this team also score the most points when playing away?
```
SELECT away_team_api_id,team_long_name, SUM(away_team_goal) FROM(
    SELECT * FROM Match
    JOIN Team
    ON Match.away_team_api_id = Team.team_api_id)
GROUP BY home_team_api_id
ORDER BY SUM(home_team_goal) DESC;
```

#### 3. How many matches resulted in a tie?
```
SELECT COUNT(*)
FROM Match
WHERE home_team_goal = away_team_goal;
```

#### 4. How many players have Smith for their last name? How many have 'smith' anywhere in their name?
```
SELECT COUNT(*)
FROM Player
WHERE player_name LIKE '_smith%';
```

```
SELECT COUNT(*)
FROM Player
WHERE player_name LIKE '%smith%';
```

#### 5. What was the median tie score? Use the value determined in the previous question for the number of tie games. *Hint:* PostgreSQL does not have a median function. Instead, think about the steps required to calculate a median and use the [`WITH`](https://www.postgresql.org/docs/8.4/static/queries-with.html) command to store stepwise results as a table and then operate on these results. 
```
WITH ordered_match AS (
  SELECT
      home_team_goal,
      row_number() OVER (ORDER BY home_team_goal) AS row_id,
      (SELECT COUNT(*) FROM Match) AS ct
  FROM Match
)

SELECT AVG(home_team_goal) AS median
FROM ordered_match
WHERE row_id BETWEEN ct/2.0 AND ct/2.0 + 1;
```

#### 6. What percentage of players prefer their left or right foot? *Hint:* Calculate either the right or left foot, whichever is easier based on how you setup the problem.
```
WITH stats AS (
    SELECT 
        DISTINCT preferred_foot,
        COUNT(player_fifa_api_id) OVER(PARTITION by preferred_foot) AS foot_count,
        (SELECT COUNT(*) FROM Player_Attributes) AS total
    FROM Player_Attributes
    )
    
SELECT preferred_foot,
       foot_count*1.0/total AS probability
FROM stats;
```
