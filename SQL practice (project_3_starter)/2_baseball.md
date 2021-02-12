# Part 2: Baseball Data

*Introductory - Intermediate level SQL*

---

## Setup

`cd` into the directory you'd like to use for this challenge. Then, download the Lahman SQL Lite dataset

```
curl -L -o lahman.sqlite https://github.com/WebucatorTraining/lahman-baseball-mysql/raw/master/lahmansbaseballdb.sqlite
```

*The `-L` follows redirects, and the `-o` uses the filename instead of outputting to the terminal.*

Make sure sqlite3 is installed

```
conda install -c anaconda sqlite
```

In your notebook, check out the schema

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect('lahman.sqlite')

query = "SELECT * FROM sqlite_master;"

df_schema = pd.read_sql_query(query, conn)

df_schema.tbl_name.unique()
```

---

Please complete this exercise using SQL Lite (i.e., the Lahman baseball data, above) and your Jupyter notebook.
  
**Details are in the Jupyter Notebook.**

#### 1. What was the total spent on salaries by each team, each year?

```
SELECT teamID, team_ID, SUM(salary)
FROM salaries
GROUP BY teamID, yearID;
```
  
  
#### 2. What is the first and last year played for each player? *Hint:* Create a new table from 'Fielding.csv'.

```
SELECT DISTINCT playerID,
        MIN(yearID) OVER (PARTITION BY playerID) AS "First Year",
        MAX(yearID) OVER (PARTITION BY playerID) AS "Last Year"
FROM fielding
GROUP BY playerID, yearID;
```

  
#### 3. Who has played the most all star games?

```
SELECT playerID, COUNT(ID)
FROM allstarfull
GROUP BY playerID
ORDER BY COUNT(ID) DESC;
```

  
#### 4. Which school has generated the most distinct players? *Hint:* Create new table from 'CollegePlaying.csv'.

```
SELECT name_full AS "School", COUNT(DISTINCT playerID) AS "Num of Players"
FROM collegeplaying
JOIN schools
ON schools.schoolID = collegeplaying.schoolID
GROUP BY name_full
ORDER BY COUNT(playerID) DESC;
```

  
#### 5. Which players have the longest career? Assume that the `debut` and `finalGame` columns comprise the start and end, respectively, of a player's career. *Hint:* Create a new table from 'Master.csv'. Also note that strings can be converted to dates using the [`DATE`](https://wiki.postgresql.org/wiki/Working_with_Dates_and_Times_in_PostgreSQL#WORKING_with_DATETIME.2C_DATE.2C_and_INTERVAL_VALUES) function and can then be subtracted from each other yielding their difference in days.

```
SELECT playerID, DATE(finalgame_date) - DATE(debut_date) AS "Career (years)"
FROM people
GROUP BY playerID
ORDER BY (DATE(finalgame_date) - DATE(debut_date)) DESC;
```

  
#### 6. What is the distribution of debut months? *Hint:* Look at the `DATE` and [`EXTRACT`](https://www.postgresql.org/docs/current/static/functions-datetime.html#FUNCTIONS-DATETIME-EXTRACT) functions.

```
SELECT STRFTIME('%m', debut) AS Month, COUNT(*)
FROM people
GROUP BY Month;
```

  
#### 7. What is the effect of table join order on mean salary for the players listed in the main (master) table? *Hint:* Perform two different queries, one that joins on playerID in the salary table and other that joins on the same column in the master table. You will have to use left joins for each since right joins are not currently supported with SQLalchemy.

##### Method 1: Only keep those who have salary info in salaries.
 `FROM salaries JOIN people`:
 
```
SELECT people.playerID, people.nameGiven, people.nameFirst, people.nameLast, AVG(salary)
FROM salaries
LEFT JOIN people
ON salaries.playerID = people.playerID
GROUP BY people.playerID;
```

    
##### Method 2: Keep all the players in people (even without salary info).
 `FROM people JOIN salaries`:
 
```
SELECT people.playerID, people.nameGiven, people.nameFirst, people.nameLast, AVG(salary)
FROM people
LEFT JOIN salaries
ON salaries.playerID = people.playerID
GROUP BY people.playerID
```

