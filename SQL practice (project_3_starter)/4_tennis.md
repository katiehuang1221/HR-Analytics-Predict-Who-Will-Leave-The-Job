# Part 4: Tennis Data

*Intermediate - Advanced level SQL*

---

## Setup

We'll be using tennis data from [here](https://archive.ics.uci.edu/ml/datasets/Tennis+Major+Tournament+Match+Statistics).

Navigate to your preferred working directory, and download the data.

```bash
curl -L -o tennis.zip http://archive.ics.uci.edu/ml/machine-learning-databases/00300/Tennis-Major-Tournaments-Match-Statistics.zip
unzip tennis.zip -d tennis
```

Make sure you have Postgres installed and initialized

```bash
brew install postgresql
brew services start postgres
```

Install SQLAlchemy if you haven't already

```
conda install -c anaconda sqlalchemy
```

Start Postgres in your terminal with the command `psql`. Then, create a `tennis` database using the `CREATE DATABASE` command.

```
psql

<you_user_name>=# CREATE DATABASE TENNIS;
CREATE DATABASE
<you_user_name>=# \q
```

Pick a table from the *tennis* folder, and upload it to the database using SQLAlchemy and Pandas.

```python
from sqlalchemy import create_engine
import pandas as pd


engine = create_engine('postgresql://<your_user_name>:localhost@localhost:5432/tennis')

aus_men = pd.read_csv('./tennis/AusOpen-men-2013.csv')

# I'm choosing to name this table "aus_men"
aus_men.to_sql('aus_men', engine, index=False)
```

*Note: In the place of `<your_user_name>` you should have your computer user name ...*

Check that you can access the table

```python
query = 'SELECT * FROM aus_men;'
df = pd.read_sql(query, engine)

df.head()
```

Do the same for the other CSV files in the *tennis* directory.

---

## The challenges!

This challenge uses only SQL queries. Please submit answers in a markdown file.
  
**Details can be found in the [Jupyter Notebook](https://github.com/katiehuang1221/onl_ds5_project_3/blob/main/SQL%20practice%20(project_3_starter)/tennis.ipynb).**

1. Using the same tennis data, find the number of matches played by
   each player in each tournament. (Remember that a player can be
   present as both player1 or player2).
   
   ```
   SELECT "Player1" AS "Player", COUNT("Player1") AS "Matched Played" FROM (
    SELECT aus_men."Player1" FROM aus_men
    UNION ALL
    SELECT aus_men."Player2" FROM aus_men) AS combine
   GROUP BY "Player1"
   ORDER BY COUNT("Player1") DESC
   LIMIT 5;
   ```
   
   * Australian Open MEN: Rafael Nadal (7)
   * French Open MEN: Rafael Nadal (7)
   * Wimbledon MEN: Andy Murray (7), Novak Djokovic (7)   
   * US Open MEN: Rafael Nadal (7)
   

2. Who has played the most matches total in all of US Open, AUST Open, 
   French Open? Answer this both for men and women.
   
   **Note: Combined 4 tournaments by**
   `CREATE VIEW all_men AS...`
   `CREATE VIEW all_women AS...`
   
   ```
   SELECT "Player1" AS "Player", COUNT("Player1") AS "Matches Played"
   FROM all_men
   GROUP BY "Player1"
   ORDER BY COUNT("Player1") DESC
   LIMIT 5;
   ```
   
   * MEN: Rafael Nadal (21)
   
   | Player               | Matches Played |
   | -----------          | -----------    |
   | Rafael Nadal         | 21             |
   | Novak Djokovic       | 17             |
   | David Ferrer         | 17             |
   | Stanislas Wawrinka   | 17             |
   | Roger Federer        | 15             |
   
     
   * WOMEN:
   
   | Player               | Matches Played |
   | -----------          | -----------    |
   | Maria Sharapova	     | 11             |
   | Victoria Azarenka	  | 11             |
   | Serena Williams	     | 11             |
   | Agnieszka Radwanska  | 11             |
   | Jelena Jankovic	     | 9              |
   
   
3. Who has the highest first serve percentage? (Just the maximum value
   in a single match.)
   
   ```
   SELECT "Player1" AS "Player", "FSP.1" AS "First Serve %"
   FROM all_men
   ORDER BY "FSP.1" DESC
   LIMIT 5;
   ```
   
   * MEN: V.Hanescu (85%)
   
   | Player               | Matches Played |
   | -----------          | -----------    |
   | V.Hanescu	           | 85             |
   | Carlos Berlocq	     | 84             |
   | Rafael Nadal	        | 84             |
   | Gael Monfils	        | 84             |
   | Ivan Dodig	        | 83             |
 
 
   * WOMEN: Sara Errani (93%)
   
   | Player               | Matches Played |
   | -----------          | -----------    |
   | Sara Errani	        | 93             |
   | Sara Errani	        | 91             |
   | Sara Errani	        | 89             |
   | Victoria Azarenka	  | 89             |
   | Sara Errani	        | 87             |

4. What are the unforced error percentages of the top three players
   with the most wins? (Unforced error percentage is % of points lost
   due to unforced errors. In a match, you have fields for number of
   points won by each player, and number of unforced errors for each
   field.)

   > **Definition of Unforced Error % (for Player1)**:
   > * Total unforce errors = double faults + unforced error (`DBF.1 + UFE.1`)
   > * Total points lost = Total points won by Player2 (`TPW.2`)
   > * **UFE %** =  `(DBF.1 + UFE.1) / TPW.2`


  
   * MEN:
   
   | Player               | Average UFE % | Total Games Won |
   | -----------          | -----------   | -----------     |
   | Rafael Nadal		     | 0.315         | 59              |
   | Novak Djokovic		  | 0.368         | 47              |
   | Stanislas Wawrinka	  | 0.402         | 46              |
 
 
   * WOMEN:
   
   | Player               | Average UFE % | Total Games Won |
   | -----------          | -----------   | -----------     |
   | Serena Williams		  | 0.491065      | 21              |
   | Victoria Azarenka    | 0.460235      | 20              |
   | Agnieszka Radwanska  | 0.285371      | 18              |
 



  
*Hint:* `SUM(double_faults)` sums the contents of an entire column. For each row, to add the field values from two columns, the syntax `SELECT name, double_faults + unforced_errors` can be used.


*Special bonus hint:* To be careful about handling possible ties, consider using [rank functions](http://www.sql-tutorial.ru/en/book_rank_dense_rank_functions.html).
