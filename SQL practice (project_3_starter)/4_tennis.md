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

1. Using the same tennis data, find the number of matches played by
   each player in each tournament. (Remember that a player can be
   present as both player1 or player2).

2. Who has played the most matches total in all of US Open, AUST Open, 
   French Open? Answer this both for men and women.

3. Who has the highest first serve percentage? (Just the maximum value
   in a single match.)

4. What are the unforced error percentages of the top three players
   with the most wins? (Unforced error percentage is % of points lost
   due to unforced errors. In a match, you have fields for number of
   points won by each player, and number of unforced errors for each
   field.)


*Hint:* `SUM(double_faults)` sums the contents of an entire column. For each row, to add the field values from two columns, the syntax `SELECT name, double_faults + unforced_errors` can be used.


*Special bonus hint:* To be careful about handling possible ties, consider using [rank functions](http://www.sql-tutorial.ru/en/book_rank_dense_rank_functions.html).
