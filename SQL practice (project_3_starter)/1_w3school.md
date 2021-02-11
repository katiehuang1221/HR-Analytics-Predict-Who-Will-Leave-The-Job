# Part 1: W3Schools SQL Lab 

*Introductory level SQL*

--

This challenge uses the [W3Schools SQL playground](http://www.w3schools.com/sql/trysql.asp?filename=trysql_select_all). Please add solutions to this markdown file and submit.

#### 1. Which customers are from the UK?
>
> ```
> SELECT CustomerName FROM Customers
> WHERE Country = 'UK';
> ```

#### 2. What is the name of the customer who has the most orders?
> 
> 
> ```
> SELECT CustomerName FROM(
> 	SELECT * FROM Orders
>     JOIN Customers
>     ON Orders.CustomerID = Customers.CustomerID)
> GROUP BY CustomerID
> Order BY COUNT(OrderID) DESC
> LIMIT 1;
> ```
> 
> Ernst Handel


#### 3. Which supplier has the highest average product price?
> 
> ```
> SELECT SupplierName, AVG(Price) FROM(
> 	SELECT * FROM Suppliers
> 	JOIN Products
> 	ON Products.SupplierID = Suppliers.SupplierID)
> GROUP BY SupplierID
> ORDER BY AVG(Price) DESC
> LIMIT 1;
> ```
> 
> Aux joyeux ecclésiastiques, 140.75

#### 4. How many different countries are all the customers from? (*Hint:* consider [DISTINCT](http://www.w3schools.com/sql/sql_distinct.asp).)
> 
> ```
> SELECT DISTINCT Country FROM Customers;
> ```
> 
> 21


#### 5. What category appears in the most orders?
> 
> ```
> WITH count_order AS (
> 	SELECT * FROM OrderDetails
> 	JOIN Products
> 	ON OrderDetails.ProductID = Products.ProductID
> 	GROUP BY CategoryID
> 	ORDER BY COUNT(OrderID) DESC)
> SELECT CategoryName FROM count_order
> JOIN Categories
> ON Categories.CategoryID = count_order.CategoryID
> LIMIT 1;
> ```
> Dairy Products

#### 6. What was the total cost for each order?
> 
> ```
> SELECT OrderID,SUM(Price) FROM(
> 	SELECT * FROM OrderDetails
> 	JOIN Products
> 	ON OrderDetails.ProductID = Products.ProductID
>     )
> GROUP BY OrderID;
> ```


#### 7. Which employee made the most sales (by total price)?
> 
> ```
> SELECT EmployeeID,OrderID,SUM(Price) FROM(
> 	SELECT * FROM OrderDetails
> 	JOIN Products
> 	ON OrderDetails.ProductID = Products.ProductID
>     JOIN Orders
>     ON Orders.OrderID = OrderDetails.OrderID
>     JOIN Employees
>     ON Employees.EmployeeID = Orders.EmployeeID
>     )
> GROUP BY OrderID
> ORDER BY SUM(Price) DESC
> LIMIT 1;
> ```
> Peacock, Margaret

#### 8. Which employees have BS degrees? (*Hint:* look at the [LIKE](http://www.w3schools.com/sql/sql_like.asp) operator.)
> 
> ```
> SELECT * FROM Employees
> WHERE Notes LIKE '%BS%';
> ```
> Leverling,	Janet
> Buchanan,	Steven

#### 9. Which supplier of three or more products has the highest average product price? (*Hint:* look at the [HAVING](http://www.w3schools.com/sql/sql_having.asp) operator.)
> ```
> SELECT SupplierName, AVG(Price) FROM Suppliers
> JOIN Products
> ON Products.SupplierID = Suppliers.SupplierID
> GROUP BY Suppliers.SupplierID
> HAVING COUNT(ProductID) >= 3
> ORDER BY AVG(Price) DESC;
> ```
> Tokyo Traders
