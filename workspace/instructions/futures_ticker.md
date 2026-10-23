# Utility functions to generate futures ticker from date

Dependency instructions : futures_ticker_convention.md

## What I want to do

I want utility functions, one for each of the futures listed in futures_ticker_convention.md. The functions should take a year and month. The conventions contain the eligible months too. Note, do not mandate the month code. Month can be numberic like 1 for Jan, 12 for Dec. Throw an error if the month supplied is not eligible. If not, convert to valid ticker using the convention which you should keep in the file using a dictionary and return
