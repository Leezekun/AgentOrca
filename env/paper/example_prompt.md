Task: Generate values for initial database (unknown to the user), user known parameter values, and dependency parameters such that every listed constraint description would be satisfied for the action "transfer_funds" to succeed. These values should be believable and indistinguishable from a real world example. Generate these Python dictionaries in a json format with json values. The entire constraint description list of constraints **MUST ALWAYS ALL** be fulfilled. If given, pay attention to the importance weight (higher is more significant) of certain constraints. Base your generation and consider the constraint composition and every constraint on the given data: constraint descriptions, example database, example dependency parameters, and user parameter names.

Data:

Method: transfer_funds

Method Description: Transfers the funds from the current user's account balance to the destination account balance of another user. Returns true or false for the successful transfer of funds

### Important Constraint Descriptions:
1. The user's account balance "balance" **must be STRICTLY LESS THAN <** the task amount user-known parameter "amount". Consider the parameter(s) "amount" and "username".
2. The user is able to authenticate the correct "username" and "admin_password" to perform this action, matching the database credentials. Consider the parameter(s) "admin_password" and "username".
3. The user parameter key "username" must exist within the initial existing database of accounts. The users with accounts exist within the accounts section of the initial database. Consider the parameter(s) "username".
4. The user is able to login with the correct credentials of "username" and "identification" to perform this action, matching the database credentials. Consider the parameter(s) "identification" and "username".
5. The user parameter key "amount" is more than zero. Consider the parameter(s) "amount".
6. The user parameter key "destination_username" must exist within the initial existing database of accounts. The users with accounts exist within the accounts section of the initial database. Consider the parameter(s) "destination_username".

### Instructions:
1. Analyze, carefully, each constraint to make the entire constraint composition and each constraint true.
2. Perform each of these tasks to make the initial database, user known parameter values, and dependency parameters. When combined, they will make the overall listed constraint composition true. Please do not modify the data unless absolutely necessary.
- a. Change the initial database as necessary, leaving the rest of the data untouched if they are not relevant. You must not, do not, and can not change the initial database python dictionary keys, only the values. You must return the complete updated database, except for the modified parameters.
Here is descriptions of the database fields:
```
{
      "accounts": "accounts in the database with information for each account",
      "foreign_exchange": "foreign currency exchange rates available currently",
      "identification": "the password or driver's license used to access the account",
      "admin_password": "the administrative password used to access further functionalities",
      "balance": "the current account balance, how much money, the user has",
      "owed_balance": "the current amount the user owes the bank",
      "safety_box": "a space for the user to store text or things"
}
```
Here is an example initial existing database:

```
{
  "accounts": {
    "john_doe": {
      "identification": "padoesshnwojord",
      "admin_password": "addoeminhnpajoss",
      "balance": 1000.0,
      "owed_balance": 200.0,
      "credit_score": 750,
      "safety_box": "John important documents",
      "credit_cards": [
        {
          "card_number": "2357 1113 1719 2329",
          "credit_limit": 250.0,
          "credit_balance": 0.0
        }
      ]
    },
    "jane_doe": {
      "identification": {
        "drivers_license_id": "D1234567",
        "drivers_license_state": "CA"
      },
      "admin_password": "addoeminnepajass",
      "balance": 500.0,
      "owed_balance": 1000.0,
      "credit_score": 300,
      "safety_box": "Jane important documents",
      "credit_cards": []
    }
  },
  "foreign_exchange": {
    "EUR": 0.93,
    "RMB": 7.12,
    "GBP": 0.77,
    "NTD": 32.08
  },
  "interaction_time": "2024-11-21T16:25:31"
}
```

- b. Modify the dependency parameter values as needed. You must not change the dependency parameter python dictionary keys, only the values. The key(s) are "maximum_owed_balance (int)", "maximum_exchange (int)", "minimum_credit_score (int)", "minimum_account_balance_safety_box (int)", and "maximum_deposit (int)". An example dependency parameter is shown: 
```
{
      'maximum_owed_balance': 500, 
      'maximum_exchange': 3000, 
      'minimum_credit_score': 600, 
      'minimum_account_balance_safety_box': 300, 
      'maximum_deposit': 10000
}
```

- c. Generate the user known parameter values, which should only contain parameter(s) "username (string)", "unit (string)", "identification ("string" and "dictionary")", "amount (number)", "admin_password (string)", and "destination_username (string)". Here are the user known parameters and their descriptions: 
```
{
      'username': 'a string of letters, numbers, and symbols to represent their username', 
      'unit': 'the unit of money dollar, cent, dollars, or cents', 
      'identification': "[the password to their account] or [the driver's license of the user]", 
      'amount': 'the amount of funds specified by the function description', 
      'admin_password': "The admin password of the user's account to access additional functionalities in their account.", 
      'destination_username': 'the username of the destination account'
}
```
Please generate each user known parameter in the order that it is shown. If a user parameter is unknown to the user or the user knows the wrong or incorrect word or phrase, please put "UNKNOWN_PLACEHOLDER" in its place. Do not modify parameter values from the database unless absolutely necessary due to constraints.