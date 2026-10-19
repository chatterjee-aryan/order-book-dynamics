import databento as db

# Establish connection and authenticate
client = db.Historical("db-4E9DqE6XPHwry9h4rrnSeSqP3fq3H")

# Authenticated request
print(client.metadata.list_datasets())
