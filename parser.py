import pandas as pd

def merge_company_data():
    company_assets = {}
    company_debt = {}
    company_amount_of_shares = {}
    merged_data = {}
    avg_share_price = {}
    capital = {}
    ROE = {}
    EPS = {}
    revenue = {}

    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            name, *values = line.split()
            help_len = len(values)
            
            for i in range(help_len-1):
                name += " "
                name += values[i]
            assets = float(values[-1])
            company_assets[name] = assets
        except EOFError:
            break
    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            name, *values = line.split()
            help_len = len(values)
            
            for i in range(help_len-1):
                name += " "
                name += values[i]
            debt = float(values[-1])
            company_debt[name] = debt
        except EOFError:
            break
    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            name, *values = line.split()
            help_len = len(values)
            
            for i in range(help_len-1):
                name += " "
                name += values[i]
            shares = float(values[-1])
            company_amount_of_shares[name] = shares
        except EOFError:
            break
    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            name, *values = line.split()
            help_len = len(values)
            
            for i in range(help_len-1):
                name += " "
                name += values[i]
            price = float(values[-1])
            avg_share_price[name] = price
        except EOFError:
            break    

    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            name, *values = line.split()
            help_len = len(values)
            
            for i in range(help_len-1):
                name += " "
                name += values[i]
            ps = float(values[-1])
            capital[name] = ps
        except EOFError:
            break   

    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            name, *values = line.split()
            help_len = len(values)
            
            for i in range(help_len-1):
                name += " "
                name += values[i]
            roe = float(values[-1])
            ROE[name] = roe
        except EOFError:
            break 

    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            name, *values = line.split()
            help_len = len(values)
            
            for i in range(help_len-1):
                name += " "
                name += values[i]
            eps = float(values[-1])
            EPS[name] = eps
        except EOFError:
            break 

    while True:
        try:
            line = input()
            if line.strip() == "":
                break
            name, *values = line.split()
            help_len = len(values)
            
            for i in range(help_len-1):
                name += " "
                name += values[i]
            revenues = float(values[-1])
            revenue[name] = revenues
        except EOFError:
            break 

    # Merge company data
    for name in company_assets:
        if name in company_debt and name in company_amount_of_shares and name in avg_share_price and name in capital and name in ROE and name in EPS and name in revenue:
            merged_data[name] = {
                "assets": company_assets[name],
                "debt": company_debt[name],
                "shares": company_amount_of_shares[name],
                "price": avg_share_price[name],
                "capital": capital[name],
                "ROE": ROE[name],
                "EPS": EPS[name],
                "revenue": revenue[name]
            }

    return merged_data

data = merge_company_data()

df = pd.DataFrame(data).T.reset_index()
df.columns = ["Company name", "Assets", "Debt","Shares","Price","capital","ROE","EPS","revenue"]

# Save DataFrame to CSV file
df.to_csv("companyssssssssss_data.csv", index=False)

