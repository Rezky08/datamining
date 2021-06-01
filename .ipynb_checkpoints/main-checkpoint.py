import twint
c = twint.Config()
# Configure
keywords = ["kasus cod","masalah cod","hapus cod"]
for keyword in keywords:

    c.Search = keyword
    c.Store_csv = True
    c.Lang = 'id'

    c.Output = "SCRAPE\{}.csv".format(keyword)
    c.Limit = 500

    c.Custom["tweet"] = ["username", "tweet"]

    # Run
    twint.run.Search(c)
