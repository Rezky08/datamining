import twint
c = twint.Config()
# Configure
keywords = ["kasus cod","masalah cod","hapus cod"]
def run_twint_search_keywords(keywords):
    c = twint.Config()
    for keyword in keywords:
        print("Searching {}".format(keyword))
        c.Search = keyword
        c.Store_csv = True
        c.Lang = 'id'
        c.Hide_output = True
        c.Output = "SCRAPE\{}.csv".format(keyword)
        c.Limit = 500

        c.Custom["tweet"] = ["username", "tweet"]

        # Run
        twint.run.Search(c)
run_twint_search_keywords(keywords)