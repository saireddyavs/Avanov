import twint

c = twint.Config()

c.Search=input("Enter sentence::")
c.Limit = 100
c.Email=True

c.Store_csv = True
c.Output = "none"
twint.run.Search(c)
    
