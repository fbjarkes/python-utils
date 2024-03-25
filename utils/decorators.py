
def try_except(func):
    def handler(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e) #TODO: user logger

    return handler