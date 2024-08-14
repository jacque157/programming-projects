class UserInputException(Exception):
    pass

class UserInput:
    def check_file(path):
        try:
            with open(path) as file:
                pass
        except Exception as e:
            raise UserInputException(str(e))
    

        

    
