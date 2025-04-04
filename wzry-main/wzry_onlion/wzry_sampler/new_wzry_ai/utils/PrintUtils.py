class PrintUtils:
    def __init__(self):
        pass

    def print_green(self,text):
        print(f"\033[92m{text}\033[0m")

print_utils=PrintUtils()