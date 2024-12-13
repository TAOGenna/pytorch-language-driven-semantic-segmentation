import argparse

class Options: 
    def __init__(self):
        parser = argparse.ArgumentParser(description="pet project")
        parser.add_argument("--model", type=str, default="modelA", help="Model name")
        self.parser = parser
    def parse(self):
        return self.parser.parse_args()

if __name__ == '__main__':
    opts = Options().parse()
    print(opts.model)
    print(opts)