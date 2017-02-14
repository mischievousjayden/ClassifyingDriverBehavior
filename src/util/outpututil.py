import os

class ConsoleFileOutput:
    def __init__(self, filename):
        try:
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.fp = open(filename, "w")
        except OSError as exc:
            self.fp = None
            raise

    def __del__(self):
        if self.fp != None:
            self.fp.close()

    def output(self, outstr):
        self.fp.write(outstr+"\n")
        print(outstr)

