# source: https://stackoverflow.com/a/6361028/12828249


class Printer:

    """
    this class prints the passed data on one line dynamically
    it is used throughout this project to provide updates
    of what a function is doing
    """

    @staticmethod
    def print(data: str):
        if data:
            sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()
