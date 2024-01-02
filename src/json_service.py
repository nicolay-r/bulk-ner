import json


class JsonlService:

    @staticmethod
    def write(output, lines_it):
        with open(output, "w") as f:
            for line in lines_it:
                json.dump(line, fp=f)
                f.write("\n")
