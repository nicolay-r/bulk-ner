import json


class JsonlService:

    @staticmethod
    def write(output, lines_it):
        with open(output, "w", encoding='utf8') as f:
            for line in lines_it:
                json.dump(line, fp=f, ensure_ascii=False)
                f.write("\n")
