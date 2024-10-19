import csv
import yaml

def dataCheck():
    with open("./configs/config.yaml") as configLoad:
        config = yaml.safe_load(configLoad)
        
        with open(config["outputPath"], newline="", encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for count, page in enumerate(reader):
                print("\n\n============= Page# {0} =============".format(count))
                if page["response"]:
                    print(" - {0} - {1}".format(page["title"], page["score"]))
                    print("Model Generated response: " + page["response"])
                else:
                    print(" - " + page["title"])
                    print(page["page"])
            

if __name__ == '__main__':
    dataCheck()
