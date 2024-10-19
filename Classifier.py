from transformers import AutoTokenizer, AutoModelForSequenceClassification # type: ignore

class Classifier:
    def __init__(self, modelName: str):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(modelName)
            self.model = AutoModelForSequenceClassification.from_pretrained(modelName)
            self.model.cuda()
        except Exception as error:
            print(error)
            raise Exception("Error Initializing Classifier")

    def scoreResult(self, response: str):
        if not response:
            return {
                "score": 0,
                "int_score": 0,
            }
        inputs = self.tokenizer(response, return_tensors="pt", padding="longest", truncation=True).to(self.model.device)
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze(-1).float().detach().cpu().numpy()
        score = logits.item()
        classResult = {
            "score": score,
            "int_score": int(round(max(0, min(score, 5)))),
        }
        print(classResult)
        return classResult
