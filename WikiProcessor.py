import argparse
import csv
import random
import re
import sys
from typing import Any, Generator, Optional, Tuple

import nltk  # type: ignore
import yaml
from Classifier import Classifier
from datasets import IterableDataset, load_dataset  # type: ignore
from huggingface_hub.hf_api import HfFolder  # type: ignore
from langchain.prompts import PromptTemplate
from mwxml import Dump  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
)


class ResponseGenerator:
    def __init__(self, config: str, useCuda: bool) -> None:
        """
        Class to manage the LLM operations needed for generating raw datasets
        from wikipedia dumps and using that data for response generation

        Args:
            config (str): config path to the config.yaml that defines static
                  variables and problem statement defined constants
            useCuda (bool): flag to allow for use of CUDA on GPU workstations

        Raises:
            Exception: Errors when failure to load config, or initialize external classes
        """
        with open(config) as configLoad:
            try:
                nltk.download("punkt_tab")
                nltk.download("punkt")
                nltk.download("stopwords")
                nltk.download("wordnet")
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words("english"))

                self.config = yaml.safe_load(configLoad)
                self.prompt = self.config["prompt"]
                self.params = self.config["parameters"]
                self.promptTemplate = PromptTemplate(
                    input_variables=["context", "topic"],
                    template=self.prompt["template"],
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.params["model"])
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.params["model"],
                    torch_dtype="auto",
                )
                if useCuda:
                    self.model.cuda()
                self.classifier = Classifier(self.config["classifier"]["model"])
            except yaml.YAMLError as error:
                print(error)
                raise Exception("not a valid config YAML")

    def process_local_dataset(self, genDataset: bool, preprocessData: bool) -> str:
        """
        controller method triggering class operations on local wikipedia dump files

        Args:
            genDataset (bool): if true, only generates preprocessed dataset for later
                training runs or repo upload
            preprocessData (bool): if true will run preprocessing and cleaning steps
                on incoming data (useful with raw dumps)

        Raises:
            Exception: Errors when there are failures retrieving or processing local

        Returns:
            str: Completion notification and output csv location
        """
        try:
            fields = (
                ["title", "page"]
                if genDataset
                else ["title", "score", "page", "response"]
            )
            with open(
                self.config["outputPath"], "w", newline="", encoding="utf8"
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writeheader()
                for pTitle, pText, pTokens in self.__yield_dump_pages(preprocessData):
                    self.__output(pTitle, pText, pTokens, writer, genDataset)
            return (
                "Processing Complete, output written to: " + self.config["outputPath"]
            )
        except Exception as error:
            print(error)
            raise Exception(
                "Error retrieving & processing Contents info from Local Source"
            )

    def process_remote_dataset(self, genDataset: bool, preprocessData: bool, hftoken: str) -> str:
        """
        controller method triggering class operations on remote dataset repositories

        Args:
            genDataset (bool): if true, only generates preprocessed dataset for later
                training runs or repo upload
            preprocessData (bool): if true will run preprocessing and cleaning steps
                on incoming data (useful with raw data)
            hftoken (str): huggingface access token

        Raises:
            Exception: Errors when failure to retrieve or process repository dataset

        Returns:
            str: Completion notification and output csv location
        """
        try:
            data_files = {
                "train": self.config["remote"]["trainData"],
                "test": self.config["remote"]["testData"],
            }
            dataset = load_dataset(
                self.config["remote"]["dataset"],
                split="train",
                data_files=data_files,
                streaming=True,
                token=hftoken,
            )

            fields = (
                ["title", "page"]
                if genDataset
                else ["title", "score", "page", "response"]
            )
            with open(
                self.config["outputPath"], "w", newline="", encoding="utf8"
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writeheader()
                for pTitle, pText, pTokens in self.__yield_remote_pages(
                    dataset, preprocessData
                ):
                    self.__output(pTitle, pText, pTokens, writer, genDataset)
            return (
                "Processing Complete, output written to: " + self.config["outputPath"]
            )
        except Exception as error:
            print(error)
            raise Exception(
                "Error retrieving & processing Contents info from Repository"
            )

    def __output(
        self,
        pTitle: str,
        pText: str,
        pTokens: list[list],
        writer: csv.DictWriter,
        genDataset: bool,
    ) -> None:
        """
        Helper method to write out results to CSV output file on disk
        """
        try:
            if genDataset:
                writer.writerow({"title": pTitle, "page": pText})
                if self.config["debug"]["printResults"]:
                    print("\n\n{0}, Token Count:{1}".format(pTitle, len(pTokens[0])))
            else:
                print("processing...", pTitle)
                response = self.__generate_q_a(pTokens)
                score = self.classifier.scoreResult(response)
                writer.writerow(
                    {
                        "title": pTitle,
                        "score": score,
                        "page": pText,
                        "response": response,
                    }
                )
                if self.config["debug"]["printResults"]:
                    print(
                        "\n\n{0} - Score:{1}\n response: {2}".format(pTitle, score, response)
                    )
        except Exception as error:
            print(error)
            raise Exception("Error writing result to disk(csv)")

    def __yield_remote_pages(
        self, dataPages: IterableDataset, preprocessData: bool
    ) -> Generator[Any, Any, Any]:
        """
        Generator method that parses through the dataset pages from remote, and streams
        (yields) out the results to output file to prevent memory issues with extremely large datasets

        Yields:
            Generator[Tuple[str, str, BatchEncoding]]: generated tuple (Title, PageText, Preprocessed Tokens)
        """
        for count, page in enumerate(dataPages):
            if count > self.config["debug"]["pageLimiter"]:
                break
            if random.randint(0, 99) >= self.config["debug"]["randSample"]:
                continue
            result = self.__preprocess_tokenize_page(
                page["title"], page["page"], preprocessData
            )
            if result:
                yield result

    def __yield_dump_pages(self, preprocessData: bool) -> Generator[Any, Any, Any]:
        """
        Generator method that parses through the dataset pages from local file, and streams
        (yields) out the results to output file to prevent memory issues with extremely large datasets

        Yields:
            Generator[Tuple[str, str, BatchEncoding]]: generated tuple (Title, PageText, Preprocessed Tokens)
        """
        # Construct dump file iterator
        dump = Dump.from_file(open(self.config["dumpPath"], encoding="utf8"))
        # Iterate through pages
        for count, page in enumerate(dump.pages):
            # Iterate through a page's revisions
            for revision in page:
                # page processing limiter & random sampling
                if count >= self.config["debug"]["pageLimiter"]:
                    break
                if random.randint(0, 99) >= self.config["debug"]["randSample"]:
                    continue
                result = self.__preprocess_tokenize_page(
                    page.title, revision.text, preprocessData
                )
                if result:
                    yield result

    def __preprocess_tokenize_page(
        self, pTitle: str, pText: str, preprocessData: bool
    ) -> Optional[Tuple[str, str, BatchEncoding]]:
        """
        Helper method to organize the processing of the page
        includes: cleaning, tokenization, filtering of unwanted pages (too large or too small)

        Returns:
            Optional[Tuple[str, str, BatchEncoding]]: tuple (Title, PageText, Preprocessed Tokens)
        """
        cleanText = self.__clean_page(pText) if preprocessData else pText
        tokenIn = self.__tokenize_page(pTitle, cleanText)
        print("preprocessing...{0}, Token Count: {1}".format(pTitle, len(tokenIn[0])))
        if self.params["minTokens"] < len(tokenIn[0]) < self.params["maxTokens"]:
            return (pTitle, cleanText, tokenIn)
        return None

    def __clean_page(self, text: str) -> str:
        """
        Helper method utilizing traditional NLP techniques to
        remove noise, normalize, filter the incoming text

        Returns:
            str: cleaned version of the original text
        """
        if not text:
            return ""
        # Remove file & web references
        text = re.sub(
            r"\[\[File(.*?)\.\]\]|\[\[File(.*?)\]\]|\[http(.*?)\]|\{\{(.*?)\}\}|\<ref(.*?)\<\/ref\>|\[\[Category\:(.*?)\]\]",
            "",
            text,
        )

        # Tokenization into words for NLP based cleaning
        tokens = word_tokenize(text)

        # Remove Noise, lower, and special characters
        tokens = [re.sub(r"[^\w\s]", "", token) for token in tokens]
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if word.isalnum()]

        # Removing stop-words and Lemmatize
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    def __tokenize_page(self, pTitle: str, pText: str) -> BatchEncoding:
        """
        Helper Method to format and tokenize the prompt (& page context)
        for consumption by the llm for processing

        Returns:
            BatchEncoding: The encodings set containing the tokens and attention mask
        """
        prompt = self.promptTemplate.format(
            context=pText, topic=self.prompt["topic"].format(topic=pTitle)
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        return inputs

    def __generate_q_a(self, tokenInput: BatchEncoding) -> str:
        """
        Primary LLM processing method, runs the prompt with preprocessed data
        through the stabilityLM model using either a greedy generation or
        a sampling based generation


        Returns:
            str: returns the text generated by the model
        """
        if not self.params["doSample"]:
            # greedy
            outputTokens = self.model.generate(
                **tokenInput,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.params["maxNewTokens"],
                repetition_penalty=self.params["repetitionPenalty"],
                do_sample=self.params["doSample"],
                num_beams=self.params["numBeams"],
            )
        else:
            # sampling
            outputTokens = self.model.generate(
                **tokenInput,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.params["maxNewTokens"],
                temperature=self.params["temperature"],
                top_p=self.params["topP"],
                top_k=self.params["topK"],
                typical_p=self.params["typicalP"],
                min_p=self.params["minP"],
                repetition_penalty=self.params["repetitionPenalty"],
                do_sample=self.params["doSample"],
            )

        response = self.tokenizer.decode(
            outputTokens[0][len(tokenInput[0]) :], skip_special_tokens=True
        )
        return response


def set_up_args(parser: argparse.ArgumentParser) -> None:
    """
    Simple Helper function to set up CLI arguments and options

    Arguments:
        parser -- argument parser object
    """
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="[OPTIONAL]  If specified will use alternate config file for\
            setting static parameters (default ./configs/config.yaml)",
    )
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        default=False,
        help="[OPTIONAL] Flag to use local dataset dump, path defined in config.\
            (Defult False, uses remote huggingface dataset)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action="store_true",
        default=False,
        help="[OPTIONAL] Flag to skip llm processing step and only clean and\
            prepare an article dataset (Default False, process and score with LLMs)",
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        action="store_true",
        default=False,
        help="[OPTIONAL] Flag to enable preprocessing and cleaning steps for\
            working with raw data (default False, skip preprocessing)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="[OPTIONAL] Flag to enable the use of CUDA for GPU environments",
    )
    parser.add_argument(
        "-t",
        "--hftoken",
        default="",
        help="[OPTIONAL] if using remote repository, include your you hugging-face auth token here",
    )
    return


def main(argv) -> None:
    try:
        parser = argparse.ArgumentParser()
        set_up_args(parser)
        args = parser.parse_args()
        if not args.local:
            HfFolder.save_token(args.hftoken)
        ResponseGen = ResponseGenerator(args.config, args.cuda)
        if args.local:
            conclusion = ResponseGen.process_local_dataset(args.dataset, args.preprocess)
        else:
            conclusion = ResponseGen.process_remote_dataset(args.dataset, args.preprocess, args.hftoken)
        print(conclusion)

    except Exception as error:
        print(str(error) + "...Exiting Program")
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
