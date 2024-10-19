
# LLM Wiki Reader
LLM wrapping application that can use an configurable LLM model to process, and use Wiki articles as context for configurable query prompts. (dockerized and CUDA enabled)

## Summary
This is a command line tool to manage the use of any arbitrary LLM for generating configurable prompt responses using Wiki articles as context. Along with the dataset generation and scoring systems needed to support this. 

It is a 3 step process with configuration options to govern or omit any step:
- Preprocessing (cleaning, noise reduction, Lemmatizing, filtering out invalid pages)
- Processing (running the prompt and context article through the Q & A generating LLM)
- Scoring (Using FineWeb-edu classifier to evaluate and academic value score the results)

### Example
Input:
```zsh
$ python WikiProcessor.py --cuda -t <your huggingfaceToken>
```

Output:
```zsh
============= Page# 1 =============
 - Alan Turing - {'score': 1.5386065244674683, 'int_score': 2}
Model Generated response: 1. What was Alan TURINGs nationality? A British B American C Australian
Answer: A
```

# Use With Docker
## How to install
Install docker on you machine

This Docker container uses CUDA so you will need Nvidia drivers 

(If you do not have a GPU you can still run program without the --cuda flag, but you will need to force install the drivers, just ignore the no GPU present warning. Running locally is easier alternative for non GPU workstations.)

- install nvidia container toolkit if linux/mac
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

- use WSL2 if windows
https://docs.nvidia.com/cuda/wsl-user-guide/index.html 

Build the docker container:
```zsh
$ docker buildx build . -t llmwiki:v1
```

## How to Run
```zsh
$ docker run --gpus all llmwiki:v1 python WikiProcessor.py --cuda -t <your huggingfaceToken>
```

# Use Locally Python venv
## How to install
Set up a clean python virtual env and install the requirements.txt

```zsh
$ virtualenv <env_name>
$ source <env_name>/bin/activate

<for windows>
$ .\env\Scripts\activate

(<env_name>)$ pip install -r path/to/requirements.txt
```

## How to Run
```zsh
$ python  WikiProcessor.py --cuda -t <your huggingfaceToken>
```


# Arguments and Options
```
usage:  WikiProcessor.py [-h] [-c CONFIG] [-l] [-d] [-p] [--cuda] [-t HFTOKEN]

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        [OPTIONAL] If specified will use alternate config file
                        for setting static parameters (default
                        ./configs/config.yaml)
  -l, --local           [OPTIONAL] Flag to use local dataset dump, path
                        defined in config. (Defult False, uses remote
                        huggingface dataset)
  -d, --dataset         [OPTIONAL] Flag to skip llm processing step and only
                        clean and prepare an article dataset (Default False,
                        process and score with LLMs)
  -p, --preprocess      [OPTIONAL] Flag to enable preprocessing and cleaning
                        steps for working with raw data (default False, skip
                        preprocessing)
  --cuda                [OPTIONAL] Flag to enable the use of CUDA for GPU
                        environments
  -t HFTOKEN, --hftoken HFTOKEN
                        [OPTIONAL] if using remote repository, include your
                        you hugging-face auth token here
```
### Recommended Use
For general running using the pre cleaned data stored in the Hugginface Repo -
(you will need your hugging face auth token for any calls using my private remote datasets)
```zsh
$ python WikiProcessor.py --cuda -t <your huggingfaceToken>
```
When you are dealing with raw data, such as a fresh wiki dump you want to use you can run the entire process like this:
```zsh
$ python WikiProcessor.py --cuda -l -p
```
or to simply generate the processed/cleaned dataset for ease of iteration later just use the Dataset option
```zsh
$ python WikiProcessor.py --cuda -l -p -d
```

you can always choose to leave off the `--cuda` flag but your performance will be considerably slower

All of the fields you could need to customize your run can be found and edited in the `config.yaml`
such as: [remote repository datasets, local input file, output file, model tuning parameters, prompt template, debug options]

to reduce runtime for testing and iteration I added a hard cutoff `pageLimiter` config that will stop processing after N pages

To ensure variability and prevent over-fit I added `randSample` which allows for a very simplistic sampling method (pseudo-random chance to select an article)

### Additional Tools
This program includes a data reader tool that allows for human readable console logging of the output
This is purely for convenience, the output is still stored in csv format for proper consumption
```zsh
$ docker run --gpus all llmwiki:v1 python DataReader.py
 or
$ python DataReader.py
```

### Config files
This program includes a config file directory. 
These files store the static configurable information that can be tweaked for different run/processing properties and input/output sources. I made them configurable values for later expansion and flexibility

You can modify the existing config, or always create a custom config file to pass in as an argument