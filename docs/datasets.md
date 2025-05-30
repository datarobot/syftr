# syftr Datasets

## Dataset Structure and Format

syftr datasets have two main subsets:

1. `groundingdata`: Contains the grounding data used in RAG scenarios.
2. `qapairs`: question-answer pairs.

syftr uses Hugging Face [datasets](https://huggingface.co/docs/datasets/en/index) format for data loading and processing. See [DataRobot-Research](https://huggingface.co/DataRobot-Research) organization for examples of already prepared datasets used in syftr.

The `qapairs` are further divided into the following splits:

* `train`: used for any training of flows. E.g., if dynamic few-shot prompting is selected, the this split is used to find suitable dynamic few-shot examples.
* `test`: during optimization, flows are evaluated against the question answer pairs this split.
* `sample`: a small sample of the question answer pairs usually drawn from the `test` split for quick sanity checks, manual inspection and unit tests.
* `holdout`: in order to not overfit to the test set, we keep a holdout set which is not used during optimization but only for final evaluation of the flows.

### Concrete Example

Let us begin by examining the [HotPotQA](https://huggingface.co/datasets/DataRobot-Research/hotpotqa) dataset. This dataset has been created from the raw HotPotQA dataset available [here](https://hotpotqa.github.io/). The script parses the raw dataset and creates a Hugging Face dataset with the following subsets:

* `qapairs_dev`: question answer pairs from `dev` set
* `qapairs_train_hard`: question answer pairs from `train_hard` set
* `qapairs_synthetic_dev`: synthetic question answer pairs which use the `groundingdata_dev` set as the grounding data
* `qapairs_synthetic_train_hard`: synthetic question answer pairs which use the `groundingdata_train_hard` set as the grounding data
* `groundingdata_dev`: grounding data created by concatenating the `context` field of all the question answer pairs in the `dev` set 
* `groundingdata_train_hard`: grounding data created by concatenating the `context` field of all the question answer pairs in the `train_hard` set

The `groundingdata_*` subsets contain the grounding data to be used for RAG workflows while the `qapairs_*` subsets contain the question answer pairs to be used for training and evaluation of the models. This is a convention we follow for all the datasets we create for syftr. This convention is not necessary though and any convention can be used as we will shortly see below. (See the [`InfiniteBench` class](../syftr/storage.py)) It is not even necessary to use Hugging Face dataset formats. syftr can work with data from any source.

## Loading data into syftr

The [`SyftrQADataset`](../syftr/storage.py) class is the interface class for any dataset to be used with syftr. It is a [Pydantic](https://docs.pydantic.dev/latest/) model which defines the structure of the dataset and provides methods to load and access the data. The two most important methods in this class are `def iter_examples(self, partition="test", use_ray=False) -> T.Iterator[QAPair]` and `def iter_grounding_data(self, **load_kwargs) -> T.Iterator[Document]`. The former iterates over the question answer pairs in the specified partition (e.g. `train`, `test`, etc.) while the latter iterates over the grounding data. The `SyftrQADataset` class provides default implementations assuming data in a certain format but this can be easily overridden by creating a custom dataset class which inherits from `SyftrQADataset` and implements the required methods.
We encourage the reader to inspect this class and then the `HotPotQAHF` class which inherits from it and implements the methods for the [HotPotQA](https://huggingface.co/datasets/DataRobot-Research/hotpotqa) dataset:

```python
class HotPotQAHF(SyftrQADataset):
    xname: T.Literal["hotpotqa_hf"] = "hotpotqa_hf"  # type: ignore
    subset: str = "dev"  # train_hard, dev
    description: str = (
        "This dataset is a vast collection of all kind of information that you can find on Wikipedia. "
        "It can be used, for instance, to retrieve straightforward facts from one or more documents, "
        "compare two entities based on shared attributes, "
        "identify relationships, roles, or attributes of entities, "
        "reason about dates, timelines, or chronological order, "
        "determine geographical relationships or locations, "
        "explain causes or sequences of events or processes, "
        "synthesize facts from multiple documents to infer answers, and "
        "validate or refute premises in the context of the question."
    )

    @property
    def name(self) -> str:
        return f"{self.xname}/{self.subset}"

    def _load_grounding_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "DataRobot-Research/hotpotqa",
                f"groundingdata_{self.subset}",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset

    def _load_qa_dataset(self) -> datasets.Dataset:
        with distributed_lock(
            self.name, timeout_s=self.load_examples_timeout_s, host_only=True
        ):
            dataset = datasets.load_dataset(
                "DataRobot-Research/hotpotqa",
                f"qapairs_{self.subset}",
                cache_dir=cfg.paths.huggingface_cache,
            )

        return dataset

    def iter_grounding_data(
        self, partition="test", **load_kwargs
    ) -> T.Iterator[Document]:
        assert partition in self.storage_partitions
        grounding_dataset = self._load_grounding_dataset()
        partition = self._get_storage_partition(partition)
        for row in grounding_dataset[partition]:
            yield Document(
                text=row["text"],
            )

    def _row_to_qapair(self, row):
        """Dataset-specific conversion of row to QAPair struct.

        Invoked by iter_examples.

        Default implementation assumes row is already in QAPair format.
        """
        return QAPair(
            question=row["question"],
            answer=row["answer"],
            _id=row["id"],
            context=[{title: sentence} for title, sentence in row["context"]],
            supporting_facts=row["supporting_facts"],
            difficulty=row["level"],
            qtype=row["type"],
        )

    def iter_examples(self, partition="test", use_ray=False) -> T.Iterator[QAPair]:
        partition = self._get_storage_partition(partition)
        qa_examples = self._load_qa_dataset()
        for row in qa_examples[partition]:
            yield self._row_to_qapair(row)
```

`iter_grounding_data()` loads the grounding data using `_load_grounding_dataset()` method and then iterates over the specified partition (e.g. `train`, `test`, etc.) to yield [LlamaIndex `Document`](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/) objects.

```python
def _row_to_qapair(self, row):
    """Dataset-specific conversion of row to QAPair struct.

    Invoked by iter_examples.

    Default implementation assumes row is already in QAPair format.
    """
    return QAPair(
        question=row["question"],
        answer=row["answer"],
        _id=row["id"],
        context=[{title: sentence} for title, sentence in row["context"]],
        supporting_facts=row["supporting_facts"],
        difficulty=row["level"],
        qtype=row["type"],
    )

def iter_examples(self, partition="test", use_ray=False) -> T.Iterator[QAPair]:
    partition = self._get_storage_partition(partition)
    qa_examples = self._load_qa_dataset()
    for row in qa_examples[partition]:
        yield self._row_to_qapair(row)
```

Similarly `iter_examples()` loads the question answer pairs using `_load_qa_dataset()` method and then iterates over the specified partition (e.g. `train`, `test`, etc.) to yield [QAPair](../../syftr/types.py) objects. The `_row_to_qapair()` method is used to convert each row of the dataset to a `QAPair` object.

### Using partition maps

The dataset partitions `test` and `train` have specific meanings within `syftr` - `train` can be used during the build phase, such as for building few-shot example indices, while `test` must be used during evaluation.

`PartitionMap` allows us to change which actual partition is used when `syftr` requests to iterate over the `train` or `test` partition. For example, setting `PartitionMap(test="sample")` will ensure that the `sample` set of QA pairs and grounding data is used at evaluation time:

```python

from syftr.storage import PartitionMap, FinanceBenchHF

dataset = FinanceBenchHF(partition_map=PartitionMap(
    test="sample"
))

## Prepared datasets

At the time of writing we have the following pre-processed datasets available in the [DataRobot-Research](https://huggingface.co/DataRobot-Research) organization for ready usage with syftr. We will continue to add more datasets and also dataloaders from different sources as we use syftr on more use-cases:

* [DataRobot-Research/hotpotqa](https://huggingface.co/datasets/DataRobot-Research/hotpotqa): [HotpotQA](https://hotpotqa.github.io/) is a large-scale question-answering dataset designed to promote
research in multi-hop reasoning. It contains approximately 113, 000 QA pairs, where answering each question
requires synthesizing information from multiple documents. The dataset emphasizes diverse reasoning skills,
including multi-hop reasoning, comparison, and causal inference, making it a valuable benchmark for RAG
flows. Each QA pair comes with one or more Wikipedia page fragments, which are used as grounding data.
We use the train_hard subset of HotpotQA, which has 15, 661 of the toughest questions and is split into
separate sample, train, test, and holdout partitions with 20, 7305, 500, and 7836 QA pairs, respectively.

* [DataRobot-Research/crag](https://huggingface.co/datasets/DataRobot-Research/crag): The [CRAG (Comprehensive RAG)](https://github.com/facebookresearch/CRAG) benchmark dataset from Meta was introduced for
KDD Cup 2024. The [AIcrowd 2024](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024) challenge contains three tasks - Task 1: retrieval summarization,
Task 2: knowledge graph and web retrieval, and Task 3: end-to-end RAG. We use the Task 3 dataset only, as this
is the closest task to the RAG task syftr is built to optimize. CRAG Task 3 contains 4, 400 QA pairs on a
variety of topics. The official Task 3 is to perform RAG over 50 web pages fetched from an Internet search engine
for each question. We attempted a different task, which is to perform RAG over all of the web pages included in
the dataset. To reduce the size of the data required for embedding and evaluation, we split the dataset into five
datasets according to the five question topics - finance, movies, music, sports, and open-domain. We further
partitioned each dataset into sample, train, test, and holdout partitions containing 5%, 42.5%, 42.5%, and
10% of the QA pairs, respectively.
The web page results for the QA pairs in each dataset and partition were used as the grounding data for RAG.
Text from the provided HTML files was converted to Markdown format using the [html2text](https://github.com/aaronsw/html2text) library.
The questions in CRAG typically contain challenging trivia about specific celebrities, events, or media, often
requiring multi-hop lookup and linguistic or numerical reasoning.
Note that our task setting differs significantly from that of the official CRAG Task 3 benchmark. We don’t
enforce a maximum generation time, don’t restrict ourselves to Llama-based models only, and perform RAG
over the entire corpus of grounding data rather than the 50 web results specific to each QA pair. Due to this, our
accuracy and latency results cannot be directly compared to the contest submissions.

* [DataRobot-Research/financebench](https://huggingface.co/datasets/DataRobot-Research/financebench): [Financebench](https://github.com/patronus-ai/financebench) is a difficult RAG-QA dataset in the financial domain. The
public test set includes 150 questions of varying difficulty, from single-hop qualitative lookup questions to
multi-hop questions requiring complex numeric computations. It also includes 368 PDF files containing SEC
filing documents from 43 companies over a seven year timespan. Answering questions using this dataset
typically requires retrieving specific facts and metrics from the appropriate document by company, filing
type, and time period. This is an important dataset, as it combines real-world use cases of computer-assisted
financial analysis and challenges of precise information retrieval from semi-structured PDF documents, with
the challenges of complex information retrieval and reasoning systems. These aspects are ubiquitious across
enterprises today.
We split the dataset into roughly equal-sized train, test, and holdout partitions (53, 49, and 48 QA pairs, respectively), with each partition having roughly equal number of companies represented. The PDFs are also
split by these partitions based on the company, so that each partition only has PDFs from companies in the
question set. This allows us to reduce the amount of grounding data in each partition, lowering the cost of
optimization, while each partition still contains a significant amount of “distractor” data. The sample partition
is drawn from the test partition and contains 11 QA pairs about PepsiCo. The PDF files were converted into
markdown format using [Aryn DocParse](https://www.aryn.ai/).

* [DataRobot-Research/drdocs](https://huggingface.co/datasets/DataRobot-Research/drdocs): The DRDocs dataset contains QA pairs about the [DataRobot](https://www.datarobot.com/) product suite, including
GUI, API, and SDK usage, and it contains a snapshot of the entire DataRobot documentation codebase. We
created the DRDocs dataset synthetically using our synthetic dataset generator. The dataset contains 100 QA pairs, split into train, test,
and holdout partitions of 10, 80, and 10 questions each.

## Adding a new dataset

### Step 1: Upload dataset to HuggingFace

To add a new dataset to Syftr, we recommend uploading it to Hugging Face in the format used by syftr. This allows for easy integration and usage with the existing syftr infrastructure. The dataset should be structured as described above, with `groundingdata` and `qapairs` subsets, and the `qapairs` subset should be further divided into `train`, `test`, `sample`, and `holdout` partitions.

This is not strictly necessary, however. syftr can work with any dataset format as long as the `SyftrQADataset` class is implemented correctly to load the data.

### Step 2: Write syftr dataloader

Adding a dataset is as simple as sub-classing the `SyftrQADataset` class and implementing the required methods. While we recommend uploading your processed dataset to Hugging Face using the convention adopted by the syftr team, that is not necessary. See the [`InfiniteBench` class](../syftr/storage.py) for using a Hugging Face dataset which uses another schema for the data or the `SyftrQADataset` class itself for using LlamaIndex's `SimpleDirectoryReader` for loading data from a local directory.

Please create a new issue on Github if you would like to contribute a new dataset to the syftr project or for assistance wrapping your dataset in a `SyftrQADataset` class. We will be happy to assist you in getting your dataset ready for use with syftr.