import datasets

from syftr.storage import (
    BrightHF,
    CragTask3HF,
    DRDocsHF,
    FinanceBenchHF,
    HotPotQAHF,
    MultiHopRAGHF,
    PartitionMap,
    PhantomWikiv050,
    SyntheticCragTask3HF,
    SyntheticFinanceBenchHF,
    SyntheticHotPotQAHF,
)


def test_drdocs_hf():
    """DrDocsHF must have two subsets: qapairs, groundingdata
    Each subset must have four partitions: train, test, sample, holdout
    groundingdata, each row must have columns: markdown, filename
    qapairs, each row much has columns: question, answer, id
    """

    # check the dataset properties using HF API
    dataset_name = "DataRobot-Research/drdocs"
    configs = datasets.get_dataset_config_names(dataset_name)
    assert "qapairs" in configs
    assert "groundingdata" in configs
    assert len(configs) == 2

    # qapairs subset
    drdocs_qapairs = datasets.load_dataset(dataset_name, "qapairs")
    assert isinstance(drdocs_qapairs, datasets.DatasetDict)
    split_names = drdocs_qapairs.keys()
    for split_name in split_names:
        assert split_name in ["train", "test", "sample", "holdout"]

    assert drdocs_qapairs["train"].num_rows == 10
    assert drdocs_qapairs["test"].num_rows == 80
    assert drdocs_qapairs["sample"].num_rows == 5
    assert drdocs_qapairs["holdout"].num_rows == 10

    for split_name in split_names:
        features = drdocs_qapairs[split_name].features
        feature_names = list(features.keys())
        assert "question" in feature_names
        assert "answer" in feature_names
        assert "id" in feature_names

    # groundingdata subset
    drdocs_groundingdata = datasets.load_dataset(dataset_name, "groundingdata")
    assert isinstance(drdocs_groundingdata, datasets.DatasetDict)
    split_names = drdocs_groundingdata.keys()
    for split_name in split_names:
        assert split_name == "train"

    assert drdocs_groundingdata["train"].num_rows == 904

    # then check dataset properties using FlowGen API
    for partition in ["train", "test", "sample", "holdout"]:
        drdocs_ds = DRDocsHF(partition_map=PartitionMap(test=partition))

        examples = list(drdocs_ds.iter_examples())
        match partition:
            case "train":
                assert len(examples) == 10
            case "test":
                assert len(examples) == 80
            case "sample":
                assert len(examples) == 5
            case "holdout":
                assert len(examples) == 10

        docs = list(drdocs_ds.iter_grounding_data())
        assert len(docs) == 904


def test_crag_hf():
    def _test_querypairs_subset(dataset: datasets.DatasetDict):
        assert isinstance(dataset, datasets.DatasetDict)
        split_names = dataset.keys()
        for split_name in split_names:
            assert split_name in ["train", "test", "sample", "holdout"]

        for split_name in split_names:
            features = dataset[split_name].features
            feature_names = set(features.keys())
            feature_names_should_be_there = set(
                [
                    "interaction_id",
                    "query_time",
                    "domain",
                    "question_type",
                    "static_or_dynamic",
                    "query",
                    "answer",
                    "search_results",
                    "split",
                    "search_results_hashes",
                    "partition",
                    "__index_level_0__",
                ]
            )
            assert feature_names_should_be_there == feature_names

    def _test_querypairs_synthetic_subset(dataset: datasets.DatasetDict):
        assert isinstance(dataset, datasets.DatasetDict)
        split_names = dataset.keys()
        for split_name in split_names:
            assert split_name in ["train", "test", "sample", "holdout"]

        for split_name in split_names:
            features = dataset[split_name].features
            feature_names = set(features.keys())
            feature_names_should_be_there = set(
                [
                    "id",
                    "query",
                    "reference_answer",
                ]
            )
            assert feature_names_should_be_there == feature_names

    def _test_groundingdata_subset(dataset: datasets.DatasetDict):
        assert isinstance(dataset, datasets.DatasetDict)
        split_names = dataset.keys()
        for split_name in split_names:
            assert split_name in ["train", "test", "sample", "holdout"]

        for split_name in split_names:
            features = dataset[split_name].features
            feature_names = set(features.keys())
            feature_names_should_be_there = set(
                [
                    "filename",
                    "markdown",
                ]
            )
            assert feature_names_should_be_there == feature_names

    # test through HF API
    dataset_name = "DataRobot-Research/crag"
    configs = datasets.get_dataset_config_names(dataset_name)
    assert "qapairs_finance" in configs
    assert "qapairs_movie" in configs
    assert "qapairs_music" in configs
    assert "qapairs_open" in configs
    assert "qapairs_sports" in configs
    assert "qapairs_synthetic_music" in configs
    assert "qapairs_synthetic_movie" in configs
    assert "qapairs_synthetic_open" in configs
    assert "qapairs_synthetic_sports" in configs
    assert "qapairs_synthetic_finance" in configs
    assert "groundingdata_finance" in configs
    assert "groundingdata_movie" in configs
    assert "groundingdata_music" in configs
    assert "groundingdata_open" in configs
    assert "groundingdata_sports" in configs
    assert len(configs) == 15

    # finance qapairs
    crag_qapairs_finance = datasets.load_dataset(dataset_name, "qapairs_finance")
    _test_querypairs_subset(crag_qapairs_finance)
    assert crag_qapairs_finance["train"].num_rows == 46
    assert crag_qapairs_finance["test"].num_rows == 47
    assert crag_qapairs_finance["sample"].num_rows == 1
    assert crag_qapairs_finance["holdout"].num_rows == 10

    # movie qapairs
    crag_qapairs_movie = datasets.load_dataset(dataset_name, "qapairs_movie")
    _test_querypairs_subset(crag_qapairs_movie)
    assert crag_qapairs_movie["train"].num_rows == 54
    assert crag_qapairs_movie["test"].num_rows == 41
    assert crag_qapairs_movie["sample"].num_rows == 7
    assert crag_qapairs_movie["holdout"].num_rows == 11

    # music qapairs
    crag_qapairs_music = datasets.load_dataset(dataset_name, "qapairs_music")
    _test_querypairs_subset(crag_qapairs_music)
    assert crag_qapairs_music["train"].num_rows == 22
    assert crag_qapairs_music["test"].num_rows == 34
    assert crag_qapairs_music["sample"].num_rows == 1
    assert crag_qapairs_music["holdout"].num_rows == 8

    # open qapairs
    crag_qapairs_open = datasets.load_dataset(dataset_name, "qapairs_open")
    _test_querypairs_subset(crag_qapairs_open)
    assert crag_qapairs_open["train"].num_rows == 42
    assert crag_qapairs_open["test"].num_rows == 52
    assert crag_qapairs_open["sample"].num_rows == 5
    assert crag_qapairs_open["holdout"].num_rows == 15

    # sports qapairs
    crag_qapairs_sports = datasets.load_dataset(dataset_name, "qapairs_sports")
    _test_querypairs_subset(crag_qapairs_sports)
    assert crag_qapairs_sports["train"].num_rows == 39
    assert crag_qapairs_sports["test"].num_rows == 46
    assert crag_qapairs_sports["sample"].num_rows == 7
    assert crag_qapairs_sports["holdout"].num_rows == 11

    # finance qapairs synthetic
    qapairs_synthetic_finance = datasets.load_dataset(
        dataset_name, "qapairs_synthetic_finance"
    )
    _test_querypairs_synthetic_subset(qapairs_synthetic_finance)
    assert qapairs_synthetic_finance["train"].num_rows == 171
    assert qapairs_synthetic_finance["test"].num_rows == 173
    assert qapairs_synthetic_finance["sample"].num_rows == 168
    assert qapairs_synthetic_finance["holdout"].num_rows == 10

    # open qapairs synthetic
    qapairs_synthetic_open = datasets.load_dataset(
        dataset_name, "qapairs_synthetic_open"
    )
    _test_querypairs_synthetic_subset(qapairs_synthetic_open)
    assert qapairs_synthetic_open["train"].num_rows == 200
    assert qapairs_synthetic_open["test"].num_rows == 198
    assert qapairs_synthetic_open["sample"].num_rows == 200
    assert qapairs_synthetic_open["holdout"].num_rows == 15

    # music qapairs synthetic
    qapairs_synthetic_music = datasets.load_dataset(
        dataset_name, "qapairs_synthetic_music"
    )
    _test_querypairs_synthetic_subset(qapairs_synthetic_music)
    assert qapairs_synthetic_music["train"].num_rows == 200
    assert qapairs_synthetic_music["test"].num_rows == 177
    assert qapairs_synthetic_music["sample"].num_rows == 146
    assert qapairs_synthetic_music["holdout"].num_rows == 8

    # movie qapairs synthetic
    qapairs_synthetic_movie = datasets.load_dataset(
        dataset_name, "qapairs_synthetic_movie"
    )
    _test_querypairs_synthetic_subset(qapairs_synthetic_movie)
    assert qapairs_synthetic_movie["train"].num_rows == 183
    assert qapairs_synthetic_movie["test"].num_rows == 197
    assert qapairs_synthetic_movie["sample"].num_rows == 165
    assert qapairs_synthetic_movie["holdout"].num_rows == 11

    # sports qapairs synthetic
    qapairs_synthetic_sports = datasets.load_dataset(
        dataset_name, "qapairs_synthetic_sports"
    )
    _test_querypairs_synthetic_subset(qapairs_synthetic_sports)
    assert qapairs_synthetic_sports["train"].num_rows == 200
    assert qapairs_synthetic_sports["test"].num_rows == 200
    assert qapairs_synthetic_sports["sample"].num_rows == 200
    assert qapairs_synthetic_sports["holdout"].num_rows == 11

    # finance groundingdata
    crag_groundingdata_finance = datasets.load_dataset(
        dataset_name, "groundingdata_finance"
    )
    _test_groundingdata_subset(crag_groundingdata_finance)
    assert crag_groundingdata_finance["train"].num_rows == 2191
    assert crag_groundingdata_finance["test"].num_rows == 2302
    assert crag_groundingdata_finance["sample"].num_rows == 50
    assert crag_groundingdata_finance["holdout"].num_rows == 498

    # movie groundingdata
    crag_groundingdata_movie = datasets.load_dataset(
        dataset_name, "groundingdata_movie"
    )
    _test_groundingdata_subset(crag_groundingdata_movie)
    assert crag_groundingdata_movie["train"].num_rows == 2654
    assert crag_groundingdata_movie["test"].num_rows == 1993
    assert crag_groundingdata_movie["sample"].num_rows == 348
    assert crag_groundingdata_movie["holdout"].num_rows == 535

    # music groundingdata
    crag_groundingdata_music = datasets.load_dataset(
        dataset_name, "groundingdata_music"
    )
    _test_groundingdata_subset(crag_groundingdata_music)
    assert crag_groundingdata_music["train"].num_rows == 1090
    assert crag_groundingdata_music["test"].num_rows == 1645
    assert crag_groundingdata_music["sample"].num_rows == 50
    assert crag_groundingdata_music["holdout"].num_rows == 382

    # open groundingdata
    crag_groundingdata_open = datasets.load_dataset(dataset_name, "groundingdata_open")
    _test_groundingdata_subset(crag_groundingdata_open)
    assert crag_groundingdata_open["train"].num_rows == 2093
    assert crag_groundingdata_open["test"].num_rows == 2570
    assert crag_groundingdata_open["sample"].num_rows == 250
    assert crag_groundingdata_open["holdout"].num_rows == 750

    # sports groundingdata
    crag_groundingdata_sports = datasets.load_dataset(
        dataset_name, "groundingdata_sports"
    )
    _test_groundingdata_subset(crag_groundingdata_sports)
    assert crag_groundingdata_sports["train"].num_rows == 1946
    assert crag_groundingdata_sports["test"].num_rows == 2279
    assert crag_groundingdata_sports["sample"].num_rows == 346
    assert crag_groundingdata_sports["holdout"].num_rows == 536

    # test through FlowGen API
    for subset in ["finance", "movie", "music", "open", "sports"]:
        for partition in ["train", "test", "sample", "holdout"]:
            crag_ds = CragTask3HF(
                partition_map=PartitionMap(test=partition), subset=subset
            )

            examples = list(crag_ds.iter_examples())
            match subset:
                case "finance":
                    match partition:
                        case "train":
                            assert len(examples) == 46
                        case "test":
                            assert len(examples) == 47
                        case "sample":
                            assert len(examples) == 1
                        case "holdout":
                            assert len(examples) == 10
                case "movie":
                    match partition:
                        case "train":
                            assert len(examples) == 54
                        case "test":
                            assert len(examples) == 41
                        case "sample":
                            assert len(examples) == 7
                        case "holdout":
                            assert len(examples) == 11
                case "music":
                    match partition:
                        case "train":
                            assert len(examples) == 22
                        case "test":
                            assert len(examples) == 34
                        case "sample":
                            assert len(examples) == 1
                        case "holdout":
                            assert len(examples) == 8
                case "open":
                    match partition:
                        case "train":
                            assert len(examples) == 42
                        case "test":
                            assert len(examples) == 52
                        case "sample":
                            assert len(examples) == 5
                        case "holdout":
                            assert len(examples) == 15
                case "sports":
                    match partition:
                        case "train":
                            assert len(examples) == 39
                        case "test":
                            assert len(examples) == 46
                        case "sample":
                            assert len(examples) == 7
                        case "holdout":
                            assert len(examples) == 11

            docs = list(crag_ds.iter_grounding_data())
            match subset:
                case "finance":
                    match partition:
                        case "train":
                            assert len(docs) == 2191
                        case "test":
                            assert len(docs) == 2302
                        case "sample":
                            assert len(docs) == 50
                        case "holdout":
                            assert len(docs) == 498
                case "movie":
                    match partition:
                        case "train":
                            assert len(docs) == 2654
                        case "test":
                            assert len(docs) == 1993
                        case "sample":
                            assert len(docs) == 348
                        case "holdout":
                            assert len(docs) == 535
                case "music":
                    match partition:
                        case "train":
                            assert len(docs) == 1090
                        case "test":
                            assert len(docs) == 1645
                        case "sample":
                            assert len(docs) == 50
                        case "holdout":
                            assert len(docs) == 382
                case "open":
                    match partition:
                        case "train":
                            assert len(docs) == 2093
                        case "test":
                            assert len(docs) == 2570
                        case "sample":
                            assert len(docs) == 250
                        case "holdout":
                            assert len(docs) == 750
                case "sports":
                    match partition:
                        case "train":
                            assert len(docs) == 1946
                        case "test":
                            assert len(docs) == 2279
                        case "sample":
                            assert len(docs) == 346
                        case "holdout":
                            assert len(docs) == 536

    # test through FlowGen API for synthetic datasets
    # TODO: defend number of examples
    for subset in ["finance", "movie", "music", "open", "sports"]:
        for partition in ["train", "test", "sample", "holdout"]:
            crag_ds = SyntheticCragTask3HF(
                partition_map=PartitionMap(test=partition), subset=subset
            )

            examples = list(crag_ds.iter_examples())
            assert len(examples) > 0

            docs = list(crag_ds.iter_grounding_data())
            assert len(docs) > 0


def test_financebench_hf():
    # test through HF API
    dataset_name = "DataRobot-Research/financebench"
    configs = datasets.get_dataset_config_names(dataset_name)
    assert "qapairs" in configs
    assert "groundingdata" in configs
    assert "qapairs_synthetic" in configs
    assert len(configs) == 3

    # qapairs subset
    financebench_qapairs = datasets.load_dataset(dataset_name, "qapairs")
    assert isinstance(financebench_qapairs, datasets.DatasetDict)
    split_names = financebench_qapairs.keys()
    for split_name in split_names:
        assert split_name in ["train", "test", "sample", "holdout"]

    for split_name in split_names:
        features = financebench_qapairs[split_name].features
        feature_names = set(features.keys())
        feature_names_should_be_there = set(
            [
                "financebench_id",
                "company",
                "doc_name",
                "question_type",
                "question_reasoning",
                "domain_question_num",
                "question",
                "answer",
                "justification",
                "dataset_subset_label",
                "evidence",
                "html_evidence",
                "_flowgen_partition",
            ]
        )
        assert feature_names_should_be_there == feature_names

    assert financebench_qapairs["train"].num_rows == 53
    assert financebench_qapairs["test"].num_rows == 49
    assert financebench_qapairs["sample"].num_rows == 11
    assert financebench_qapairs["holdout"].num_rows == 48

    # qapairs synthetic subset
    financebench_qapairs_synthetic = datasets.load_dataset(
        dataset_name, "qapairs_synthetic"
    )
    assert isinstance(financebench_qapairs_synthetic, datasets.DatasetDict)
    split_names = financebench_qapairs_synthetic.keys()
    for split_name in split_names:
        assert split_name in ["train", "test", "sample", "holdout"]

    for split_name in split_names:
        features = financebench_qapairs_synthetic[split_name].features
        feature_names = set(features.keys())
        feature_names_should_be_there = set(
            [
                "id",
                "query",
                "reference_answer",
            ]
        )
        assert feature_names_should_be_there == feature_names

    assert financebench_qapairs_synthetic["train"].num_rows == 200
    assert financebench_qapairs_synthetic["test"].num_rows == 200
    assert financebench_qapairs_synthetic["sample"].num_rows == 200
    assert financebench_qapairs_synthetic["holdout"].num_rows == 48

    # groundingdata subset
    financebench_groundingdata = datasets.load_dataset(dataset_name, "groundingdata")
    assert isinstance(financebench_groundingdata, datasets.DatasetDict)
    split_names = financebench_groundingdata.keys()
    for split_name in split_names:
        assert split_name in ["train", "test", "sample", "holdout"]

    for split_name in split_names:
        features = financebench_groundingdata[split_name].features
        feature_names = set(features.keys())
        feature_names_should_be_there = set(
            [
                "filename",
                "html",
            ]
        )
        assert feature_names_should_be_there == feature_names

    assert financebench_groundingdata["train"].num_rows == 63
    assert financebench_groundingdata["test"].num_rows == 72
    assert financebench_groundingdata["sample"].num_rows == 13
    assert financebench_groundingdata["holdout"].num_rows == 159

    # test through FlowGen API
    for partition in ["train", "test", "sample", "holdout"]:
        fb_hf_ds = FinanceBenchHF(partition_map=PartitionMap(test=partition))

        examples = list(fb_hf_ds.iter_examples())
        match partition:
            case "train":
                assert len(examples) == 53
            case "test":
                assert len(examples) == 49
            case "sample":
                assert len(examples) == 11
            case "holdout":
                assert len(examples) == 48

        docs = list(fb_hf_ds.iter_grounding_data())
        match partition:
            case "train":
                assert len(docs) == 63
            case "test":
                assert len(docs) == 72
            case "sample":
                assert len(docs) == 13
            case "holdout":
                assert len(docs) == 159

    for partition in ["train", "test", "sample", "holdout"]:
        fb_hf_ds = SyntheticFinanceBenchHF(partition_map=PartitionMap(test=partition))

        examples = list(fb_hf_ds.iter_examples())
        match partition:
            case "train":
                assert len(examples) == 200
            case "test":
                assert len(examples) == 200
            case "sample":
                assert len(examples) == 200
            case "holdout":
                assert len(examples) == 48

        docs = list(fb_hf_ds.iter_grounding_data())
        match partition:
            case "train":
                assert len(docs) == 63
            case "test":
                assert len(docs) == 72
            case "sample":
                assert len(docs) == 13
            case "holdout":
                assert len(docs) == 159


def test_hotpotqa_hf():
    # test through HF API
    dataset_name = "DataRobot-Research/hotpotqa"
    configs = datasets.get_dataset_config_names(dataset_name)
    assert "qapairs_dev" in configs
    assert "qapairs_train_hard" in configs
    assert "qapairs_synthetic_dev" in configs
    assert "qapairs_synthetic_train_hard" in configs
    assert "groundingdata_dev" in configs
    assert "groundingdata_train_hard" in configs
    assert len(configs) == 6

    # qapairs_dev subset
    hotpotqa_qapairs_dev = datasets.load_dataset(dataset_name, "qapairs_dev")
    assert isinstance(hotpotqa_qapairs_dev, datasets.DatasetDict)
    split_names = hotpotqa_qapairs_dev.keys()
    for split_name in split_names:
        assert split_name in ["train", "test", "sample", "holdout"]

    for split_name in split_names:
        features = hotpotqa_qapairs_dev[split_name].features
        feature_names = set(features.keys())
        feature_names_should_be_there = set(
            ["question", "answer", "id", "type", "level", "supporting_facts", "context"]
        )
    assert feature_names_should_be_there == feature_names

    assert hotpotqa_qapairs_dev["train"].num_rows == 6880
    assert hotpotqa_qapairs_dev["test"].num_rows == 100
    assert hotpotqa_qapairs_dev["sample"].num_rows == 20
    assert hotpotqa_qapairs_dev["holdout"].num_rows == 405

    # qapairs_synthetic_dev subset
    hotpotqa_qapairs_synthetic_dev = datasets.load_dataset(
        dataset_name, "qapairs_synthetic_dev"
    )
    assert isinstance(hotpotqa_qapairs_synthetic_dev, datasets.DatasetDict)
    split_names = hotpotqa_qapairs_synthetic_dev.keys()
    for split_name in split_names:
        assert split_name in ["train", "test", "sample", "holdout"]

    for split_name in split_names:
        features = hotpotqa_qapairs_synthetic_dev[split_name].features
        feature_names = set(features.keys())
        feature_names_should_be_there = set(["id", "query", "reference_answer"])
    assert feature_names_should_be_there == feature_names

    assert hotpotqa_qapairs_synthetic_dev["train"].num_rows == 200
    assert hotpotqa_qapairs_synthetic_dev["test"].num_rows == 200
    assert hotpotqa_qapairs_synthetic_dev["sample"].num_rows == 194
    assert hotpotqa_qapairs_synthetic_dev["holdout"].num_rows == 7836

    # qapairs_synthetic_train_hard subset
    hotpotqa_qapairs_synthetic_train_hard = datasets.load_dataset(
        dataset_name, "qapairs_synthetic_train_hard"
    )
    assert isinstance(hotpotqa_qapairs_synthetic_train_hard, datasets.DatasetDict)
    split_names = hotpotqa_qapairs_synthetic_train_hard.keys()
    for split_name in split_names:
        assert split_name in ["train", "test", "sample", "holdout"]

    for split_name in split_names:
        features = hotpotqa_qapairs_synthetic_train_hard[split_name].features
        feature_names = set(features.keys())
        feature_names_should_be_there = set(["id", "query", "reference_answer"])
    assert feature_names_should_be_there == feature_names

    assert hotpotqa_qapairs_synthetic_train_hard["train"].num_rows == 200
    assert hotpotqa_qapairs_synthetic_train_hard["test"].num_rows == 200
    assert hotpotqa_qapairs_synthetic_train_hard["sample"].num_rows == 196
    assert hotpotqa_qapairs_synthetic_train_hard["holdout"].num_rows == 7836

    # qapairs_train_hard subset
    hotpotqa_qapairs_train_hard = datasets.load_dataset(
        dataset_name, "qapairs_train_hard"
    )
    assert isinstance(hotpotqa_qapairs_train_hard, datasets.DatasetDict)
    split_names = hotpotqa_qapairs_train_hard.keys()
    for split_name in split_names:
        assert split_name in ["train", "test", "sample", "holdout"]

    for split_name in split_names:
        features = hotpotqa_qapairs_train_hard[split_name].features
        feature_names = set(features.keys())
        feature_names_should_be_there = set(
            ["question", "answer", "id", "type", "level", "supporting_facts", "context"]
        )
    assert feature_names_should_be_there == feature_names

    assert hotpotqa_qapairs_train_hard["train"].num_rows == 7305
    assert hotpotqa_qapairs_train_hard["test"].num_rows == 500
    assert hotpotqa_qapairs_train_hard["sample"].num_rows == 20
    assert hotpotqa_qapairs_train_hard["holdout"].num_rows == 7836

    # groundingdata_dev subset
    hotpotqa_groundingdata_dev = datasets.load_dataset(
        dataset_name, "groundingdata_dev"
    )
    assert isinstance(hotpotqa_groundingdata_dev, datasets.DatasetDict)
    split_names = hotpotqa_groundingdata_dev.keys()
    for split_name in split_names:
        assert split_name in ["train", "test", "sample", "holdout"]

    for split_name in split_names:
        features = hotpotqa_groundingdata_dev[split_name].features
        feature_names = set(features.keys())
        feature_names_should_be_there = set(["text"])
        assert feature_names_should_be_there == feature_names

    assert hotpotqa_groundingdata_dev["train"].num_rows == 68414
    assert hotpotqa_groundingdata_dev["test"].num_rows == 990
    assert hotpotqa_groundingdata_dev["sample"].num_rows == 200
    assert hotpotqa_groundingdata_dev["holdout"].num_rows == 4038

    # groundingdata_train_hard subset
    hotpotqa_groundingdata_train_hard = datasets.load_dataset(
        dataset_name, "groundingdata_train_hard"
    )
    assert isinstance(hotpotqa_groundingdata_train_hard, datasets.DatasetDict)
    split_names = hotpotqa_groundingdata_train_hard.keys()
    for split_name in split_names:
        assert split_name in ["train", "test", "sample", "holdout"]

    for split_name in split_names:
        features = hotpotqa_groundingdata_train_hard[split_name].features
        feature_names = set(features.keys())
        feature_names_should_be_there = set(["text"])
        assert feature_names_should_be_there == feature_names

    assert hotpotqa_groundingdata_train_hard["train"].num_rows == 72714
    assert hotpotqa_groundingdata_train_hard["test"].num_rows == 4972
    assert hotpotqa_groundingdata_train_hard["sample"].num_rows == 200
    assert hotpotqa_groundingdata_train_hard["holdout"].num_rows == 78102

    # test through FlowGen API
    partition = "test"
    subset = "train_hard"

    for subset in ["dev", "train_hard"]:
        for partition in ["train", "test", "sample", "holdout"]:
            hotpotqa_ds = HotPotQAHF(
                partition_map=PartitionMap(test=partition), subset=subset
            )

            examples = list(hotpotqa_ds.iter_examples())
            if subset == "dev" and partition == "train":
                assert len(examples) == 6880
            elif subset == "dev" and partition == "test":
                assert len(examples) == 100
            elif subset == "dev" and partition == "sample":
                assert len(examples) == 20
            elif subset == "dev" and partition == "holdout":
                assert len(examples) == 405
            elif subset == "train_hard" and partition == "train":
                assert len(examples) == 7305
            elif subset == "train_hard" and partition == "test":
                assert len(examples) == 500
            elif subset == "train_hard" and partition == "sample":
                assert len(examples) == 20
            elif subset == "train_hard" and partition == "holdout":
                assert len(examples) == 7836

            docs = list(hotpotqa_ds.iter_grounding_data())
            if subset == "dev" and partition == "train":
                assert len(docs) == 68414
            elif subset == "dev" and partition == "test":
                assert len(docs) == 990
            elif subset == "dev" and partition == "sample":
                assert len(docs) == 200
            elif subset == "dev" and partition == "holdout":
                assert len(docs) == 4038
            elif subset == "train_hard" and partition == "train":
                assert len(docs) == 72714
            elif subset == "train_hard" and partition == "test":
                assert len(docs) == 4972
            elif subset == "train_hard" and partition == "sample":
                assert len(docs) == 200
            elif subset == "train_hard" and partition == "holdout":
                assert len(docs) == 78102

    for subset in ["dev", "train_hard"]:
        for partition in ["train", "test", "sample", "holdout"]:
            synthetic_hotpotqa_ds = SyntheticHotPotQAHF(
                partition_map=PartitionMap(test=partition), subset=subset
            )

            examples = list(synthetic_hotpotqa_ds.iter_examples())
            if subset == "dev" and partition == "train":
                assert len(examples) == 200
            elif subset == "dev" and partition == "test":
                assert len(examples) == 200
            elif subset == "dev" and partition == "sample":
                assert len(examples) == 194
            elif subset == "dev" and partition == "holdout":
                assert len(examples) == 7836
            elif subset == "train_hard" and partition == "train":
                assert len(examples) == 200
            elif subset == "train_hard" and partition == "test":
                assert len(examples) == 200
            elif subset == "train_hard" and partition == "sample":
                assert len(examples) == 196
            elif subset == "train_hard" and partition == "holdout":
                assert len(examples) == 7836

            docs = list(synthetic_hotpotqa_ds.iter_grounding_data())
            if subset == "dev" and partition == "train":
                assert len(docs) == 68414
            elif subset == "dev" and partition == "test":
                assert len(docs) == 990
            elif subset == "dev" and partition == "sample":
                assert len(docs) == 200
            elif subset == "dev" and partition == "holdout":
                assert len(docs) == 4038
            elif subset == "train_hard" and partition == "train":
                assert len(docs) == 72714
            elif subset == "train_hard" and partition == "test":
                assert len(docs) == 4972
            elif subset == "train_hard" and partition == "sample":
                assert len(docs) == 200
            elif subset == "train_hard" and partition == "holdout":
                assert len(docs) == 78102


def test_phantomwikiv050_hf():
    for subset in [
        "depth_20_size_25_seed_1",
        "depth_20_size_25_seed_2",
        "depth_20_size_25_seed_3",
        "depth_20_size_50_seed_1",
        "depth_20_size_50_seed_2",
        "depth_20_size_50_seed_3",
        "depth_20_size_100_seed_1",
        "depth_20_size_100_seed_2",
        "depth_20_size_100_seed_3",
        "depth_20_size_200_seed_1",
        "depth_20_size_200_seed_2",
        "depth_20_size_200_seed_3",
        "depth_20_size_300_seed_1",
        "depth_20_size_300_seed_2",
        "depth_20_size_300_seed_3",
        "depth_20_size_400_seed_1",
        "depth_20_size_400_seed_2",
        "depth_20_size_400_seed_3",
        "depth_20_size_500_seed_1",
        "depth_20_size_500_seed_2",
        "depth_20_size_500_seed_3",
        "depth_20_size_1000_seed_1",
        "depth_20_size_1000_seed_2",
        "depth_20_size_1000_seed_3",
        "depth_20_size_2500_seed_1",
        "depth_20_size_2500_seed_2",
        "depth_20_size_2500_seed_3",
        "depth_20_size_5000_seed_1",
        "depth_20_size_5000_seed_2",
        "depth_20_size_5000_seed_3",
        "depth_20_size_10000_seed_1",
        "depth_20_size_10000_seed_2",
        "depth_20_size_10000_seed_3",
    ]:
        for partition in ["train", "test", "sample", "holdout"]:
            phantomwiki_ds = PhantomWikiv050(
                partition_map=PartitionMap(test=partition), subset=subset
            )

            examples = list(phantomwiki_ds.iter_examples())
            assert len(examples) > 0
            docs = list(phantomwiki_ds.iter_grounding_data())
            assert len(docs) > 0


def test_multihoprag_hf():
    for partition in ["train", "test", "sample", "holdout"]:
        multihoprag_ds = MultiHopRAGHF(partition_map=PartitionMap(test=partition))

        examples = list(multihoprag_ds.iter_examples())
        assert len(examples) > 0

        docs = list(multihoprag_ds.iter_grounding_data())
        assert len(docs) > 0


def test_bright_hf():
    subsets = [
        "earth_science",
        "biology",
        "economics",
        "psychology",
        "robotics",
        "stackoverflow",
        "sustainable_living",
        "pony",
        "leetcode",
        "aops",
        "theoremqa_theorems",
        "theoremqa_questions",
    ]

    for subset in subsets:
        print(f"Testing subset: {subset}")
        for partition in ["train", "test", "sample", "holdout"]:
            bright_ds = BrightHF(
                partition_map=PartitionMap(test=partition), subset=subset
            )

            examples = list(bright_ds.iter_examples())
            assert len(examples) > 0

            docs = list(bright_ds.iter_grounding_data())
            assert len(docs) > 0
