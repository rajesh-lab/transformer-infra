# data utils

Contains common utils for data related stuff.

`prepare_sharded_tokenized_dataset.py`: tokenize dataset into small (~100M) shards, refer to `dataloader.py` to use these shards for training.

`dataloader.py`: contains a pytorch dataloader to consume these shards during training.