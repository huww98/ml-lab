{
    epoch: 50,
    log_interval: 10,
    dataset: {
        data_path: 'exp7/data/cmn-eng/cmn.txt',
        source_lang: 1,
        max_len: 16,
        langs: [
            {
                name: 'English',
                split: 'word',
            },
            {
                name: '中文',
                split: 'char',
            },
        ],
    },
    dataloader: {
        train_weight: 7,
        eval_weight: 3,
        split_seed: 42,
        train: {
            batch_size: 256,
        },
        eval: {
            batch_size: 512,
        },
    },
    model: {
        hidden_size:: 512,
        encoder: {
            hidden_size: $.model.hidden_size,
        },
        decoder: {
            hidden_size: $.model.hidden_size,
            dropout: 0.1,
            max_source_len: $.dataset.max_len,
            max_decode_len: 32,
        }
    },
}
