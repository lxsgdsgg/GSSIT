CUDA_VISIBLE_DEVICES=0 python3 run_classifier.py \
	--data_dir=data/new/ \
	--task_name=opendomain \
	--vocab_file=./pretrained_bert/vocab.txt \
	--bert_config_file=./pretrained_bert/bert_config.json \
	--output_dir=/nas/wangbei/bert_model/new/ \
	--do_train=true \
	--do_eval=true \
	--init_checkpoint=./pretrained_bert/model.ckpt-1000000 \
	--max_seq_length=200 \
	--train_batch_size=32 \
	--learning_rate=5e-5 \
	--num_train_epoch=5