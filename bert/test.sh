CUDA_VISIBLE_DEVICES=3 python3 run_classifier.py \
	--data_dir=data/new/ \
	--task_name=intensionmining \
	--vocab_file=./pretrained_bert/vocab.txt \
	--bert_config_file=./pretrained_bert/bert_config.json \
	--output_dir=/nas/wangbei/bert_model/new/ \
	--do_predict=true \
	--max_seq_length=200
