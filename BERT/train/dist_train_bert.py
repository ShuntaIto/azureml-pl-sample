import os
import pytorch_lightning as pl

from transformers import BertJapaneseTokenizer

from argparse import ArgumentParser
from azureml_env_adapter import set_environment_variables, set_environment_variables_for_nccl_backend
from model import BERTClassificationModel
from livedoor_data import LivedoorNewsDataModule
from azureml.core import Run

def cli_main():
    pl.seed_everything(42, workers=True)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--max_epochs", default=20, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--strategy", default='ddp', type=str)
    parser.add_argument("--gpus", default=1, type=int)

    #parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    #set_environment_variables()
    set_environment_variables_for_nccl_backend(single_node=int(args.num_nodes) > 1)

    # ------------
    # data
    # ------------

    tokenizer = BertJapaneseTokenizer.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking')
    data_module = LivedoorNewsDataModule(tokenizer, args.batch_size)

    # ------------
    # model
    # ------------
    model = BERTClassificationModel(output_lr=args.learning_rate)

    # fix BERT model
    for param in model.bert.parameters():
        param.requires_grad = False
    
    for param in model.bert.encoder.layer[-1].parameters():
        param.requires_grad = True


    # ------------
    # training
    # ------------

    run = Run.get_context()
    ws = run.experiment.workspace
    mlflow_url = ws.get_mlflow_tracking_uri()

    mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()
    mlf_logger = pl.loggers.MLFlowLogger(
        experiment_name=run.experiment.name,
        tracking_uri=mlflow_url)
    mlf_logger._run_id = run.id

    #trainer = pl.Trainer.from_argparse_args(args)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        accelerator='gpu',
        logger=mlf_logger
    )
    trainer.fit(model, datamodule=data_module)

    # ------------
    # testing
    # ------------
    result = trainer.test(model, datamodule=data_module)
    print(result)

    # ------------
    # model saving
    # ------------
    model_path = os.path.join('outputs', "model.ckpt")
    trainer.save_checkpoint(model_path)
    
    #model_name = "model" + str(os.environ["OMPI_COMM_WORLD_RANK"]) + ".onnx"
    #model_path = os.path.join('outputs', model_name)
#
    #dummy_input_ids = torch.randint(low=1, high=10000, size=(10, 512), device='cuda')
    #dummy_attention_mask = torch.ones(size=(10, 512), dtype=torch.long, device='cuda')
    #dummy_token_type_ids = torch.zeros(size=(10, 512), dtype=torch.long, device='cuda')
    #
    #dummy_input = (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)
    #dummy_output = model(dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)
#
    #torch.onnx.export(
    #    model=model,
    #    args=dummy_input,
    #    example_outputs=(dummy_output),
    #    f=model_path,
    #    opset_version=12,
    #    verbose=True,
    #    input_names=["input_ids","attention_mask","token_type_ids"],
    #    output_names=["output"],
    #    dynamic_axes={
    #        'input_ids': [0, 1], 
    #        'attention_mask': [0, 1], 
    #        'token_type_ids': [0, 1], 
    #        'output': [0]
    #    }
    #)


if __name__ == "__main__":
    cli_main()
