```python
# To run eval.
python tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c_split --config configs/vqa/m4c_textvqa/tap_refine.yml --save_dir save/m4c_base_val --run_type val --resume_file save/finetuned/textvqa_tap_base_best.ckpt --evalai_inference true --verbose_dump true
```

```python
# Modifications for pythia/trainers/base_trainer.py
# to save all metadata info about the dataset.
def predict_for_evalai(self, dataset_type):
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        with torch.no_grad():
            self.model.eval()
            message = "Starting {} inference for evalai".format(dataset_type)
            self.writer.write(message)
            count = 0
            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()

                for batch in tqdm(dataloader):
                    count += 1
                    prepared_batch = reporter.prepare_batch(batch)
                    # model_output = self.model(prepared_batch)
                    # report = Report(prepared_batch, model_output)
                    # reporter.add_to_report(report)

                    temp = OrderedDict()
                    for k, v in prepared_batch.items():
                        if isinstance(v, torch.Tensor):
                            temp[k] = v.cpu().numpy()
                        else:
                            temp[k] = v
                    with open(os.path.join(self.config.training_parameters.save_dir, 'meta_%s_%d.pkl'%(dataset_type, count)), 'wb') as fp:
                        pickle.dump(temp, fp)

            self.writer.write("Finished predicting")
            self.writer.write("Final count: %d"%count)
            
            # self.writer.write("Saving metadata to disk at %s" %\
            #      self.config.training_parameters.save_dir)
            self.writer.write("Saved metadata to disk.")

            self.model.train()
```

# TAP: Text-Aware Pre-training
[TAP: Text-Aware Pre-training for Text-VQA and Text-Caption](https://arxiv.org/pdf/2012.04638.pdf)

by [Zhengyuan Yang](https://zyang-ur.github.io/), [Yijuan Lu](https://scholar.google.com/citations?user=cpkrT44AAAAJ&hl=en), [Jianfeng Wang](https://scholar.google.com/citations?user=vJWEw_8AAAAJ&hl=en), [Xi Yin](https://xiyinmsu.github.io/), [Dinei Florencio](https://www.microsoft.com/en-us/research/people/dinei/), [Lijuan Wang](https://www.microsoft.com/en-us/research/people/lijuanw/), [Cha Zhang](https://www.microsoft.com/en-us/research/people/chazhang/), [Lei Zhang](https://www.microsoft.com/en-us/research/people/leizhang/), and [Jiebo Luo](http://cs.rochester.edu/u/jluo)

IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021, Oral


### Introduction
We propose Text-Aware Pre-training (TAP) for Text-VQA and Text-Caption tasks.
For more details, please refer to our
[paper](https://arxiv.org/pdf/2012.04638.pdf).

<!-- Note: This codebase is still in beta release. We are continue organizing the repo and completing the doumentations. Meanwhile, please feel free to contact me regarding any issue and request for clarification. -->

<p align="center">
  <img src="http://www.cs.rochester.edu/u/zyang39/TAP/intro.jpg" width="75%"/>
</p>

### Citation

    @inproceedings{yang2021tap,
      title={TAP: Text-Aware Pre-training for Text-VQA and Text-Caption},
      author={Yang, Zhengyuan and Lu, Yijuan and Wang, Jianfeng and Yin, Xi and Florencio, Dinei and Wang, Lijuan and Zhang, Cha and Zhang, Lei and Luo, Jiebo},
      booktitle={CVPR},
      year={2021}
    }

### Prerequisites
* A Linux distro (*this has not been tested on any other OS*)
* Python 3.6
* Conda (for package, env management) <https://www.anaconda.com/products/individual>

## Installation

1. Clone the repository
    ```
    git clone https://github.com/microsoft/TAP.git
    cd TAP
    python setup.py develop
    ```

2. Data

* Please refer to the Readme in the ``data`` folder.
<!-- * Please refer to [AzCopy executable tools](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy) for downloading data/models. -->

### Training
3. Train the model, run the code under main folder. 
Using flag ``--pretrain`` to access the pre-training mode, otherwise the main QA/Captioning losses are used to optimize the model. Example yml files are in ``configs`` folder. Detailed configs are in [released models](https://github.com/microsoft/TAP/tree/main/data).

    Pre-training:
    ```
    python -m torch.distributed.launch --nproc_per_node $num_gpu tools/run.py --pretrain --tasks vqa --datasets $dataset --model $model --seed $seed --config configs/vqa/$dataset/"$pretrain_yml".yml --save_dir save/$pretrain_savedir training_parameters.distributed True

    # for example
    python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --pretrain --tasks vqa --datasets m4c_textvqa --model m4c_split --seed 13 --config configs/vqa/m4c_textvqa/tap_base_pretrain.yml --save_dir save/m4c_split_pretrain_test training_parameters.distributed True
    ```

    Fine-tuning:
    ```
    python -m torch.distributed.launch --nproc_per_node $num_gpu tools/run.py --tasks vqa --datasets $dataset --model $model --seed $seed --config configs/vqa/$dataset/"$refine_yml".yml --save_dir save/$refine_savedir --resume_file save/$pretrain_savedir/$savename/best.ckpt training_parameters.distributed True

    # for example
    python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c_split --seed 13 --config configs/vqa/m4c_textvqa/tap_refine.yml --save_dir save/m4c_split_refine_test --resume_file save/pretrained/textvqa_tap_base_pretrain.ckpt training_parameters.distributed True
    ```

4. Evaluate the model, run the code under main folder. 
Set up val or test set by ``--run_type``.

    ```
    python -m torch.distributed.launch --nproc_per_node $num_gpu tools/run.py --tasks vqa --datasets $dataset --model $model --config configs/vqa/$dataset/"$refine_yml".yml --save_dir save/$refine_savedir --run_type val --resume_file save/$refine_savedir/$savename/best.ckpt training_parameters.distributed True

    # for example
    python -m torch.distributed.launch --nproc_per_node 4 tools/run.py --tasks vqa --datasets m4c_textvqa --model m4c_split --config configs/vqa/m4c_textvqa/tap_refine.yml --save_dir save/m4c_split_refine_test --run_type val --resume_file save/finetuned/textvqa_tap_base_best.ckpt training_parameters.distributed True
    ```

5. Captioning evaluation.
    ```
    python projects/M4C_Captioner/scripts/textcaps_eval.py --set val --pred_file YOUR_VAL_PREDICTION_FILE
    ```

## Performance and Pre-trained Models
Please check the detailed experiment settings in our [paper](https://arxiv.org/pdf/2012.04638.pdf). 

[Model checkpoints (~17G)](https://tapvqacaption.blob.core.windows.net/data/save). 

```
path/to/azcopy copy https://tapvqacaption.blob.core.windows.net/data/save <local_path>/save --recursive
```

Please refer to the Readme in the ``data`` folder for the detailed instructions on azcopy downloading.
<table>
    <thead>
        <tr>
            <th>Text-VQA</th>
            <th>TAP</th>
            <th>TAP** (with extra data)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>TextVQA</td>
            <td>49.91</td>
            <td>54.71</td>
        </tr>
        <tr>
            <td>STVQA</td>
            <td>45.29</td>
            <td>50.83 </td>
        </tr>
    </tbody>
</table>

<table>
    <thead>
        <tr>
            <th>Text-Captioning</th>
            <th>TAP</th>
            <th>TAP** (with extra data)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>TextCaps</td>
            <td>105.05</td>
            <td>109.16</td>
        </tr>
    </tbody>
</table>

### Credits
The project is built based on the following repository:
* [MMF: A multimodal framework for vision and language research](https://github.com/facebookresearch/mmf/).