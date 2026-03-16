数据来源：[NLB_maze](https://neurallatents.github.io/datasets.html)

数据格式：[Notebook](https://github.com/neurallatents/neurallatents.github.io/blob/master/notebooks/mc_maze.ipynb)

stardard数据下载:

```
dandi download DANDI:000128/0.220113.0400
```

---

长期数据的下载网页：[Using adversarial networks to extend brain computer interface decoding accuracy over time](https://zenodo.org/records/8271239)

示例代码使用的是 `Chewie_CO_2016.7z` 数据集. 

在根目录下, 执行 

```
python -m train.main --train --backbone gru
```

即通过 `train.main` 模块, 以 `gru` 作为 backbone 进行训练.

---

```python
── 000128
│   ├── dandiset.yaml
│   └── sub-Jenkins
├── README.md
├── __pycache__
│   ├── Config.cpython-310.pyc
│   ├── Config.cpython-39.pyc
│   ├── Cross_Attention.cpython-39.pyc
│   ├── Dataloader.cpython-310.pyc
│   ├── Dataloader.cpython-39.pyc
│   ├── GRU.cpython-39.pyc
│   ├── Model.cpython-39.pyc
│   ├── Output_Decoder.cpython-39.pyc
│   ├── RoPE.cpython-39.pyc
│   ├── S4D.cpython-39.pyc
│   ├── engine.cpython-39.pyc
│   ├── evaluate.cpython-39.pyc
│   ├── metrics.cpython-39.pyc
│   ├── plotting.cpython-39.pyc
│   ├── train.cpython-39.pyc
│   └── utils.cpython-39.pyc
├── attachment
│   └── heatmap.png
├── checkpoints
│   ├── possm_gru_seed42.pt
│   └── possm_s4d_seed42.pt
├── data
├── dataset_mc_maze.ipynb
├── eval
│   └── gru_seed42
├── graphs
│   ├── loss_comparison.png
│   └── s4d_loss.png
├── log
│   ├── events.out.tfevents.1767787340.Dominant.3886.0
│   ├── events.out.tfevents.1767787903.Dominant.5215.0
│   ├── events.out.tfevents.1768638415.Dominant.2466.0
│   ├── gru
│   └── s4d
├── long_term_data
│   ├── Chewie_CO_2016
│   └── Chewie_processed
├── long_term_log
│   └── gru
├── long_term_model_gru.pth
├── possm
│   ├── __init__.py
│   ├── __pycache__
│   ├── config
│   ├── data
│   ├── models
│   └── utils
├── processed_data
│   ├── meta_data.json
│   └── sliced_trials.pt
├── pyproject.toml
├── requirements.txt
├── s4_test
│   ├── __pycache__
│   ├── data.py
│   ├── model.py
│   ├── s4_toy.py
│   ├── state-spaces
│   ├── state_spaces_s4.egg-info
│   └── train.py
├── scripts
│   ├── __pycache__
│   ├── plot_loss.py
│   └── plotting.py
├── test.ipynb
├── train
│   ├── __pycache__
│   ├── engine.py
│   ├── evaluate.py
│   ├── long_term_inference.py
│   ├── long_term_main.py
│   ├── main.py
│   └── train.py
└── uv.lock
```

长期数据评估结果: 

```bash
(possm) hyc@Dominant:/mnt/d/codefiles/python/myPOSSM/rePOSSM$ python -m train.main --eval --backbone gru
============================================================
Running POSSM with backbone = gru
Model is saved at: ./checkpoints/long_term_model_gru.pt
============================================================
Using device: cuda
Loading model from ./checkpoints/long_term_model_gru.pt...
Training Baseline (Session 0) Loaded.
--------------------------------------------------
Evaluating Session 0... R2: 0.0687 (MSE: 0.0640)
Evaluating Session 1... R2: 0.0416 (MSE: 0.0457)
Evaluating Session 2... R2: 0.0474 (MSE: 0.0458)
Evaluating Session 3... R2: -0.0119 (MSE: 0.0666)
Evaluating Session 4... R2: 0.0604 (MSE: 0.0488)
Evaluating Session 7... R2: 0.0341 (MSE: 0.0486)
Evaluating Session 11... R2: 0.0361 (MSE: 0.0527)

==================================================
Session    | R2 (Avg)   | R2 (X)     | R2 (Y)     | MSE       
--------------------------------------------------
0          | 0.0687     | -0.0037    | 0.1412     | 0.0640    
1          | 0.0416     | -0.0009    | 0.0840     | 0.0457    
2          | 0.0474     | -0.0011    | 0.0959     | 0.0458    
3          | -0.0119    | 0.0001     | -0.0238    | 0.0666    
4          | 0.0604     | 0.0078     | 0.1129     | 0.0488    
7          | 0.0341     | -0.0001    | 0.0683     | 0.0486    
11         | 0.0361     | 0.0037     | 0.0685     | 0.0527    
==================================================
```

```bash
(possm) hyc@Dominant:/mnt/d/codefiles/python/myPOSSM/rePOSSM$ python -m train.main --eval --backbone s4d
============================================================
Running POSSM with backbone = s4d
Model is saved at: ./checkpoints/long_term_model_s4d.pt
============================================================
Using device: cuda
Loading model from ./checkpoints/long_term_model_s4d.pt...
Training Baseline (Session 0) Loaded.
--------------------------------------------------
Evaluating Session 0... R2: 0.3388 (MSE: 0.0469)
Evaluating Session 1... R2: 0.1464 (MSE: 0.0404)
Evaluating Session 2... R2: -0.0129 (MSE: 0.0495)
Evaluating Session 3... R2: 0.0188 (MSE: 0.0671)
Evaluating Session 4... R2: -1.3658 (MSE: 0.1198)
Evaluating Session 7... R2: 0.0684 (MSE: 0.0471)
Evaluating Session 11... R2: -2.3554 (MSE: 0.1833)

==================================================
Session    | R2 (Avg)   | R2 (X)     | R2 (Y)     | MSE       
--------------------------------------------------
0          | 0.3388     | 0.2287     | 0.4489     | 0.0469    
1          | 0.1464     | 0.1703     | 0.1225     | 0.0404    
2          | -0.0129    | -0.2974    | 0.2717     | 0.0495    
3          | 0.0188     | -0.1006    | 0.1381     | 0.0671    
4          | -1.3658    | -0.8307    | -1.9008    | 0.1198    
7          | 0.0684     | -0.0101    | 0.1469     | 0.0471    
11         | -2.3554    | -2.4705    | -2.2404    | 0.1833    
==================================================
```



1. 数据集
2. RNN 等架构的性能对比.