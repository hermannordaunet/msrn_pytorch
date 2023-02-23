# Master-Codebase

## Codebase for my master project

Folder structure:

```
├── models # trained pytorch models
│
├── reports
│   ├── figures
├── src
│   ├── models
│   │   ├── eenets.py
│   │   ├── evaluate_model.py
│   │   ├── train_model.py
│   ├── exitblock.py
├── utils
├── visualization
└── .gitignore
```

Early test for trying to get the folder structure down. The idea is to have the ability to have a pytorch model and add exit blocks to any layer.

This might need to be adjusted so that there are test at the layer we add the exit block to, but that is later.
