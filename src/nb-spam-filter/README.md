# 🚀 Fast Email Spam Filter

Uses **Naive Bayes** for **ultra-fast** spam/ham classification.  
Trains in seconds, infers instantly! ⚡

## Setup 🛠️
```bash
pip install scikit-learn pandas numpy
python generate_dataset.py  # Creates data/emails.csv (2000 emails)
python email_classifier.py  # Train + benchmark
```

## Outputs 📊
- Training time: ~0.01s  
- Inference: ~1ms for 500 emails  
- Accuracy: ~95%  

Simple, deployable ML! 🎯
