# Getting Started Guide 🚀

Welcome to your Machine Learning journey! This guide will help you set up everything you need to start learning.

## Step 1: Install Python

### Check if Python is installed:
```bash
python --version
# or
python3 --version
```

You need Python 3.8 or higher.

### If Python is not installed:
- **Mac**: Download from [python.org](https://www.python.org/downloads/) or use Homebrew: `brew install python3`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **Linux**: Usually pre-installed, or use: `sudo apt-get install python3`

## Step 2: Install Required Libraries

### Option A: Using pip (Recommended for beginners)
```bash
# Navigate to this directory
cd /Users/iwizards/Srinidhi/PersonalLearning/MachineLearning

# Install all required packages
pip install -r requirements.txt
```

### Option B: Using a Virtual Environment (Recommended for best practices)
```bash
# Create a virtual environment
python3 -m venv ml_env

# Activate it
# On Mac/Linux:
source ml_env/bin/activate
# On Windows:
ml_env\Scripts\activate

# Install packages
pip install -r requirements.txt
```

## Step 3: Install Jupyter Notebook

Jupyter Notebook is great for learning ML interactively:

```bash
pip install jupyter
```

### Start Jupyter:
```bash
jupyter notebook
```

This will open a browser window where you can create and run notebooks.

## Step 4: Verify Installation

Create a test file `test_setup.py`:

```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

print("✅ NumPy version:", np.__version__)
print("✅ Pandas version:", pd.__version__)
print("✅ Scikit-learn version:", sklearn.__version__)
print("\n🎉 Everything is set up correctly!")
```

Run it:
```bash
python test_setup.py
```

If you see version numbers, you're good to go!

## Step 5: Start Learning!

### Week 1-2: Foundations
1. Open `01_foundations/01_python_basics.py`
2. Complete the exercises
3. Create a Jupyter notebook to practice data manipulation

### Week 3-4: First ML Model
1. Run `02_intro_ml/01_first_ml_model.py`
2. Understand each step
3. Experiment by changing the data or parameters

### Continue with the learning path in README.md!

## Alternative: Use Google Colab (No Installation Needed!)

If you don't want to install anything locally:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Create a new notebook
4. Start coding! (Most libraries are pre-installed)

## Troubleshooting

### "pip: command not found"
- Try `pip3` instead of `pip`
- Or `python -m pip install ...`

### "Permission denied" errors
- Use `pip install --user ...` to install for your user only
- Or use a virtual environment (recommended)

### Import errors
- Make sure you activated your virtual environment (if using one)
- Try reinstalling: `pip install --upgrade package_name`

## Need Help?

- Check the README.md for learning resources
- Join ML communities (Reddit r/MachineLearning, Discord)
- Google your error messages (everyone does this!)
- Practice, practice, practice!

## Next Steps

1. ✅ Complete setup
2. ✅ Run `01_foundations/01_python_basics.py`
3. ✅ Start Phase 1: Foundations
4. ✅ Build your first project!

**You've got this! Happy learning! 🎓**

