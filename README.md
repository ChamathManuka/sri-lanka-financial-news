# Sri Lanka Financial News Dataset

## Overview
This dataset contains **Sri Lankan news articles collected from publicly available archives**, curated for **research and machine learning purposes**.  

It was used in my Master’s research for **news-driven stock price prediction using Graph Neural Networks**, but it is shared publicly for **educational, research, and experimentation purposes**.

---

## Contents

| Folder / File | Description |
|---------------|------------|
| `data/` | Cleaned and structured news articles in CSV / JSON format |
| `preprocessing/` | Scripts used for cleaning, tokenizing, and preparing data |
| `README.md` | This file, describing the dataset and usage |
| `LICENSE` | License governing usage of the dataset |

---

## Data Description
- Articles collected from **publicly accessible news archives** in Sri Lanka    
- Suitable for:
  - Natural Language Processing (NLP) tasks  
  - Sentiment analysis  
  - Graph Neural Network modeling  
  - Financial news prediction experiments  

---

## Usage Example

```python
import pandas as pd

# Load CSV dataset
df = pd.read_csv("data/sri_lanka_news.csv")

# Inspect first few rows
print(df.head())
```

## Author
**Chamath Gamage**  
Backend Engineer | Python | Java | ML & NLP

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.  
You are free to use and adapt this project for research and educational purposes **with proper credit**. Commercial use is not allowed.

For full license text, see [LICENSE](LICENSE).
